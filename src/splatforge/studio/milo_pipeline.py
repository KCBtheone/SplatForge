"""MILo representation callbacks consumed by the existing normal Trainer."""

from itertools import chain

import torch

from .losses import photometric_loss
from .milo_mesh import gaussian_pivots, gaussian_pivot_mesh, triangulate_pivots_scipy_cpu_reference
from .milo_population import (
    clone_critical_gaussians, critical_gaussians, gaussian_contributions,
    reinitialize_gaussians, sample_depth_points, sample_important_gaussians,
)
from .milo_raster_reference import IndependentMeshRenderer
from .milo_state import MiloMeshState
from .milo_supervision import (
    DepthObservation, fuse_signed_distances, mesh_consistency_losses,
    mesh_interior_labels, raw_sdf_initialization, volumetric_normal_loss,
)
from .pipeline import GaussianPipeline
from .renderers import GaussianView


class MiloPipeline(GaussianPipeline):
    def __init__(self, config, method):
        super().__init__(config, method)
        self.mesh_state = MiloMeshState(device=self.model.device, dtype=self.model.means.dtype)
        self.mesh_renderer = IndependentMeshRenderer(config.render.near, config.render.far, config.render.tile_size)
        self.completed_step = 0
        self._loss_step = 0
        self.mesh_initialization = None
        self.population_events = {'aggressive_clone': 0, 'depth_reinitialization': 0, 'base_prune': 0}

    def parameters(self):
        return chain(self.model.parameters(), self.mesh_state.parameters())

    def parameter_groups(self):
        return [*super().parameter_groups(), self.mesh_state.parameter_group()]

    def training_loss(self, sample, step):
        if (step >= 8000) != self.mesh_state.active:
            raise ValueError('MILo mesh activation must match the actual 8000-step state boundary')
        self._loss_step = step
        # MILo Eq. (3) has no depth-distortion objective. The original batched
        # API still computes depth/normals when geometric=False; it disables
        # only the unused L1/L2 distortion outputs.
        render_training = getattr(self.renderer, 'render_training', None)
        output = (render_training(GaussianView.from_model(self.model), sample['camera'], geometric=False)
            if render_training is not None else self.render(sample['camera']))
        prediction = self.model.compensate(output.rgb, sample['index'])
        # All image terms use the same sum over pixels. Color/SSIM channels are
        # averaged by the owned photometric primitive; this reduction is saved
        # in config rather than silently rescaling only the mesh terms.
        terms = {'photometric': photometric_loss(prediction, sample['image'], .2)
            * sample['camera'].width * sample['camera'].height}
        if step >= 3000:
            normal, valid = self.ops('geometry.depth_normals@1', output.depth.squeeze(-1), sample['camera'].K)
            valid = valid & (sample['mask'].squeeze(-1) > 0)
            terms['volumetric_normal'] = .05 * volumetric_normal_loss(output.normals, normal, valid)
            if step >= 8000:
                mesh = self.mesh_state(self.model.means, self.model.quats, self.model.params['log_scales'])
                mesh_render = self.mesh_renderer.render(mesh, sample['camera'])
                terms.update(mesh_consistency_losses(output.depth, normal, mesh_render, valid))
                terms.update(self.mesh_state.regularization())
        return sum(terms.values()), terms, output

    def after_backward(self, output):
        if self._loss_step < 3000:
            self.densifier.observe(output, self.model)

    def _training_samples(self):
        for index in self.data.train_indices:
            yield self.data.sample(index, self.model.device)

    @torch.no_grad()
    def _contributions(self, *, include_depth=False):
        for sample in self._training_samples():
            contribution = gaussian_contributions(self.model, sample['camera'], self.config.render,
                self.ops, include_depth=include_depth)
            yield sample, contribution

    @torch.no_grad()
    def _aggressive_clone(self, optimizer):
        strength = self.model.means.new_zeros(self.model.count)
        for _, contribution in self._contributions():
            strength = torch.maximum(strength, contribution.critical_strength)
        ids = critical_gaussians(strength)
        added = clone_critical_gaussians(self.model, self.densifier, optimizer, ids)
        self.population_events['aggressive_clone'] += 1
        return added

    @torch.no_grad()
    def _depth_reinitialize(self, optimizer):
        samples = ((sample['camera'], sample['image'], contribution)
            for sample, contribution in self._contributions(include_depth=True))
        points, colors = sample_depth_points(samples, self.model.count, seed=self.config.train.seed)
        reinitialize_gaussians(self.model, self.densifier, optimizer, points, colors)
        self.population_events['depth_reinitialization'] += 1

    @torch.no_grad()
    def _gaussian_depth_observations(self, scene=None):
        for sample in self._training_samples():
            output = (self.render(sample['camera']) if scene is None
                else self.renderer.render(scene, sample['camera']))
            depth = output.depth.detach()
            valid = torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0) & (output.alpha[..., 0] > 0)
            yield DepthObservation(sample['camera'], depth, valid)

    @torch.no_grad()
    def _mesh_depth_observations(self, mesh=None):
        if mesh is None:
            mesh = self.mesh_state(self.model.means, self.model.quats, self.model.params['log_scales'])
        for sample in self._training_samples():
            rendered = self.mesh_renderer.render(mesh, sample['camera'])
            yield DepthObservation(sample['camera'], rendered.depth, rendered.valid)

    @torch.no_grad()
    def _initialize_mesh(self, optimizer, step):
        importance = self.model.means.new_zeros(self.model.count)
        pixels = 0
        for sample, contribution in self._contributions():
            importance += contribution.importance
            pixels += sample['camera'].width * sample['camera'].height
        selected = sample_important_gaussians(importance/pixels,
            self.config.milo.pivot_gaussians, seed=self.config.train.seed)
        old_count = self.model.count
        means, quats = self.model.means[selected], self.model.quats[selected]
        log_scales = self.model.params['log_scales'][selected]
        scene = GaussianView(means, quats, log_scales.exp(), self.model.opacities[selected],
            self.model.sh[selected], int(self.model.active_sh_degree))
        sites = gaussian_pivots(means, quats, log_scales).reshape(-1, 3)
        distances, observations = fuse_signed_distances(sites, self._gaussian_depth_observations(scene))
        seen = observations > 0
        if not bool(seen.any()):
            raise ValueError('no selected MILo site has an input-rendered depth observation')
        scale = distances[seen].abs().max()
        if not bool(torch.isfinite(scale) & (scale > 0)):
            raise ValueError('MILo observed signed-distance initialization has zero/nonfinite scale')
        # One global normalization preserves Eq. (2) interpolation ratios.
        # Unobserved sites are explicitly exterior, never guessed interior.
        initialized = torch.where(seen, distances, scale)
        raw_sdf = raw_sdf_initialization(initialized, scale).reshape(-1, 9)
        cells = triangulate_pivots_scipy_cpu_reference(sites)
        prepared_mesh = gaussian_pivot_mesh(means, quats, log_scales, raw_sdf, cells)
        if not len(prepared_mesh.faces):
            raise ValueError('MILo input-rendered SDF initialization has no surface')
        # Raster/dependency/topology failures occur before population or Adam
        # mutation, preserving the last volume state for a diagnosed repair.
        interior = mesh_interior_labels(sites, self._mesh_depth_observations(prepared_mesh)).reshape(-1, 9)
        group = [g for g in optimizer.param_groups if g.get('name') == 'milo_raw_sdf']
        if len(group) != 1 or len(group[0]['params']) != 1 or group[0]['params'][0] is not self.mesh_state.raw_sdf:
            raise ValueError('MILo initialization requires its bound empty SDF optimizer group')
        self.densifier._replace(self.model, optimizer, selected, selected.new_empty(0))
        ids = torch.arange(self.model.count, device=self.model.device)
        self.mesh_state.initialize(ids, raw_sdf, cells, step=step, optimizer=optimizer)
        self.mesh_state.interior.copy_(interior)
        self.mesh_state.interior_step.fill_(step)
        self.mesh_initialization = {'step': step, 'gaussians_before': old_count,
            'pivot_gaussians': self.model.count, 'observed_sites': int(seen.sum()),
            'unobserved_exterior_sites': int((~seen).sum()), 'distance_scale': float(scale),
            'scale_policy': self.config.milo.sdf_initialization_scale,
            'topology_backend': self.config.milo.topology_backend}
        self.population_events['base_prune'] += 1

    @torch.no_grad()
    def after_step(self, step, optimizer):
        if step <= self.completed_step:
            raise ValueError('MILo population callbacks require a new completed optimizer step')
        if step < 3000:
            self.densifier.after_step(step, self.model, optimizer)
            if step >= 500 and step % 250 == 0:
                self._aggressive_clone(optimizer)
            if step == 2000:
                self._depth_reinitialize(optimizer)
        if step == 8000:
            self._initialize_mesh(optimizer, step)
        if step >= 8000:
            if not self.mesh_state.active:
                raise ValueError('MILo cannot skip its mesh initialization boundary')
            self.mesh_state.refresh_topology(self.model.means, self.model.quats,
                self.model.params['log_scales'], step)
            if int(self.mesh_state.interior_step) < 0 or step-int(self.mesh_state.interior_step) >= 200:
                self.mesh_state.refresh_interior(self.model.means, self.model.quats,
                    self.model.params['log_scales'], self._mesh_depth_observations(), step)
        self.completed_step = step

    def statistics(self):
        return {**super().statistics(), 'mesh_active': self.mesh_state.active,
            'mesh_pivots': len(self.mesh_state.pivot_ids), 'tetrahedra': len(self.mesh_state.cells),
            'phase': 'photometric' if self.completed_step < 3000 else ('volumetric' if self.completed_step < 8000 else 'mesh')}

    def state_dict(self):
        return {**super().state_dict(), 'milo': {'mesh': self.mesh_state.state_dict(),
            'completed_step': self.completed_step, 'initialization': self.mesh_initialization,
            'population_events': dict(self.population_events)}}

    def load_state_dict(self, state):
        if 'milo' not in state:
            raise ValueError('a Gaussian-only checkpoint is not a MILo mesh/training state')
        super().load_state_dict(state)
        own = state['milo']
        self.mesh_state.load_state_dict(own['mesh'])
        self.completed_step = own['completed_step']
        self.mesh_initialization = own['initialization']
        self.population_events = dict(own['population_events'])
        if (self.completed_step >= 8000) != self.mesh_state.active:
            raise ValueError('restored MILo stage and mesh activation disagree')
        if self.mesh_state.active and int(self.mesh_state.pivot_ids.max()) >= self.model.count:
            raise ValueError('restored MILo pivots address a different Gaussian population')
