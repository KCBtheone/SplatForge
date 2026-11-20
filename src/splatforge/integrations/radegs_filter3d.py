"""Owned camera-sampling 3D Gaussian convolution for the RaDe paper recipe.

The pixel variance is 0.2, so a world pixel interval T gives variance 0.2*T*T.
This scale-equivariant interpretation follows the sampling interval and fixed
projected pixel variance; the printed Mip-Splatting Eq.9 writes s/nu rather than
s/nu**2. No author implementation is used. The original 2D filter is unchanged.
Unobserved centers retain their original fields: no evidence-based bandwidth is
available there, and this component does not introduce a pruning operation.
"""
from functools import wraps

import torch

from ..studio.renderers import GaussianView

PIXEL_VARIANCE = 0.2
STATE_KIND = 'radegs-camera-sampling-filter3d'


@torch.no_grad()
def sampling_interval(means, cameras, *, near, far):
    if means.dtype != torch.float32 or means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError('camera sampling requires FP32 Gaussian means')
    if not cameras:
        raise ValueError('training cameras are required')
    rate = means.new_zeros(len(means))
    for camera in cameras:
        K = camera.K.to(means)
        view = camera.world_to_camera.to(means)
        point = means @ view[:3, :3].T + view[:3, 3]
        z = point[:, 2]
        safe = z.clamp_min(near)
        u = point[:, 0] / safe * K[0, 0] + K[0, 2]
        v = point[:, 1] / safe * K[1, 1] + K[1, 2]
        visible = ((z > near) & (z < far) & (u >= 0) & (u < camera.width)
                   & (v >= 0) & (v < camera.height))
        frequency = K[:2, :2].diagonal().max() / safe
        rate = torch.maximum(rate, torch.where(visible, frequency, 0))
    observed = rate > 0
    interval = torch.where(observed, rate.clamp_min(torch.finfo(rate.dtype).tiny).reciprocal(), 0)
    return interval, observed


def convolve_fields(scales, opacity, sigma):
    """Same raw parameters, convolved covariance and 3D mass compensation."""
    if any(value.dtype != torch.float32 for value in (scales, opacity, sigma)):
        raise ValueError('3D convolution requires FP32 activated fields and cache')
    if scales.shape != (len(sigma), 3) or opacity.shape[0] != len(sigma):
        raise ValueError('filter cache does not match the current Gaussian population')
    smoothed = torch.hypot(scales, sigma.reshape(-1, 1))
    ratio = scales / smoothed.clamp_min(torch.finfo(scales.dtype).tiny)
    coefficient = ratio.prod(-1).reshape((-1,) + (1,) * (opacity.ndim - 1))
    return smoothed, opacity * coefficient


class FilteredRenderer:
    def __init__(self, delegate, state):
        object.__setattr__(self, 'delegate', delegate)
        object.__setattr__(self, 'filter_state', state)

    def __getattr__(self, name):
        target = getattr(self.delegate, name)
        if name not in ('render', 'render_rgb', 'render_training'):
            return target
        @wraps(target)
        def render(scene, camera, *args, **kwargs):
            self.filter_state.counts['filter3d_render'] += 1
            return target(self.filter_state.apply(scene, track=True), camera, *args, **kwargs)
        return render

    def __setattr__(self, name, value):
        if name in ('delegate', 'filter_state'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.delegate, name, value)


class SamplingFilter3D:
    def __init__(self, pipeline, counts, *, step, restored=None, allow_bootstrap=False, resumed=False):
        self.pipeline, self.counts = pipeline, counts
        self.cameras = [pipeline.data.sample(i, pipeline.model.device)['camera']
                        for i in pipeline.data.train_indices]
        if len(self.cameras) != 49:
            raise ValueError('the bound original49 training cameras are required')
        self.near, self.far = pipeline.config.render.near, pipeline.config.render.far
        if restored is None:
            if resumed and not allow_bootstrap:
                raise ValueError('missing derived filter state; no silent checkpoint fallback')
            counts['filter3d_bootstrap' if resumed else 'filter3d_initialize'] += 1
            self.refresh(step)
        else:
            if (restored['kind'] != STATE_KIND or restored['pixel_variance'] != PIXEL_VARIANCE
                or restored['unobserved_policy'] != 'retain_unconstrained'
                or restored['training_views'] != 49):
                raise ValueError('saved filter semantics differ')
            self.sigma = restored['sigma'].to(pipeline.model.device)
            self.observed = restored['observed'].to(pipeline.model.device)
            self.last_refresh_step = restored['last_refresh_step']
            if self.last_refresh_step > step:
                raise ValueError('filter cache is newer than the restored model')
            self.validate()
            counts['filter3d_restore'] += 1

    @torch.no_grad()
    def refresh(self, step):
        interval, self.observed = sampling_interval(self.pipeline.model.means, self.cameras,
                                                   near=self.near, far=self.far)
        self.sigma = interval * PIXEL_VARIANCE ** .5
        self.last_refresh_step = int(step)
        self.counts['filter3d_refresh'] += 1
        self.validate()

    def validate(self):
        if (self.sigma.dtype != torch.float32 or self.observed.dtype != torch.bool
            or self.sigma.shape != (self.pipeline.model.count,)
            or self.observed.shape != self.sigma.shape
            or self.sigma.requires_grad or not bool(torch.isfinite(self.sigma).all())
            or bool((self.sigma < 0).any())):
            raise ValueError('invalid FP32 population-matched filter state')

    def apply(self, scene, *, track=False):
        scales, opacity = convolve_fields(scene.scales, scene.opacities, self.sigma)
        if track:
            self.counts['filter3d_fields_forward'] += 1
            for value, role in ((scales, 'filter3d_scaling_backward'),
                                (opacity, 'filter3d_opacity_backward')):
                if value.requires_grad:
                    def backward(gradient, key=role):
                        self.counts[key] += 1
                        return gradient
                    value.register_hook(backward)
        return GaussianView(scene.means, scene.quats, scales, opacity, scene.sh, scene.sh_degree)

    def state_dict(self):
        return dict(kind=STATE_KIND, pixel_variance=PIXEL_VARIANCE, training_views=49,
                    unobserved_policy='retain_unconstrained', sigma=self.sigma,
                    observed=self.observed, last_refresh_step=self.last_refresh_step)

    @torch.no_grad()
    def reset_filtered_opacity(self, step, optimizer):
        """Cap the rendered peak after population edits and bandwidth refresh."""
        model = self.pipeline.model
        parameter = model.params['opacity_logits']
        _, coefficient = convolve_fields(model.scales, torch.ones_like(parameter), self.sigma)
        before = parameter.sigmoid() * coefficient
        cap = parameter.new_tensor(self.pipeline.densifier.opacity_reset_max)
        # rho <= cap needs no clipping. Work in logits so rho ~= cap does not
        # introduce logit(1), and keep every internal/reset tensor finite FP32.
        limit = torch.where(coefficient > cap,
                            cap.log() - (coefficient - cap).clamp_min(torch.finfo(parameter.dtype).tiny).log(),
                            torch.full_like(parameter, torch.finfo(parameter.dtype).max))
        parameter.clamp_(max=limit)
        state = optimizer.state.get(parameter, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key].zero_()
        after = parameter.sigmoid() * coefficient
        expected = before.clamp_max(cap)
        error = (after - expected).abs()
        bad = int((error > 3e-5 + 3e-4 * expected.abs()).sum())
        lost = int(((before >= 1 / 255) & (after < 1 / 255)).sum())
        finite = all(bool(torch.isfinite(value).all()) for value in (parameter, coefficient, after, limit))
        moment_max = max((float(state[k].abs().max()) for k in ('exp_avg', 'exp_avg_sq') if k in state), default=0.)
        self.last_opacity_reset = dict(step=int(step), population=model.count,
            filter_refresh_step=self.last_refresh_step, raw_opacity_max=float(parameter.sigmoid().max()),
            effective_opacity_max=float(after.max()), effective_reset_max_error=float(error.max()),
            effective_reset_bad_count=bad, possible_support_before=int((before >= 1 / 255).sum()),
            possible_support_after=int((after >= 1 / 255).sum()), lost_possible_support=lost,
            opacity_moment_max=moment_max, Adam_step=float(state['step']) if 'step' in state else None,
            finite=finite, model_io_fp32=parameter.dtype == torch.float32,
            passed=bool(finite and bad == 0 and lost == 0 and moment_max <= 3e-5
                        and float(after.max()) <= float(cap) + 3e-5))
        self.counts['filtered_opacity_reset'] += 1
        self.counts['filtered_opacity_reset_guard'] += 1
        if not self.last_opacity_reset['passed']:
            raise ValueError('Filtered opacity reset violated the rendered peak/Adam support boundary')

    def attach(self, renderer, *, filtered_opacity_reset=False):
        if filtered_opacity_reset:
            if self.pipeline.densifier.opacity_reset_max != .01:
                raise ValueError('The bound filtered-opacity reset maximum is .01')
            self.pipeline.densifier.defer_opacity_reset = True
        self.pipeline.renderer = FilteredRenderer(renderer, self)
        original_state = self.pipeline.state_dict
        original_after = self.pipeline.after_step
        def state_dict():
            state = original_state()
            state['radegs_filter3d'] = self.state_dict()
            return state
        def after_step(step, optimizer):
            result = original_after(step, optimizer)
            if step % 100 == 0 or len(self.sigma) != self.pipeline.model.count:
                self.refresh(step)
            density = self.pipeline.densifier.config if filtered_opacity_reset else None
            if (density is not None and density.enabled and 0 < step < density.stop
                    and density.reset_opacity_every > 0 and step % density.reset_opacity_every == 0):
                self.reset_filtered_opacity(step, optimizer)
            return result
        self.pipeline.state_dict = state_dict
        self.pipeline.after_step = after_step


def primitive_guard(entry, model, state, counts):
    """Independent double convolution/determinant reference; original tolerances."""
    raw = (model.params['log_scales'], model.params['opacity_logits'])
    versions = [value._version for value in raw]
    scales, opacity = convolve_fields(raw[0].exp(), raw[1].sigmoid(), state.sigma)
    cot = scales.new_tensor([.3, -.7, .2])
    actual = torch.autograd.grad((scales * cot).sum() - .4 * opacity.sum(), raw)
    shadow = [value.detach().clone().requires_grad_(True) for value in raw]
    s = shadow[0].exp().cpu().double()
    a = shadow[1].sigmoid().cpu().double()
    variance = state.sigma.cpu().double().square()[:, None]
    reference_scales = (s.square() + variance).sqrt()
    coefficient = (s.square().prod(-1) / (s.square() + variance).prod(-1)).sqrt()
    reference_opacity = a * coefficient.reshape_as(a)
    expected = torch.autograd.grad((reference_scales * cot.cpu().double()).sum()
                                   - .4 * reference_opacity.sum(), shadow)
    pairs = [('scales', scales, reference_scales), ('opacity', opacity, reference_opacity),
             *zip(('log_scales', 'opacity_logits'), actual, expected)]
    errors = {name: entry.depth_field_error(x.detach().cpu().double(), y.detach().cpu().double())
              for name, x, y in pairs}
    readonly = versions == [value._version for value in raw] and all(value.grad is None for value in raw)
    counts['filter3d_operator_guard'] += 1
    return dict(passed=readonly and all(value['passed'] for value in errors.values()),
                errors=errors, readonly=readonly, rows=model.count,
                observed_centers=int(state.observed.sum()),
                scope='Full FP32 activated-field convolution F/true VJP; independent CPU double determinant reference only scores, no model update')
