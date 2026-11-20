"""MILo's input-only importance and Mini-Splatting2 population operations.

The contribution pass uses the owned RaDe-GS reference footprint, order and
alpha compositor. It is a bounded eager reference, not an accelerated renderer.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .model import GaussianScene
from .renderers import GaussianView, project_geometry
from .tile_binning import tile_members


@dataclass
class GaussianContributions:
    importance: torch.Tensor
    critical_strength: torch.Tensor
    depth: torch.Tensor | None
    valid: torch.Tensor


@torch.no_grad()
def gaussian_contributions(model, camera, config, ops, *, include_depth=False):
    """Sum T*alpha and find each pixel's maximum-weight Gaussian.

    Optional depth is the midpoint of the winning ray/ellipsoid intersection,
    Mini-Splatting2 Eq. (2), in camera z units. No image or GT is needed.
    """
    scene = GaussianView.from_model(model)
    means, _, cov_camera, jacobian, xy, screen, filtered = project_geometry(
        scene, camera, ops, config.eps2d, config.near)
    inverse = torch.linalg.inv(filtered)
    radii = config.sigma_extent * torch.linalg.eigvalsh(filtered)[:, -1].clamp_min(0).sqrt()
    radial = F.normalize(means, dim=-1)
    transform = torch.cat((jacobian, radial[:, None]), 1)
    covariance = transform @ cov_camera @ transform.transpose(-1, -2)
    gain = (covariance[:, 2:3, :2] @ torch.linalg.inv(covariance[:, :2, :2])).squeeze(1)
    opacity = scene.opacities
    if config.antialiased:
        opacity = opacity * (torch.linalg.det(screen).clamp_min(0)
            / torch.linalg.det(filtered).clamp_min(1e-12)).sqrt()
    bounds = torch.cat((xy, radii[:, None], means[:, 2:3]), 1).cpu().numpy()
    members = tile_members(bounds, width=camera.width, height=camera.height,
        tile_size=config.tile_size, near=config.near, far=config.far)
    rays = ops("camera.pixel_rays@1", camera.K, width=camera.width, height=camera.height)
    importance = model.means.new_zeros(model.count)
    critical = torch.zeros_like(importance)
    valid_map = torch.zeros((camera.height, camera.width), dtype=torch.bool, device=model.device)
    depth_map = model.means.new_zeros(camera.height, camera.width, 1) if include_depth else None
    inverse_cov = torch.linalg.inv(cov_camera) if include_depth else None
    columns = math.ceil(camera.width / config.tile_size)
    for tile_id, raw_ids in enumerate(members):
        if not len(raw_ids):
            continue
        top, left = (tile_id // columns) * config.tile_size, (tile_id % columns) * config.tile_size
        h, w = min(config.tile_size, camera.height-top), min(config.tile_size, camera.width-left)
        ray = rays[top:top+h, left:left+w].reshape(-1, 3)
        pixels = (ray @ camera.K.T)[:, :2]
        transmittance = ray.new_ones(len(ray))
        best_weight = ray.new_zeros(len(ray))
        best_id = torch.full((len(ray),), -1, dtype=torch.long, device=model.device)
        chunk = max(1, config.tile_event_budget // len(ray))
        for offset in range(0, len(raw_ids), chunk):
            ids = torch.as_tensor(raw_ids[offset:offset+chunk], device=model.device)
            delta = pixels[None] - xy[ids, None]
            quadratic = torch.einsum('npi,nij,npj->np', delta, inverse[ids], delta)
            radial_depth = torch.linalg.vector_norm(means[ids], dim=-1)[:, None] + (gain[ids, None]*delta).sum(-1)
            z = radial_depth / torch.linalg.vector_norm(ray, dim=-1)
            support = (quadratic <= config.sigma_extent**2) & (z > config.near) & (z < config.far)
            alpha = (opacity[ids, None] * torch.where(support, torch.exp(-.5*quadratic), 0)).clamp(0, .999)
            prefix = torch.cumprod(torch.cat((alpha.new_ones(1, len(ray)), 1-alpha), 0), 0)
            weights = alpha * prefix[:-1] * transmittance[None]
            importance.index_add_(0, ids, weights.sum(-1))
            value, local = weights.max(0)
            replace = value > best_weight  # equal contributions keep the stable first event
            best_id = torch.where(replace, ids[local], best_id)
            best_weight = torch.maximum(best_weight, value)
            transmittance *= prefix[-1]
        present = best_id >= 0
        critical.scatter_reduce_(0, best_id[present], best_weight[present], reduce='amax', include_self=True)
        valid_map[top:top+h, left:left+w] = present.reshape(h, w)
        if include_depth:
            selected = best_id[present]
            direction = ray[present]
            product = (inverse_cov[selected] @ direction[..., None]).squeeze(-1)
            midpoint = (product * means[selected]).sum(-1) / (product * direction).sum(-1)
            depth = ray.new_zeros(len(ray))
            depth[present] = midpoint
            depth_map[top:top+h, left:left+w, 0] = depth.reshape(h, w)
    if not bool(torch.isfinite(importance).all() & torch.isfinite(critical).all()):
        raise FloatingPointError('non-finite MILo rendering contributions')
    return GaussianContributions(importance, critical, depth_map, valid_map)


def critical_gaussians(strength, *, keep_quantile=.99):
    """Keep 99% of positive pixel winners, removing the weakest 1%.

    The paper calls .99 a keep quantile. This explicit lower-tail convention
    avoids silently interpreting it as retaining only the strongest 1%.
    """
    if not 0 < keep_quantile <= 1:
        raise ValueError('critical keep quantile must be in (0,1]')
    positive = (strength > 0).nonzero().flatten()
    if not len(positive):
        return positive
    threshold = torch.quantile(strength[positive], 1-keep_quantile)
    return positive[strength[positive] >= threshold]


@torch.no_grad()
def sample_important_gaussians(importance, count, *, seed):
    if type(count) is not int or count < 1 or not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()):
        raise ValueError('finite nonnegative importance and a positive sample count required')
    positive = (importance > 0).nonzero().flatten()
    if not len(positive):
        raise ValueError('no Gaussian contributes to a training view')
    if len(positive) <= count:
        return positive
    generator = torch.Generator(device=importance.device).manual_seed(seed)
    chosen = torch.multinomial(importance[positive], count, replacement=False, generator=generator)
    return positive[chosen].sort().values


@torch.no_grad()
def clone_critical_gaussians(model, densifier, optimizer, critical_ids):
    """Two coincident clones with the paper's opacity/covariance transition."""
    if not len(critical_ids):
        return 0
    critical_ids = critical_ids.unique(sorted=True)
    old_logit = model.params['opacity_logits'][critical_ids].detach()
    log_complement = -.5 * F.softplus(old_logit)
    new_opacity = -torch.expm1(log_complement)
    new_logit = new_opacity.log() - log_complement
    ratio = old_logit.sigmoid() / (2*new_opacity - new_opacity.square()/math.sqrt(2))
    new_scales = model.params['log_scales'][critical_ids].detach() + ratio.log()[:, None]
    previous_count = model.count
    keep = torch.arange(previous_count, device=model.device)
    densifier._replace(model, optimizer, keep, critical_ids,
        child_means=model.means.detach()[critical_ids].clone(), child_log_scales=new_scales)
    children = torch.arange(previous_count, model.count, device=model.device)
    for ids in (critical_ids, children):
        model.params['opacity_logits'][ids] = new_logit
        model.params['log_scales'][ids] = new_scales
    return len(critical_ids)


@torch.no_grad()
def sample_depth_points(contribution_samples, count, *, seed):
    """Uniform reservoir of input-colored winning-ray midpoint points.

    Samples stream one original training view at a time. At most count plus
    one view of positions/colors is retained; no all-view event tensor exists.
    """
    points = colors = keys = None
    generator = None
    for camera, image, contribution in contribution_samples:
        if contribution.depth is None:
            raise ValueError('depth reinitialization requires winning Gaussian midpoints')
        valid = contribution.valid & torch.isfinite(contribution.depth[..., 0]) & (contribution.depth[..., 0] > 0)
        y, x = valid.nonzero(as_tuple=True)
        if not len(y):
            continue
        pixel = torch.stack((x.to(image.dtype)+.5, y.to(image.dtype)+.5, torch.ones_like(x, dtype=image.dtype)), -1)
        ray = pixel @ torch.linalg.inv(camera.K).T
        view = ray * contribution.depth[y, x]
        pose = camera.camera_to_world
        world = view @ pose[:3, :3].T + pose[:3, 3]
        if generator is None:
            generator = torch.Generator(device=world.device).manual_seed(seed)
        priority = torch.rand(len(world), device=world.device, generator=generator)
        points = world if points is None else torch.cat((points, world))
        colors = image[y, x] if colors is None else torch.cat((colors, image[y, x]))
        keys = priority if keys is None else torch.cat((keys, priority))
        if len(points) > count:
            chosen = keys.topk(count, sorted=False).indices
            points, colors, keys = points[chosen], colors[chosen], keys[chosen]
    if points is None or len(points) < count:
        raise ValueError('not enough observed depth pixels for the original Gaussian population')
    return points, colors


@torch.no_grad()
def reinitialize_gaussians(model, densifier, optimizer, points, colors):
    """A new depth-point representation starts its own Gaussian Adam moments."""
    replacement = GaussianScene(points, colors, model.config, len(model.appearance))
    replacement.active_sh_degree.copy_(model.active_sh_degree)
    groups = {g.get('name'): g for g in optimizer.param_groups}
    for name, parameter in model.params.items():
        if name not in groups or len(groups[name]['params']) != 1 or groups[name]['params'][0] is not parameter:
            raise ValueError('depth reinitialization needs the bound named Gaussian optimizer groups')
    for name, parameter in list(model.params.items()):
        optimizer.state.pop(parameter, None)
        model.params[name] = replacement.params[name]
        groups[name]['params'][0] = model.params[name]
    densifier.gradient = points.new_zeros(model.count)
    densifier.observations = points.new_zeros(model.count)
