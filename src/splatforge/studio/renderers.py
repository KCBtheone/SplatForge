"""Renderer protocols and a tiled, differentiable paper reference.

Tile selection is a discrete eager operation with an explicit host copy of
bounds. Numerical operators and their gradients stay on the model device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from ..gaussian import SurfaceOperators
from ..providers.torch.gaussian.reference import rotation_matrix
from .plugins import ComponentRegistry


@dataclass
class GaussianView:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor
    sh_degree: int

    @classmethod
    def from_model(cls, model):
        return cls(
            model.means,
            model.quats,
            model.scales,
            model.opacities,
            model.sh,
            int(model.active_sh_degree),
        )

    def isolated_geometry(self, selected):
        def isolate(tensor):
            shape = (len(selected),) + (1,) * (tensor.ndim - 1)
            return torch.where(selected.reshape(shape), tensor, tensor.detach())

        return GaussianView(
            isolate(self.means),
            isolate(self.quats),
            self.scales.detach(),
            self.opacities.detach(),
            self.sh.detach(),
            self.sh_degree,
        )


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    normals: torch.Tensor
    distortion_l1: torch.Tensor | None
    distortion_l2: torch.Tensor
    median_depth: torch.Tensor
    ray_color: torch.Tensor
    ambiguity_mask: torch.Tensor
    projected_means: torch.Tensor | None = None
    projected_radii: torch.Tensor | None = None
    normal_numerator: torch.Tensor | None = None
    distortion_profile: str = "both"


class Renderer(Protocol):
    def render(self, scene: GaussianView, camera, *, selected=None) -> RenderOutput: ...


def project_geometry(scene, camera, ops, eps2d, near=0.01):
    rotation = camera.world_to_camera[:3, :3]
    means = scene.means @ rotation.T + camera.world_to_camera[:3, 3]
    cov = ops("gaussian.covariance@1", scene.quats, scene.scales)
    cov_camera = rotation @ cov @ rotation.T
    x, y, z = means.unbind(-1)
    # Culled centers must not create enormous/rank-deficient Jacobians.
    # Their arithmetic still appears in a batched backward with zero adjoints;
    # 0*Inf would otherwise poison gradients of entirely invisible Gaussians.
    active = z > near
    safe = torch.where(active, z, torch.ones_like(z))
    x = torch.where(active, x, torch.zeros_like(x))
    y = torch.where(active, y, torch.zeros_like(y))
    zero = torch.zeros_like(z)
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    jacobian = torch.stack(
        (
            fx / safe,
            zero,
            -fx * x / safe.square(),
            zero,
            fy / safe,
            -fy * y / safe.square(),
        ),
        -1,
    ).reshape(-1, 2, 3)
    means2d = torch.stack(
        (fx * x / safe + camera.K[0, 2], fy * y / safe + camera.K[1, 2]), -1
    )
    screen_cov = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
    filtered = screen_cov + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d
    return means, cov, cov_camera, jacobian, means2d, screen_cov, filtered


class TorchRenderer:
    def __init__(self, config, *, mode, trunc_sigma=2.0, ops=None):
        self.config = config
        self.mode = mode
        self.trunc_sigma = trunc_sigma
        self.ops = ops or SurfaceOperators()

    def render(self, scene, camera, *, selected=None):
        config = self.config
        ops = self.ops
        means, cov, cov_camera, J, means2d, screen_cov, filtered = project_geometry(
            scene, camera, ops, config.eps2d, config.near
        )
        if means2d.requires_grad:
            means2d.retain_grad()
            means2d.absgrad = torch.zeros_like(means2d)
        inverse_screen = torch.linalg.inv(filtered)
        extent = self.trunc_sigma if self.mode == "ambisur" else config.sigma_extent
        radii = extent * torch.linalg.eigvalsh(filtered)[:, -1].clamp_min(0).sqrt()
        opacity = scene.opacities
        if config.antialiased:
            opacity = (
                opacity
                * (
                    torch.linalg.det(screen_cov).clamp_min(0)
                    / torch.linalg.det(filtered).clamp_min(1e-12)
                ).sqrt()
            )
        colors = ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )
        ray_colors = (
            ops(
                "gaussian.sh_color@1",
                scene.sh,
                scene.means.detach() - camera.origin,
                degree=scene.sh_degree,
            )
            if self.mode == "ambisur"
            else colors
        )
        rays = ops(
            "camera.pixel_rays@1", camera.K, width=camera.width, height=camera.height
        )
        if self.mode == "radegs":
            radial = F.normalize(means, dim=-1)
            fullJ = torch.cat((J, radial[:, None, :]), 1)
            transformed = fullJ @ cov_camera @ fullJ.transpose(-1, -2)
            # Conditional mean of radial distance in locally affine ray space.
            gain = (
                transformed[:, 2:3, :2] @ torch.linalg.inv(transformed[:, :2, :2])
            ).squeeze(1)
            ray_normal = torch.cat((gain, -torch.ones_like(gain[:, :1])), 1)
            gaussian_normals = F.normalize(
                (fullJ.transpose(-1, -2) @ ray_normal.unsqueeze(-1)).squeeze(-1), dim=-1
            )
        elif self.mode == "ambisur":
            rotation = rotation_matrix(scene.quats)
            axis = scene.scales.detach().argmin(-1)
            world_normal = rotation.gather(
                2, axis[:, None, None].expand(-1, 3, 1)
            ).squeeze(-1)
            gaussian_normals = world_normal @ camera.world_to_camera[:3, :3].T
            gaussian_normals = torch.where(
                (gaussian_normals * means).sum(-1, keepdim=True) > 0,
                -gaussian_normals,
                gaussian_normals,
            )
            plane_distance = (gaussian_normals * means).sum(-1)
        elif self.mode == "gof":
            inverse_cov = torch.linalg.inv(cov_camera)
            # An exact project/unproject composition exposes view-space means
            # for densification without replacing the ray-Gaussian geometry.
            proxy_xy = (
                (means2d - camera.K[:2, 2])
                * means[:, 2:3]
                / torch.stack((camera.K[0, 0], camera.K[1, 1]))
            )
            proxy_means = torch.cat((proxy_xy, means[:, 2:3]), -1)
        else:
            raise ValueError(f"unsupported Torch rendering semantics: {self.mode}")
        bounds = (
            torch.cat(
                (means2d.detach(), radii.detach()[:, None], means.detach()[:, 2:3]), 1
            )
            .cpu()
            .numpy()
        )
        tile = config.tile_size
        images = {
            name: []
            for name in (
                "rgb",
                "alpha",
                "depth",
                "normals",
                "distortion_l1",
                "distortion_l2",
                "median_depth",
                "ray_color",
                "ambiguity_mask",
            )
        }
        for top in range(0, camera.height, tile):
            row = {name: [] for name in images}
            height = min(tile, camera.height - top)
            for left in range(0, camera.width, tile):
                width = min(tile, camera.width - left)
                visible = (
                    (bounds[:, 0] + bounds[:, 2] >= left)
                    & (bounds[:, 0] - bounds[:, 2] <= left + width)
                    & (bounds[:, 1] + bounds[:, 2] >= top)
                    & (bounds[:, 1] - bounds[:, 2] <= top + height)
                    & (bounds[:, 3] > config.near)
                    & (bounds[:, 3] < config.far)
                )
                indices = torch.as_tensor(
                    visible.nonzero()[0], device=scene.means.device, dtype=torch.long
                )
                indices = indices[means[indices, 2].argsort(stable=True)]
                tile_rays = rays[top : top + height, left : left + width].reshape(-1, 3)
                pixels = (tile_rays @ camera.K.T)[:, :2]
                delta = pixels.unsqueeze(0) - means2d[indices, None]
                squared = torch.einsum(
                    "npi,nij,npj->np", delta, inverse_screen[indices], delta
                )
                if self.mode == "gof":
                    pixel_means = proxy_means[indices, None, :].expand(
                        -1, len(tile_rays), -1
                    )
                    if pixel_means.requires_grad:
                        z = means[indices, 2].detach()
                        focal = torch.stack((camera.K[0, 0], camera.K[1, 1]))

                        def accumulate(gradient, ids=indices, depths=z, target=means2d):
                            value = (
                                (gradient[..., :2] * depths[:, None, None] / focal)
                                .abs()
                                .sum(1)
                            )
                            target.absgrad.index_add_(0, ids, value)

                        pixel_means.register_hook(accumulate)
                    qd = torch.einsum("nij,pj->npi", inverse_cov[indices], tile_rays)
                    denominator = (qd * tile_rays).sum(-1).clamp_min(1e-12)
                    depth = (pixel_means * qd).sum(-1) / denominator
                    offset = depth[..., None] * tile_rays - pixel_means
                    exponent = torch.einsum(
                        "npi,nij,npj->np", offset, inverse_cov[indices], offset
                    )
                    response = torch.exp(-0.5 * exponent.clamp_min(0))
                    normals = F.normalize(-qd, dim=-1)
                else:
                    if delta.requires_grad:

                        def accumulate(gradient, ids=indices, target=means2d):
                            target.absgrad.index_add_(0, ids, gradient.abs().sum(1))

                        delta.register_hook(accumulate)
                    response = torch.exp(-0.5 * squared)
                    normals = gaussian_normals[indices, None, :].expand(
                        -1, len(tile_rays), -1
                    )
                    if self.mode == "radegs":
                        radial_depth = torch.linalg.vector_norm(means[indices], dim=-1)[
                            :, None
                        ] + (gain[indices, None] * delta).sum(-1)
                        depth = radial_depth / torch.linalg.vector_norm(
                            tile_rays, dim=-1
                        )
                    else:
                        denominator = (normals * tile_rays).sum(-1)
                        safe = torch.where(
                            denominator.abs() > 1e-8,
                            denominator,
                            torch.ones_like(denominator),
                        )
                        depth = plane_distance[indices, None] / safe
                        response = torch.where(denominator.abs() > 1e-8, response, 0)
                response = torch.where(
                    (squared <= extent**2)
                    & (depth > config.near)
                    & (depth < config.far),
                    response,
                    0,
                )
                alphas = opacity[indices, None] * response
                rgb, alpha, znum, nnum, l1, l2, median, weights = ops(
                    "splat.alpha_composite@1", alphas, colors[indices], depth, normals
                )
                expected = znum / alpha.clamp_min(1e-8)
                if self.mode == "ambisur":
                    dnum = (weights * plane_distance[indices, None]).sum(0)
                    denominator = (nnum * tile_rays).sum(-1)
                    expected = torch.where(
                        denominator.abs() > 1e-8,
                        dnum
                        / torch.where(
                            denominator.abs() > 1e-8,
                            denominator,
                            torch.ones_like(denominator),
                        ),
                        0,
                    )
                expected = torch.where(alpha > 1e-8, expected, 0)
                ray_color = (
                    ops(
                        "gaussian.ray_color_variance@1",
                        weights,
                        ray_colors[indices],
                        rgb,
                    )
                    if self.mode == "ambisur"
                    else torch.zeros_like(alpha)
                )
                ambiguity = (
                    (weights * selected[indices, None].to(weights.dtype)).sum(0)
                    if selected is not None
                    else torch.zeros_like(alpha)
                )
                background = rgb.new_tensor(config.background)
                values = {
                    "rgb": rgb + (1 - alpha[:, None]) * background,
                    "alpha": alpha[:, None],
                    "depth": expected[:, None],
                    "normals": F.normalize(nnum, dim=-1),
                    "distortion_l1": l1[:, None],
                    "distortion_l2": l2[:, None],
                    "median_depth": median[:, None],
                    "ray_color": ray_color[:, None],
                    "ambiguity_mask": ambiguity[:, None],
                }
                for name, value in values.items():
                    row[name].append(value.reshape(height, width, -1))
            for name in row:
                images[name].append(torch.cat(row[name], 1))
        return RenderOutput(
            **{name: torch.cat(value, 0) for name, value in images.items()},
            projected_means=means2d,
            projected_radii=radii,
        )


renderers = ComponentRegistry("splatforge.renderers")
renderers.register("torch", TorchRenderer)


def _batched_factory(config, **kwargs):
    from .batched_renderer import BatchedTorchRenderer

    return BatchedTorchRenderer(config, **kwargs)


renderers.register("torch-batched", _batched_factory)


def _cuda_composite_factory(config, **kwargs):
    from ..gaussian.ops import create_surface_runtime
    from ..providers.torch.gaussian.cuda_composite import implementation

    candidate, capability = implementation()
    runtime = create_surface_runtime(
        implementations=[candidate], capabilities=[capability]
    )
    kwargs["ops"] = SurfaceOperators(
        runtime,
        implementations={
            "splat.alpha_composite@3": candidate.manifest.implementation_id
        },
    )
    return _batched_factory(config, **kwargs)


renderers.register("torch-batched-cuda-composite", _cuda_composite_factory)


def _compiled_factory(config, **kwargs):
    from .batched_renderer import BatchedTorchRenderer

    return BatchedTorchRenderer(config, compiled=True, **kwargs)


renderers.register("torch-compiled", _compiled_factory)


def _gsplat_rade_factory(config, *, mode, trunc_sigma, ops):
    from ..integrations.gsplat_radegs import GsplatRadeRenderer

    if mode != "radegs":
        raise ValueError(
            "gsplat-radegs implements RaDe-GS, not AmbiSuR/GOF; select the corresponding explicit backend"
        )
    return GsplatRadeRenderer(config, ops=ops)


renderers.register("gsplat-radegs", _gsplat_rade_factory)


def _gsplat_rade_moment_factory(config, *, mode, trunc_sigma, ops):
    from ..integrations.gsplat_radegs_moments import GsplatRadeMomentRenderer

    if mode != "radegs":
        raise ValueError("native RaDe-GS moment backend only supports RaDe-GS")
    return GsplatRadeMomentRenderer(config, ops=ops)


renderers.register("gsplat-radegs-moments", _gsplat_rade_moment_factory)


def _gsplat_rade_stable_factory(config, *, mode, trunc_sigma, ops):
    from ..integrations.gsplat_radegs_moments import GsplatRadeMomentRenderer

    if mode != "radegs":
        raise ValueError("stable native moment backend only supports RaDe-GS")
    return GsplatRadeMomentRenderer(config, ops=ops, stable_geometry=True)


renderers.register("gsplat-radegs-stable-moments", _gsplat_rade_stable_factory)


def make_renderer(config, method, ops):
    return renderers.get(config.backend)(
        config,
        mode=method.rendering_mode,
        trunc_sigma=method.config.loss.trunc_sigma,
        ops=ops,
    )
