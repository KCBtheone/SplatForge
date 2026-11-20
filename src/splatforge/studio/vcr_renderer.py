"""Studio adapter for VCR-GauS (Chen et al., 2024).

The native rasterizer returns tangent-plane z-depth and camera-space normals
oriented away from the camera. The projection uses a centred principal point.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..providers.torch.gaussian.reference import rotation_matrix, sh_color
from .renderers import GaussianView, RenderOutput


def projection_matrix(fov_x: float, fov_y: float, near: float, far: float, device=None, dtype=torch.float32):
    """Camera-to-clip matrix with the centred camera convention."""
    tan_half_y = math.tan(fov_y * 0.5)
    tan_half_x = math.tan(fov_x * 0.5)
    top = tan_half_y * near
    bottom = -top
    right = tan_half_x * near
    left = -right
    matrix = torch.zeros(4, 4, device=device, dtype=dtype)
    matrix[0, 0] = 2 * near / (right - left)
    matrix[1, 1] = 2 * near / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1
    matrix[2, 2] = far / (far - near)
    matrix[2, 3] = -(far * near) / (far - near)
    return matrix


def centred_intrinsic(camera):
    """Centred intrinsic at the training resolution (reference convention)."""
    height, width = int(camera.height), int(camera.width)
    focal_x = float(camera.K[0, 0])
    focal_y = float(camera.K[1, 1])
    intrinsic = torch.eye(3, dtype=torch.float32, device=camera.K.device)
    intrinsic[0, 0] = focal_x
    intrinsic[1, 1] = focal_y
    intrinsic[0, 2] = width / 2
    intrinsic[1, 2] = height / 2
    return intrinsic


def unit_ray_directions(intrinsic, height: int, width: int, device, dtype):
    """Per-pixel unit view directions in camera space, shaped (3, H, W).

    The kernel reads the buffer as three stacked planes, so the memory layout
    has to be contiguous in that order.
    """
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    rays = torch.stack(
        (
            (grid_x - intrinsic[0, 2]) / intrinsic[0, 0],
            (grid_y - intrinsic[1, 2]) / intrinsic[1, 1],
            torch.ones_like(grid_x),
        ),
        dim=-1,
    )
    rays = F.normalize(rays, dim=-1)
    return rays.permute(2, 0, 1).contiguous()


def shortest_axis_normals(quats, scales):
    """Per-Gaussian surface normal along the shortest covariance axis."""
    rotation = rotation_matrix(quats)
    axis = scales.detach().argmin(-1)
    return rotation.gather(2, axis[:, None, None].expand(-1, 3, 1)).squeeze(-1)


class VCRNativeRenderer:
    """Render with SplatForge's VCR-GauS CUDA kernels."""

    provides_rade_normal_numerator = False

    def __init__(self, config, *, mode, trunc_sigma=2.0, ops=None):
        if mode != "vcrgaus":
            raise ValueError("the native renderer implements VCR-GauS only")
        self.config = config
        self.mode = mode
        self.trunc_sigma = trunc_sigma
        from ..providers.torch.gaussian.vcr_cuda import count, rasterize

        self._rasterize = rasterize
        self._count = count

    def _camera_geometry(self, camera):
        near = float(self.config.near)
        far = float(self.config.far)
        fov_x = 2 * math.atan(camera.width / (2 * float(camera.K[0, 0])))
        fov_y = 2 * math.atan(camera.height / (2 * float(camera.K[1, 1])))
        world_to_camera = camera.world_to_camera
        projection = projection_matrix(
            fov_x,
            fov_y,
            near,
            far,
            device=world_to_camera.device,
            dtype=world_to_camera.dtype,
        )
        view_matrix = world_to_camera.transpose(0, 1)
        full_projection = view_matrix @ projection.transpose(0, 1)
        camera_center = view_matrix.inverse()[3, :3]
        return view_matrix, full_projection, camera_center, fov_x, fov_y

    def _sh_colours(self, scene, camera_center):
        return sh_color(
            scene.sh,
            scene.means - camera_center,
            degree=int(scene.sh_degree),
        )

    def render(self, scene: GaussianView, camera, *, selected=None) -> RenderOutput:
        means = scene.means
        device = means.device
        dtype = means.dtype

        world_normal = shortest_axis_normals(scene.quats, scene.scales)
        view_direction = means - camera.origin
        orientation = torch.where(
            (view_direction * world_normal).sum(-1, keepdim=True) > 0, 1.0, -1.0
        ).to(dtype)
        camera_normal = (world_normal * orientation) @ camera.world_to_camera[
            :3, :3
        ].T

        view_matrix, full_projection, camera_center, fov_x, fov_y = (
            self._camera_geometry(camera)
        )
        directions = unit_ray_directions(
            centred_intrinsic(camera), int(camera.height), int(camera.width),
            device, dtype,
        )
        colours = self._sh_colours(scene, camera_center)
        background = torch.as_tensor(
            self.config.background, device=device, dtype=dtype
        ).reshape(3)

        screen_xy = torch.zeros(
            (len(means), 2), device=device, dtype=dtype, requires_grad=True
        )
        colour, depth, normal, alpha, distortion, radii = self._rasterize(
            screen_xy,
            means,
            scene.scales,
            scene.quats,
            scene.opacities.reshape(-1, 1),
            colours,
            camera_normal,
            directions,
            view_matrix,
            full_projection,
            background,
            float(camera.K[0, 0]),
            float(camera.K[1, 1]),
            math.tan(fov_x * 0.5),
            math.tan(fov_y * 0.5),
            int(camera.width),
            int(camera.height),
        )
        alpha = alpha.permute(1, 2, 0)
        zeros = torch.zeros_like(alpha)
        return RenderOutput(
            rgb=colour.permute(1, 2, 0),
            alpha=alpha,
            depth=depth.permute(1, 2, 0),
            normals=F.normalize(normal.permute(1, 2, 0), dim=-1),
            distortion_l1=None,
            distortion_l2=distortion.permute(1, 2, 0),
            median_depth=zeros,
            ray_color=zeros,
            ambiguity_mask=zeros,
            projected_means=screen_xy,
            projected_radii=radii,
            normal_numerator=normal.permute(1, 2, 0),
            distortion_profile="l2",
        )

    @torch.no_grad()
    def render_importance(self, scene: GaussianView, camera):
        """Per Gaussian pixel activation counts from the native count kernel."""
        means = scene.means
        dtype = means.dtype
        view_matrix, full_projection, _, fov_x, fov_y = self._camera_geometry(camera)
        counts = self._count(
            means,
            scene.scales,
            scene.quats,
            scene.opacities.reshape(-1, 1),
            view_matrix,
            full_projection,
            float(camera.K[0, 0]),
            float(camera.K[1, 1]),
            math.tan(fov_x * 0.5),
            math.tan(fov_y * 0.5),
            int(camera.width),
            int(camera.height),
        )
        return counts.reshape(-1).to(dtype)

