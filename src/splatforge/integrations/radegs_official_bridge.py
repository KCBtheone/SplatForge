"""Owned RaDe-GS module bundle for the trusted full-training entry."""

from __future__ import annotations

import math

import torch

module_files = (
    "src/splatforge/studio/methods/radegs.py",
    "src/splatforge/studio/losses.py",
    "src/splatforge/studio/multiview.py",
    "src/splatforge/integrations/gsplat_radegs_moments.py",
    "src/splatforge/integrations/radegs_ordered_adjoint.py",
)


def install():
    """Import every owned mathematical module and report the active bundle."""
    try:
        from . import gsplat_radegs_moments, radegs_ordered_adjoint
        from ..studio import losses, multiview
        from ..studio.methods import radegs
    except ImportError:
        # Trusted runners or bootstrap loaders may execute this file directly
        # or via a loader that sets __package__ without fully initializing the
        # parent packages in sys.modules. Fall back to absolute imports which
        # work regardless of how the bootstrap loaded this file.
        from splatforge.integrations import gsplat_radegs_moments, radegs_ordered_adjoint
        from splatforge.studio import losses, multiview
        from splatforge.studio.methods import radegs

    if radegs.RaDeGS.default_config().render.backend != "gsplat-radegs-moments":
        raise RuntimeError("RaDe-GS bridge requires the owned moments renderer")

    from splatforge.gaussian import SurfaceOperators
    from splatforge.studio.renderers import (
        GaussianView,
        bind_configured_ordered_adjoint,
    )

    render_config = radegs.RaDeGS.default_config().render
    owned_ops = SurfaceOperators()
    owned_renderer = gsplat_radegs_moments.GsplatRadeMomentRenderer(
        render_config,
        ops=owned_ops,
        exact_basis_geometry=True,
        centered_l2=True,
        depth_xy_bridge=True,
        ordered_adjoint_build_directory=getattr(
            render_config, "ordered_adjoint_build_directory", ""
        ),
        replay_transmittance=getattr(render_config, "replay_transmittance", False),
    )
    bind_configured_ordered_adjoint(owned_renderer)

    class OfficialCameraView:
        __slots__ = (
            "K",
            "origin",
            "camera_to_world",
            "world_to_camera",
            "width",
            "height",
        )

        def __init__(self, camera):
            width = int(camera.image_width)
            height = int(camera.image_height)
            device = torch.device("cuda")
            dtype = torch.float32
            if hasattr(camera, "Fx") and hasattr(camera, "Fy"):
                fx = torch.as_tensor(camera.Fx, device=device, dtype=dtype)
                fy = torch.as_tensor(camera.Fy, device=device, dtype=dtype)
                cx = torch.as_tensor(camera.Cx, device=device, dtype=dtype)
                cy = torch.as_tensor(camera.Cy, device=device, dtype=dtype)
            else:
                fx = 0.5 * width / math.tan(0.5 * float(camera.FoVx))
                fy = 0.5 * height / math.tan(0.5 * float(camera.FoVy))
                cx = 0.5 * width - 0.5
                cy = 0.5 * height - 0.5
            self.K = torch.stack(
                (
                    torch.stack((fx, fx.new_zeros(()), cx)),
                    torch.stack((fx.new_zeros(()), fy, cy)),
                    torch.stack((fx.new_zeros(()), fx.new_zeros(()), fx.new_ones(()))),
                )
            )
            self.world_to_camera = camera.world_view_transform.to(
                device=device, dtype=dtype
            )
            self.camera_to_world = torch.linalg.inv(self.world_to_camera)
            self.origin = self.camera_to_world[:3, 3].detach()
            self.width = width
            self.height = height

    def owned_scene(pc, scaling_modifier):
        scales, opacities = pc.get_scaling_n_opacity_with_3D_filter
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier
        return GaussianView(
            pc.get_xyz,
            pc.get_rotation,
            scales,
            opacities,
            pc.get_features,
            int(pc.active_sh_degree),
        )

    def owned_depth_normal(camera, depth):
        if depth.ndim != 2:
            raise ValueError("owned depth-to-normal requires an HxW depth map")
        normal, _ = owned_ops("geometry.depth_normals@1", depth, camera.K)
        return normal.permute(2, 0, 1).contiguous()

    def owned_depth_double_normal(camera, expected_depth, median_depth):
        return torch.stack(
            (
                owned_depth_normal(camera, expected_depth),
                owned_depth_normal(camera, median_depth),
            ),
            dim=0,
        )

    def owned_render(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        kernel_size,
        scaling_modifier=1.0,
        require_depth=True,
    ):
        del kernel_size, pipe
        camera = OfficialCameraView(viewpoint_camera)
        scene = owned_scene(pc, scaling_modifier)
        if tuple(bg_color.detach().cpu().tolist()) == (1.0, 1.0, 1.0):
            owned_renderer.config.background = (1.0, 1.0, 1.0)
        elif tuple(bg_color.detach().cpu().tolist()) == (0.0, 0.0, 0.0):
            owned_renderer.config.background = (0.0, 0.0, 0.0)
        else:
            owned_renderer.config.background = tuple(
                float(value) for value in bg_color.detach().cpu().tolist()
            )
        output = (
            owned_renderer.render(scene, camera)
            if require_depth
            else owned_renderer.render_rgb(scene, camera)
        )
        expected_depth = output.depth[..., 0]
        median_depth = output.median_depth[..., 0]
        normal = output.normals.permute(2, 0, 1).contiguous()
        viewspace_points = torch.cat(
            (
                output.projected_means.reshape(-1, 2),
                scene.means[:, 2:3],
            ),
            dim=-1,
        )
        return {
            "render": output.rgb.permute(2, 0, 1).contiguous(),
            "mask": output.alpha.permute(2, 0, 1).contiguous(),
            "expected_depth": expected_depth,
            "median_depth": median_depth,
            "viewspace_points": viewspace_points,
            "visibility_filter": output.projected_radii > 0,
            "radii": output.projected_radii,
            "normal": normal,
        }

    import gaussian_renderer
    import utils.graphics_utils as official_graphics_utils

    gaussian_renderer.render = owned_render
    official_graphics_utils.depth_to_normal = owned_depth_normal
    official_graphics_utils.depth_double_to_normal = owned_depth_double_normal

    return {
        "renderer": gsplat_radegs_moments.GsplatRadeMomentRenderer,
        "method": radegs.RaDeGS,
        "loss": losses.normal_consistency,
        "multiview": multiview.multiview_losses,
        "adjoint": radegs_ordered_adjoint,
        "owned_renderer": owned_renderer,
        "owned_ops": owned_ops,
        "render": owned_render,
        "depth_to_normal": owned_depth_normal,
        "depth_double_to_normal": owned_depth_double_normal,
        "module_files": module_files,
    }
