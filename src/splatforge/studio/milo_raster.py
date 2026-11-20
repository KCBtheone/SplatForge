"""MILo mesh depth/face-normal rendering with explicit nvdiffrast dispatch.

Camera coordinates are x-right/y-down/z-forward; nvdiffrast clip coordinates
and image storage follow OpenGL. No other renderer is a silent fallback.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MeshRender:
    depth: torch.Tensor
    normals: torch.Tensor
    valid: torch.Tensor
    face_ids: torch.Tensor


def camera_vertices(vertices, camera):
    transform = camera.world_to_camera.to(vertices)
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def clip_coordinates(vertices, camera, near, far):
    if not 0 < near < far:
        raise ValueError("mesh clipping bounds must be positive and ordered")
    view = camera_vertices(vertices, camera)
    projected = view @ camera.K.to(vertices).T
    z = view[:, 2]
    return torch.stack((2 * projected[:, 0] / camera.width - z,
                        z - 2 * projected[:, 1] / camera.height,
                        (far + near) / (far - near) * z - 2 * far * near / (far - near), z), -1)


def decode_render(vertices, faces, camera, raster, depth):
    """Convert nvdiffrast buffers to the existing top-down camera convention."""
    face_ids = raster[0, ..., 3].to(torch.long) - 1
    valid = face_ids >= 0
    sites = camera_vertices(vertices, camera)[faces]
    normals = F.normalize(torch.cross(sites[:, 1] - sites[:, 0], sites[:, 2] - sites[:, 0], dim=-1), dim=-1)
    normal_image = torch.where(valid[..., None], normals[face_ids.clamp_min(0)], 0)
    return MeshRender(depth[0].flip(0), normal_image.flip(0), valid.flip(0), face_ids.flip(0))


class NvdiffrastMeshRenderer:
    def __init__(self, near=.01, far=10000.):
        self.near, self.far = near, far
        self._context = None
        self._device = None
        self.counts = defaultdict(int)

    def render(self, mesh, camera):
        vertices, faces = mesh.vertices, mesh.faces
        if vertices.device.type != "cuda" or vertices.dtype != torch.float32:
            raise ValueError("nvdiffrast mesh rendering requires CUDA float32; no CPU fallback")
        if not len(faces):
            zero = vertices.sum() * 0
            shape = (camera.height, camera.width)
            return MeshRender(vertices.new_zeros((*shape, 1)) + zero,
                vertices.new_zeros((*shape, 3)) + zero,
                torch.zeros(shape, dtype=torch.bool, device=vertices.device),
                torch.full(shape, -1, dtype=torch.long, device=vertices.device))
        import nvdiffrast.torch as dr

        if self._context is None:
            self._context = dr.RasterizeCudaContext(device=vertices.device)
            self._device = vertices.device
        if self._device != vertices.device:
            raise ValueError("mesh rasterizer context cannot silently change CUDA devices")
        pos = clip_coordinates(vertices, camera, self.near, self.far)[None].contiguous()
        tri = faces.to(torch.int32).contiguous()
        raster, _ = dr.rasterize(self._context, pos, tri, (camera.height, camera.width), grad_db=False)
        self.counts["rasterize"] += 1
        attributes = camera_vertices(vertices, camera)[:, 2:3][None].contiguous()
        depth, _ = dr.interpolate(attributes, raster, tri)
        self.counts["interpolate"] += 1
        depth = dr.antialias(depth.contiguous(), raster, pos, tri)
        self.counts["antialias"] += 1
        return decode_render(vertices, faces, camera, raster, depth)
