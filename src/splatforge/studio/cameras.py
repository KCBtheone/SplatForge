"""Camera records independent of the scene representation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Camera:
    K: torch.Tensor
    world_to_camera: torch.Tensor
    width: int
    height: int
    name: str = ""

    @property
    def camera_to_world(self):
        return torch.linalg.inv(self.world_to_camera)

    @property
    def origin(self):
        return self.camera_to_world[:3, 3]

    def to(self, device, dtype=torch.float32):
        return Camera(
            self.K.to(device=device, dtype=dtype),
            self.world_to_camera.to(device=device, dtype=dtype),
            self.width,
            self.height,
            self.name,
        )
