"""Explicit base MILo recipe; paper schedules are not shortened for tests."""

from dataclasses import dataclass, field

from .config import StudioConfig


@dataclass
class MiloConfig:
    # The paper does not publish a single cross-scene base population size.
    # This is our explicit development setting, not an attributed author default.
    pivot_gaussians: int = 100_000
    variant: str = 'base'
    topology_backend: str = 'scipy-cpu-reference'
    mesh_backend: str = 'torch-polygon-reference'
    photometric_reduction: str = 'pixel-sum-channel-mean'
    sdf_initialization_scale: str = 'observed-global-absmax'


@dataclass
class MiloStudioConfig(StudioConfig):
    method: str = 'milo'
    milo: MiloConfig = field(default_factory=MiloConfig)

    def validate(self):
        super().validate()
        if self.method != 'milo' or self.milo.variant != 'base':
            raise ValueError('this recipe implements the MILo base/RaDe-GS variant')
        if type(self.milo.pivot_gaussians) is not int or self.milo.pivot_gaussians < 1:
            raise ValueError('milo.pivot_gaussians must be a positive integer')
        if self.milo.topology_backend != 'scipy-cpu-reference' or self.milo.mesh_backend != 'torch-polygon-reference':
            raise ValueError('MILo topology/mesh backends must be explicitly implemented')
        if self.milo.photometric_reduction != 'pixel-sum-channel-mean' or self.milo.sdf_initialization_scale != 'observed-global-absmax':
            raise ValueError('MILo reduction and SDF initialization cannot change silently')
        if (self.loss.ssim, self.loss.normal, self.loss.geometric_start, self.loss.depth_distortion, self.loss.multiview) != (.2, .05, 3000, 0., 0.):
            raise ValueError('MILo paper losses and the 3000-step normal start are fixed')
        if self.densify.start != 500 or self.densify.stop != 3000 or self.train.steps > 18000:
            raise ValueError('MILo uses 500..3000 densification and at most 18000 training steps')
        if self.model.appearance:
            raise ValueError('the base MILo recipe has no appearance embedding')
        return self
