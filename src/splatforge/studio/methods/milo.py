"""Independent MILo base/RaDe-GS method, using the normal Trainer factory."""

from .base import SurfaceMethod
from ..milo_config import MiloStudioConfig


class MILo(SurfaceMethod):
    name = 'milo'
    description = 'MILo differentiable mesh-in-the-loop surface reconstruction'
    rendering_mode = 'radegs'
    exporter = 'milo'

    @classmethod
    def default_config(cls):
        config = MiloStudioConfig()
        config.render.backend = 'torch-batched'
        config.loss.geometric_start = 3000
        config.loss.normal = .05
        config.loss.depth_distortion = 0.
        config.densify.stop = 3000
        config.train.steps = 18000
        config.train.output = 'outputs/milo'
        return config

    @classmethod
    def make_pipeline(cls, config):
        from ..milo_pipeline import MiloPipeline
        return MiloPipeline(config, cls(config))
