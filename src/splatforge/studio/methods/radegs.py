from ..losses import normal_consistency
from .base import SurfaceMethod


class RaDeGS(SurfaceMethod):
    name = "radegs"
    description = "RaDe-GS depth/normal rasterization; direct Apache gsplat port plus canonical L2 loss"
    rendering_mode = "radegs"

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.render.backend = "gsplat-radegs"
        config.loss.normal = 5.0
        # RaDe-GS paper v2, section 4.1.1: 15k photometric-only updates,
        # followed by 15k depth/normal-constrained updates. Not 1k warmup.
        config.loss.geometric_start = 15000
        return config

    def regularization(self, pipeline, sample, output, step):
        if step < self.config.loss.geometric_start:
            return {}
        return {
            "depth_distortion": self.config.loss.depth_distortion
            * output.distortion_l2.mean(),
            "normal_consistency": self.config.loss.normal
            * normal_consistency(
                output, sample["camera"], pipeline.ops, sample["mask"]
            ),
        }

    def photometric_only(self, step):
        return step < self.config.loss.geometric_start
