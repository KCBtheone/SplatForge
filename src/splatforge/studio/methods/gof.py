from ..losses import normal_consistency
from .base import SurfaceMethod


class GOF(SurfaceMethod):
    name = "gof"
    description = (
        "Ray-Gaussian opacity field with adaptive tetrahedral level-set extraction"
    )
    rendering_mode = "gof"
    exporter = "gof"

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.densify.absolute_gradients = True
        config.densify.prune_opacity = 0.05
        return config

    def regularization(self, pipeline, sample, output, step):
        if step < self.config.loss.geometric_start:
            return {}
        return {
            "depth_distortion": self.config.loss.depth_distortion
            * output.distortion_l1.mean(),
            "normal_consistency": self.config.loss.normal
            * normal_consistency(
                output, sample["camera"], pipeline.ops, sample["mask"]
            ),
        }

    def photometric_only(self, step):
        return step < self.config.loss.geometric_start
