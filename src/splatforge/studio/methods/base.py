"""Small method extension interface; a plugin may replace the entire pipeline."""

from ..config import StudioConfig
from ..losses import photometric_loss


class SurfaceMethod:
    name = ""
    description = ""
    rendering_mode = ""
    exporter = "tsdf"
    requires_prior = False

    def __init__(self, config):
        self.config = config

    @classmethod
    def default_config(cls):
        return StudioConfig(method=cls.name)

    @classmethod
    def make_pipeline(cls, config):
        from ..pipeline import GaussianPipeline

        return GaussianPipeline(config, cls(config))

    def losses(self, pipeline, sample, output, step):
        predicted = pipeline.model.compensate(output.rgb, sample["index"])
        return {
            "photometric": photometric_loss(
                predicted, sample["image"], self.config.loss.ssim
            )
        }

    def selection(self, model, ops):
        return None

    def regularization(self, pipeline, sample, output, step):
        return {}

    def photometric_only(self, step):
        return False
