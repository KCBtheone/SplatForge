from ..losses import normal_consistency
from ..renderers import RenderRequest
from .base import SurfaceMethod


_RGB = RenderRequest((("rgb", 3),), ("rgb",))
_GEOMETRY = RenderRequest(
    (("rgb", 3), ("alpha", 1), ("depth", 1), ("normal_numerator", 3), ("distortion_l2", 1)),
    ("rgb", "alpha", "depth", "normal_numerator", "distortion_l2"),
)


class RaDeGS(SurfaceMethod):
    name = "radegs"
    description = "RaDe-GS surface reconstruction"
    rendering_mode = "radegs"

    def render_request(self, step):
        return _RGB if self.photometric_only(step) else _GEOMETRY

    def losses(self, pipeline, sample, output, step):
        if self.config.model.paper_decoupled_appearance:
            return {"photometric":pipeline.model.decoupled_appearance.photometric_loss(
                output.rgb,sample["image"],sample["index"],self.config.loss.ssim)}
        return super().losses(pipeline,sample,output,step)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        # Native moments remains an explicit experimental selection until its
        # real geometry-output and update gates pass. Never promote a finite-
        # only or RGB-only diagnostic into the default training backend.
        config.render.backend = "gsplat-radegs-moments"
        config.loss.normal = 0.05
        config.loss.geometric_start = 7000
        config.loss.depth_distortion = 0.0
        return config

    def regularization(self, pipeline, sample, output, step):
        if step < self.config.loss.geometric_start:
            return {}
        if getattr(output, "normal_numerator", None) is None:
            raise RuntimeError("RaDe-GS geometry requires raw normal_numerator")
        return {
            "depth_distortion": self.config.loss.depth_distortion
            * output.distortion_l2.mean(),
            "normal_consistency": self.config.loss.normal
            * normal_consistency(
                output, sample["camera"], pipeline.ops, sample["mask"],
                normalize_numerator=self.config.loss.radegs_normalize_numerator,
            ),
        }

    def photometric_only(self, step):
        return step < self.config.loss.geometric_start
