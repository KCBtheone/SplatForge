from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from splatforge.studio import exporters, trainer
from splatforge.studio.artifacts import load_checkpoint
from splatforge.studio.demo import make_demo
from splatforge.studio.methods.radegs import RaDeGS
from splatforge.studio.pipeline import GaussianPipeline


@pytest.fixture
def config(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = RaDeGS.default_config()
        config.data.source = str(make_demo(tmp_path / "data", width=16, points=16))
        config.data.downscale = 1
        config.data.init_points = 16
        config.train.output = str(tmp_path / "run")
        config.train.device = "cpu"
        config.train.steps = 1
        config.train.checkpoint_every = 1
        config.train.eval_every = 0
        config.train.export_on_finish = False
        config.render.backend = "torch"
        config.model.opacity = 0.6
        config.loss.geometric_start = 0
        config.densify.start = 1
        config.densify.interval = 1
        config.densify.gradient_threshold = 0.0
        config.densify.max_gaussians = 24
        config.export.kind = "gaussians"
        yield config
    finally:
        torch.set_num_threads(threads)


def test_radegs_resume_after_densification_preserves_layout_and_gradients(config):
    initial = trainer.Trainer(config)
    initial.run()
    saved = load_checkpoint(initial.root / "checkpoints/latest.pt")
    assert saved["step"] == 1
    assert saved["progress"]["next_stage"] == "evaluate"
    assert initial.pipeline.model.count > config.data.init_points

    config.train.steps = 2
    resumed = trainer.Trainer(config, resume="latest")
    pipeline = resumed.run()
    assert resumed.step == 2
    assert resumed.order == saved["sampler"]["order"]
    assert resumed.order_offset == saved["sampler"]["offset"] + 1
    for group in resumed.optimizer.param_groups:
        for parameter in group["params"]:
            assert resumed.optimizer.state[parameter]["exp_avg"].shape == parameter.shape

    camera = pipeline.data.sample(0)["camera"]
    result = pipeline.render(camera)
    request = pipeline.method.render_request(2)
    request.validate(result, camera)
    with pytest.raises(RuntimeError, match="depth must support backward"):
        request.validate(replace(result, depth=result.depth.detach()), camera)
    cropped = replace(camera, K=camera.K.clone())
    cropped.K[0, 2] -= 1
    with pytest.raises(ValueError, match="different pixel grid"):
        request.validate(result, cropped)

    model = pipeline.model
    gradient, = torch.autograd.grad(
        result.normal_numerator.square().sum() + result.depth.sum(), model.means
    )
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    reverse = torch.arange(model.count - 1, -1, -1)
    pipeline.densifier._replace(model, resumed.optimizer, reverse, reverse[:0])
    with pytest.raises(ValueError, match="stale Gaussian layout"):
        pipeline.densifier.observe(result, model)


def test_resume_retries_failed_export_without_repeating_evaluation(config, monkeypatch):
    config.train.export_on_finish = True
    first = trainer.Trainer(config)

    def interrupted(*args, **kwargs):
        raise OSError("interrupted export")

    with monkeypatch.context() as patch:
        patch.setattr(exporters, "export_pipeline", interrupted)
        with pytest.raises(OSError, match="interrupted export"):
            first.run()
    record = json.loads((first.root / "state.json").read_text(encoding="utf-8"))
    assert record["progress"]["next_stage"] == "export"
    assert (first.root / "evaluation_test.json").is_file()

    def replayed(*args, **kwargs):
        pytest.fail("a completed stage was repeated")

    monkeypatch.setattr(trainer, "evaluate_views", replayed)
    monkeypatch.setattr(trainer.Trainer, "_train", replayed)
    trainer.Trainer(config, resume="latest").run()
    assert (first.root / "gaussians.ply").is_file()
    record = json.loads((first.root / "state.json").read_text(encoding="utf-8"))
    assert record["stage"] == "COMPLETE"
    assert record["progress"]["next_stage"] == "complete"
    monkeypatch.setattr(exporters, "export_pipeline", replayed)
    trainer.Trainer(config, resume="latest").run()


def test_failed_post_update_keeps_last_complete_checkpoint(config, monkeypatch):
    config.train.steps = 2
    original = GaussianPipeline.after_step

    def interrupted(pipeline, step, optimizer):
        original(pipeline, step, optimizer)
        if step == 2:
            raise RuntimeError("interrupted topology update")

    monkeypatch.setattr(GaussianPipeline, "after_step", interrupted)
    run = trainer.Trainer(config)
    with pytest.raises(RuntimeError, match="interrupted topology update"):
        run.run()
    checkpoint = load_checkpoint(run.root / "checkpoints/latest.pt")
    assert checkpoint["step"] == 1
    assert checkpoint["sampler"]["offset"] == 1
    assert not (run.root / "checkpoints/step_00000002.pt").exists()


def test_optimizer_restore_rejects_reordered_equal_shape_parameters():
    model = torch.nn.ParameterDict({
        "first": torch.nn.Parameter(torch.zeros(3)),
        "second": torch.nn.Parameter(torch.zeros(3)),
    })
    pipeline = SimpleNamespace(model=model)
    adam = torch.optim.Adam([model["first"], model["second"]])
    payload = {
        "optimizer_layout": trainer.optimizer_layout(pipeline, adam),
        "optimizer": adam.state_dict(),
    }
    reordered = torch.optim.Adam([model["second"], model["first"]])
    with pytest.raises(ValueError, match="optimizer parameter names or layout changed"):
        trainer.restore_optimizer(pipeline, reordered, payload)
