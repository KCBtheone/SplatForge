from types import SimpleNamespace
from unittest.mock import patch

from splatforge.studio.resources import capacity


def test_unconfigured_data_root_skips_capacity_gate(monkeypatch):
    monkeypatch.delenv("SPLATFORGE_DATA_ROOT", raising=False)
    assert capacity(None) == {"allowed": True, "policy": "no-data-root"}


def test_configured_data_root_applies_disk_capacity_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("SPLATFORGE_DATA_ROOT", str(tmp_path))
    config = SimpleNamespace(train=SimpleNamespace(
        output=tmp_path / "example/run", device="cpu"
    ))
    with patch("splatforge.studio.resources.shutil.disk_usage",
               return_value=SimpleNamespace(free=19 * 1024**3)):
        report = capacity(config)
    assert report["allowed"] is False
    assert report["disk_min_bytes"] == 20 * 1024**3
