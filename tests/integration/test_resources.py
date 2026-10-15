from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from splatforge.studio.resources import capacity


def test_inaccessible_root_does_not_break_a_non_autodl_linux_host():
    with patch("os.stat", side_effect=PermissionError("inaccessible /root")):
        assert capacity(None) == {"allowed": True, "policy": "local-no-autodl-gate"}


def test_real_autodl_path_still_applies_the_data_disk_capacity_gate():
    config = SimpleNamespace(train=SimpleNamespace(
        output=Path("/root/autodl-tmp") / "example/run", device="cpu"
    ))
    with (
        patch("os.path.isdir", return_value=True),
        patch("splatforge.studio.resources.shutil.disk_usage",
              return_value=SimpleNamespace(free=19 * 1024**3)),
    ):
        report = capacity(config)
    assert report["allowed"] is False
    assert report["disk_min_bytes"] == 20 * 1024**3
