"""AutoDL capacity policy: 8 GiB GPU and 20 GiB data disk; no RAM gate."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import torch


def capacity(config):
    data_disk = Path("/root/autodl-tmp")
    # Unlike Path.is_dir on Python 3.10-3.12, os.path.isdir treats an
    # inaccessible /root on ordinary Linux accounts as an unavailable path.
    if not os.path.isdir(data_disk):
        return {"allowed": True, "policy": "local-no-autodl-gate"}
    output = Path(config.train.output).resolve()
    if not output.is_relative_to(data_disk.resolve()):
        raise ValueError(
            "AutoDL output, logs and checkpoints must live under /root/autodl-tmp"
        )
    disk_free = shutil.disk_usage(data_disk).free
    gpu_free = None
    if torch.device(config.train.device).type == "cuda":
        device = torch.device(config.train.device)
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        gpu_free = torch.cuda.mem_get_info(index)[0]
    return {
        "allowed": disk_free >= 20 * 1024**3
        and (gpu_free is None or gpu_free >= 8 * 1024**3),
        "gpu_free_bytes": gpu_free,
        "disk_free_bytes": disk_free,
        "gpu_min_bytes": 8 * 1024**3,
        "disk_min_bytes": 20 * 1024**3,
    }


def wait_for_capacity(config, write_state, *, check_gpu=True):
    while True:
        report = capacity(config)
        if not check_gpu and "disk_free_bytes" in report:
            report["allowed"] = report["disk_free_bytes"] >= report["disk_min_bytes"]
        if report["allowed"]:
            return report
        write_state("WAIT_RESOURCE_GATE", capacity=report)
        time.sleep(30)
