"""Capacity checks for runs with a configured data root."""

from __future__ import annotations

import shutil
import time

import torch

from .._paths import data_root, output_path


def capacity(config, *, check_gpu=True):
    data_disk = data_root()
    if data_disk is None:
        return {"allowed": True, "policy": "no-data-root"}
    output_path(config.train.output)
    disk_free = shutil.disk_usage(data_disk).free
    gpu_free = None
    if check_gpu and torch.device(config.train.device).type == "cuda":
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
        report = capacity(config, check_gpu=check_gpu)
        if report["allowed"]:
            return report
        write_state("WAIT_RESOURCE_GATE", capacity=report)
        time.sleep(30)
