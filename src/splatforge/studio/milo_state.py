"""Trainable MILo mesh state; no trainer, worker, or scheduling loop."""

import torch
from torch import nn

from .milo_mesh import gaussian_pivots, gaussian_pivot_mesh, triangulate_pivots_scipy_cpu_reference
from .milo_supervision import mesh_interior_labels, sdf_regularizers


class MiloMeshState(nn.Module):
    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.raw_sdf = nn.Parameter(torch.empty(0, 9, device=device, dtype=dtype))
        self.register_buffer("pivot_ids", torch.empty(0, dtype=torch.long, device=device))
        self.register_buffer("cells", torch.empty(0, 4, dtype=torch.long, device=device))
        self.register_buffer("interior", torch.empty(0, 9, dtype=torch.bool, device=device))
        self.register_buffer("topology_step", torch.tensor(-1, dtype=torch.long, device=device))
        self.register_buffer("interior_step", torch.tensor(-1, dtype=torch.long, device=device))

    @property
    def active(self):
        return bool(len(self.pivot_ids))

    def parameter_group(self):
        return {"params": [self.raw_sdf], "name": "milo_raw_sdf", "lr": .025}

    def initialize(self, pivot_ids, raw_sdf, cells, *, step, optimizer=None):
        if self.active:
            raise ValueError("MILo mesh state is already initialized")
        if pivot_ids.ndim != 1 or pivot_ids.dtype != torch.long or raw_sdf.shape != (len(pivot_ids), 9):
            raise ValueError("one selected Gaussian index and nine independent raw SDFs required")
        if not len(pivot_ids) or cells.ndim != 2 or cells.shape[1] != 4 or cells.dtype != torch.long:
            raise ValueError("nonempty pivots and int64 tetrahedral cells required")
        group = None
        if optimizer is not None:
            matches = [g for g in optimizer.param_groups if g.get("name") == "milo_raw_sdf"]
            if len(matches) != 1 or len(matches[0]["params"]) != 1 or matches[0]["params"][0] is not self.raw_sdf:
                raise ValueError("optimizer must already contain this mesh state's named SDF group")
            group = matches[0]
        old = self.raw_sdf
        self.raw_sdf = nn.Parameter(raw_sdf.detach().to(old).clone())
        self.pivot_ids = pivot_ids.detach().to(device=old.device).clone()
        self.cells = cells.detach().to(device=old.device).clone()
        self.interior = torch.zeros_like(self.raw_sdf, dtype=torch.bool)
        self.topology_step.fill_(step)
        self.interior_step.fill_(-1)
        if group is not None:
            group["params"][0] = self.raw_sdf
            optimizer.state.pop(old, None)

    def forward(self, means, quaternions, log_scales):
        if not self.active:
            raise ValueError("MILo mesh state must be initialized before mesh rendering")
        return gaussian_pivot_mesh(means, quaternions, log_scales, self.raw_sdf, self.cells,
            pivot_ids=self.pivot_ids)

    @torch.no_grad()
    def refresh_topology(self, means, quaternions, log_scales, step):
        if not self.active:
            raise ValueError("initialize the mesh state before refreshing topology")
        if step - int(self.topology_step) < 500:
            return False
        sites = gaussian_pivots(means, quaternions, log_scales, pivot_ids=self.pivot_ids)
        self.cells = triangulate_pivots_scipy_cpu_reference(sites.reshape(-1, 3))
        self.topology_step.fill_(step)
        return True

    @torch.no_grad()
    def refresh_interior(self, means, quaternions, log_scales, observations, step):
        if not self.active:
            raise ValueError("initialize the mesh state before interior feedback")
        if int(self.interior_step) >= 0 and step - int(self.interior_step) < 200:
            return False
        sites = gaussian_pivots(means, quaternions, log_scales, pivot_ids=self.pivot_ids)
        self.interior = mesh_interior_labels(sites.reshape(-1, 3), observations).reshape_as(self.raw_sdf)
        self.interior_step.fill_(step)
        return True

    def regularization(self):
        return sdf_regularizers(self.raw_sdf, self.interior)

    def load_state_dict(self, state_dict, strict=True):
        # Normal Trainer restores representation before constructing Adam.
        # A live initialized state cannot silently replace optimizer parameters.
        shape = state_dict["raw_sdf"].shape
        if self.active and self.raw_sdf.shape != shape:
            raise ValueError("restore a differently sized mesh into a fresh state before constructing Adam")
        if not self.active:
            self.raw_sdf = nn.Parameter(torch.empty(shape, device=self.raw_sdf.device, dtype=self.raw_sdf.dtype))
            for name in ("pivot_ids", "cells", "interior"):
                value = state_dict[name]
                old = getattr(self, name)
                setattr(self, name, torch.empty(value.shape, dtype=old.dtype, device=old.device))
        return super().load_state_dict(state_dict, strict=strict)
