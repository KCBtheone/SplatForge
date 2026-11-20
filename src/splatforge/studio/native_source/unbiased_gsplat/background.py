"""Scene background composition for premultiplied native-family RGB.

SPDX-License-Identifier: Apache-2.0
The background carries no depth, normal, opacity or trainable geometry.
"""
from __future__ import annotations
import torch
from torch import Tensor


def composite_background(rgb: Tensor, alpha: Tensor, background: Tensor | None) -> Tensor:
    """Complete foreground radiance using its unconsumed transmittance."""
    if background is None:
        return rgb
    if rgb.shape[-1] != 3 or alpha.shape != rgb.shape[:-1]:
        raise ValueError('Expected RGB and its matching accumulated alpha')
    value = torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device)
    return rgb + (1-alpha[..., None])*value
