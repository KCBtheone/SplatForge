"""Exact center/radius tile lists, constructed once per Gaussian footprint."""

from __future__ import annotations

import math

import numpy as np


def tile_members(bounds, *, width, height, tile_size, near, far):
    columns, rows = math.ceil(width / tile_size), math.ceil(height / tile_size)
    members = [[] for _ in range(columns * rows)]
    if not np.isfinite(bounds).all():
        raise FloatingPointError("non-finite projected bounds")
    # Global stable depth order induces exactly the original stable local order.
    for index in np.argsort(bounds[:, 3], kind="stable"):
        x, y, radius, z = bounds[index]
        if (
            not near < z < far
            or x + radius < 0
            or y + radius < 0
            or x - radius > width
            or y - radius > height
        ):
            continue
        # The reference includes both sides of exact tile-edge tangencies.
        left = max(0, math.ceil(float((x - radius) / tile_size)) - 1)
        right = min(columns - 1, math.floor(float((x + radius) / tile_size)))
        top = max(0, math.ceil(float((y - radius) / tile_size)) - 1)
        bottom = min(rows - 1, math.floor(float((y + radius) / tile_size)))
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                members[row * columns + col].append(index)
    return [np.asarray(items, dtype=np.int64) for items in members]
