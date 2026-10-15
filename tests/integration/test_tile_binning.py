import numpy as np

from splatforge.studio.tile_binning import tile_members


def test_sparse_binning_matches_every_exhaustive_tile_and_stable_order():
    rng = np.random.default_rng(703)
    bounds = rng.normal(size=(250, 4)).astype(np.float32)
    bounds[:, :2] = bounds[:, :2] * 35 + 16
    bounds[:, 2] = rng.uniform(0, 45, len(bounds))
    bounds[:, 3] = rng.uniform(-1, 10, len(bounds))
    bounds[0] = [16, 16, 0, 2]  # inclusive tangent on four tile boundaries
    bounds[1] = [40, 24, 7, 2]  # partial last tile
    result = tile_members(bounds, width=33, height=21, tile_size=16, near=0.01, far=9)
    index = 0
    for top in range(0, 21, 16):
        for left in range(0, 33, 16):
            height, width = min(16, 21 - top), min(16, 33 - left)
            visible = (
                (bounds[:, 0] + bounds[:, 2] >= left)
                & (bounds[:, 0] - bounds[:, 2] <= left + width)
                & (bounds[:, 1] + bounds[:, 2] >= top)
                & (bounds[:, 1] - bounds[:, 2] <= top + height)
                & (bounds[:, 3] > 0.01)
                & (bounds[:, 3] < 9)
            )
            ids = np.flatnonzero(visible)
            expected = ids[np.argsort(bounds[ids, 3], kind="stable")]
            np.testing.assert_array_equal(result[index], expected)
            index += 1
