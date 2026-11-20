"""Source-owned sampling replay for legacy checkpoints.

SPDX-License-Identifier: Apache-2.0
Replays RNG consumption only; it does not replay or invent optimizer updates.
"""
from __future__ import annotations
import random


def replay_sampler(updates, seed, training_indices, neighbor_graph, *, has_pgsr):
    rng = random.Random(seed)
    stack = []
    multiview_updates = 0
    for step in range(updates):
        if not stack:
            stack = list(training_indices)
            rng.shuffle(stack)
        index = stack.pop()
        if has_pgsr and neighbor_graph[index] and step >= 1500 and step % 4 == 0:
            rng.choice(neighbor_graph[index])
            multiview_updates += 1
    return {'python_rng':rng.getstate(), 'view_stack':stack,
            'regularizer_updates':max(0,updates-1000), 'multiview_updates':multiview_updates}
