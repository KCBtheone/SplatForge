"""Explicit opacity-reset lifetime for owned commercial quality experiments.

SPDX-License-Identifier: Apache-2.0
This is a selectable engineering schedule, not an author implementation.
"""
from .commercial_model import reset_opacities


def opacity_reset_due(family: str, step: int, *, policy: str) -> bool:
    if family not in ['pgsr', 'unbiased']:
        raise ValueError('Unknown native family')
    if policy not in ['legacy', 'refinement']:
        raise ValueError('Unknown opacity-reset policy')
    if not isinstance(step, int) or step < 0:
        raise ValueError('Step must be a nonnegative zero-based integer')
    scheduled = step > 0 and step % 3000 == 0
    stop = 15000 if family == 'pgsr' else 20000
    return scheduled and (policy == 'legacy' or step < stop)


def reset_opacities_if_due(parameters, optimizer, family, step, *, policy):
    if not opacity_reset_due(family, step, policy=policy):
        return False
    reset_opacities(parameters, optimizer, family)
    return True


def opacity_reset_histories_match(family, completed_updates, *, previous_policy, requested_policy):
    """A prefix may change policy only before an actual reset has differed."""
    if not isinstance(completed_updates, int) or isinstance(completed_updates, bool) or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    # Validate both policies even when the prefix precedes the first reset.
    opacity_reset_due(family, 0, policy=previous_policy)
    opacity_reset_due(family, 0, policy=requested_policy)
    return all(opacity_reset_due(family, step, policy=previous_policy)
               == opacity_reset_due(family, step, policy=requested_policy)
               for step in range(3000, completed_updates, 3000))
