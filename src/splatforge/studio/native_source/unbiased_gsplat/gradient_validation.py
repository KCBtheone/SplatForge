"""Reject invalid Adam inputs with one successful-path device synchronization.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import torch


REFERENCE_SURFEL_NORMAL_START_STEP = 7000
REFERENCE_SURFEL_CONVERGENCE_START_STEP = 10000


def validate_adam_inputs(loss, families, exposure=None):
    if loss.numel() != 1:
        raise ValueError('Adam requires a scalar loss')
    named = [(family + ':' + name, parameter)
             for family, parameters in families.items() for name, parameter in parameters.items()]
    if exposure is not None:
        named.append(('exposure', exposure))
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError('Missing derivative before Adam: ' + name)
    flags = [torch.isfinite(loss).all()]
    flags.extend(torch.isfinite(parameter.grad).all() for _, parameter in named)
    if bool(torch.stack(flags).all()):
        return
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError('Non-finite scalar before Adam')
    for name, parameter in named:
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('Non-finite derivative before Adam: ' + name)


def population_statistics_needed(family, step):
    return step < (15000 if family == 'pgsr' else 20000)


def geometry_regularizers_active(family, step, *, unbiased_start_iter=1000, smoke=False):
    """Owned caller schedule; the opt-in timing control changes no tensor formula."""
    if family not in ('pgsr', 'unbiased'):
        raise ValueError('Unknown native geometry family')
    if type(step) is not int or step < 0:
        raise ValueError('Geometry step must be a nonnegative integer')
    if type(unbiased_start_iter) is not int or unbiased_start_iter < 0:
        raise ValueError('Unbiased start iteration must be a nonnegative integer')
    if type(smoke) is not bool:
        raise ValueError('Smoke schedule flag must be boolean')
    start = unbiased_start_iter if family == 'unbiased' else 1000
    return smoke or step >= start


def validate_view_batch_size(batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('View batch size must be a positive integer')
    return batch_size


def surfel_regularizer_activity(step, *, dual_geometry=False, reference_regularizers=False,
                                unbiased_start_iter=1000, smoke=False):
    """Zero-based step identifies the upcoming actual update, including resume."""
    if type(dual_geometry) is not bool:
        raise TypeError('dual_geometry must be boolean')
    if type(reference_regularizers) is not bool:
        raise TypeError('reference_regularizers must be boolean')
    legacy = geometry_regularizers_active('unbiased', step,
        unbiased_start_iter=unbiased_start_iter, smoke=smoke)
    if reference_regularizers or dual_geometry:
        return step >= REFERENCE_SURFEL_CONVERGENCE_START_STEP, step >= REFERENCE_SURFEL_NORMAL_START_STEP
    return legacy, legacy


def validate_orientation_resume(previous_config, orientation_init):
    if orientation_init not in ('uniform_sphere','positive_uniform','sfm_plane'):
        raise ValueError('Unknown Surfel orientation initialization')
    if previous_config.get('unbiased_orientation_init','uniform_sphere')!=orientation_init:
        raise ValueError('Resume Surfel orientation initialization mismatch; start the changed recipe from raw SfM')


def validate_surface_midpoint_resume(previous_config, surface_midpoint):
    if type(surface_midpoint) is not bool:
        raise TypeError('Surfel surface midpoint must be boolean')
    previous = previous_config.get('unbiased_surface_midpoint', False)
    if type(previous) is not bool or previous != surface_midpoint:
        raise ValueError('Resume Surfel surface output mismatch; start the changed recipe from raw SfM')


def validate_front_convergence_resume(previous_config, front_convergence):
    if type(front_convergence) is not bool:
        raise TypeError('Surfel front convergence must be boolean')
    previous = previous_config.get('unbiased_front_convergence', False)
    if type(previous) is not bool or previous != front_convergence:
        raise ValueError('Resume Surfel convergence contract mismatch; start the changed recipe from raw SfM')
    if front_convergence and previous_config.get('unbiased_convergence_units') != 'normalized_camera_radius':
        raise ValueError('Resume Surfel convergence units mismatch; start the changed recipe from raw SfM')


def validate_coherent_midpoint_resume(previous_config, coherent_midpoint):
    if type(coherent_midpoint) is not bool:
        raise TypeError('Surfel coherent midpoint must be boolean')
    previous=previous_config.get('unbiased_coherent_midpoint',False)
    if type(previous) is not bool or previous != coherent_midpoint:
        raise ValueError('Resume Surfel midpoint derivative mismatch; start the changed recipe from raw SfM')


def validate_input_photo_consensus_resume(previous_config, input_photo_consensus):
    if type(input_photo_consensus) is not bool:
        raise TypeError('Input photo consensus must be boolean')
    previous=previous_config.get('unbiased_input_photo_consensus',False)
    if type(previous) is not bool or previous != input_photo_consensus:
        raise ValueError('Resume input photo consensus mismatch; start the changed recipe from raw SfM')


def validate_terminal_surface_resume(previous_config, terminal_surface):
    if type(terminal_surface) is not bool:
        raise TypeError('Surfel terminal surface must be boolean')
    previous = previous_config.get('unbiased_terminal_surface', False)
    if type(previous) is not bool or previous != terminal_surface:
        raise ValueError('Resume Surfel terminal surface mismatch; start the changed recipe from raw SfM')


def batch_loss_for_backward(loss, batch_size):
    """Average complete original per-view objectives before one optimizer step."""
    validate_view_batch_size(batch_size)
    if not isinstance(loss, torch.Tensor) or not loss.is_floating_point() or loss.numel() != 1:
        raise ValueError('A view loss must be a floating scalar tensor')
    return loss if batch_size == 1 else loss / batch_size


def unscale_view_gradient(gradient, batch_size):
    """Recover each view's original screen gradient for population statistics."""
    validate_view_batch_size(batch_size)
    if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point():
        raise ValueError('Screen statistics require a floating gradient tensor')
    return gradient.clone() if batch_size == 1 else gradient * batch_size


def validate_view_batch_resume(previous_config, batch_size, *, training_state_present):
    validate_view_batch_size(batch_size)
    previous = validate_view_batch_size(previous_config.get('view_batch_size', 1))
    if previous != batch_size:
        raise ValueError('Resume view batch size mismatch; start the changed recipe from raw SfM')
    if batch_size > 1 and not training_state_present:
        raise ValueError('Batched checkpoint requires saved training state')
