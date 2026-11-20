"""Explicit semantic roles for two unchanged native reconstruction families.

SPDX-License-Identifier: Apache-2.0
Semantic hair/body roles do not change a family's representation or geometry.
"""
import numpy as np
import torch


def native_family_roles(hair_family='pgsr'):
    if hair_family not in ['pgsr','unbiased']:
        raise ValueError('Hair family must be pgsr or unbiased')
    return hair_family, 'unbiased' if hair_family=='pgsr' else 'pgsr'


def initial_family_masks(hair_mask, hair_family='pgsr'):
    hair,body=native_family_roles(hair_family)
    mask=np.asarray(hair_mask)
    if mask.dtype!=np.bool_ or mask.ndim!=1:
        raise ValueError('Initial semantic hair partition must be a one-dimensional boolean array')
    return {hair:mask,body:~mask}


def ordered_native_layers(layers, hair_family='pgsr'):
    hair,body=native_family_roles(hair_family)
    if set(layers)!= {'pgsr','unbiased'}:
        raise ValueError('Joint roles require both unchanged native families')
    return layers[hair],layers[body]


def native_anchor_weights(policy, hair_family='pgsr'):
    hair,body=native_family_roles(hair_family)
    weights=policy['native_anchor_weights']
    return {hair:weights[0],body:weights[1]}


def selected_native_normals(layers, geometry, hair_family='pgsr'):
    hair,body=ordered_native_layers(layers,hair_family)
    return torch.where(geometry['hair_selected'][...,None],hair['normals'],body['normals'])
