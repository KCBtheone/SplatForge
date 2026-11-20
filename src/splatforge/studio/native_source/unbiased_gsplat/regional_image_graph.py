"""CUDA graph replay of the original fixed-image objective and its ATen VJP.

SPDX-License-Identifier: Apache-2.0
Only tensor execution is captured. Native rendering, parameters, population
statistics, finite-gradient checks and optimizer updates stay in their caller.
"""
from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .regional_image_fusion import joint_image_objective


class TensorOutputLayout:
    """Keep original tensor leaves and literal metadata in their original tree."""

    def __init__(self, value):
        leaves, self.spec = tree_flatten(value)
        self.tensor_indices = tuple(i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        self.requires_grad = tuple(leaves[i].requires_grad for i in self.tensor_indices)
        self.literals = [None if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves]
        if not self.tensor_indices or any(
            leaf is not None and type(leaf) not in (bool, int, float, str)
            for leaf in self.literals
        ):
            raise TypeError('Image graph outputs require tensors and literal metadata')

    def tensors(self, value):
        leaves, spec = tree_flatten(value)
        if spec != self.spec or len(leaves) != len(self.literals):
            raise ValueError('Image graph output structure changed')
        indices = tuple(i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        if indices != self.tensor_indices:
            raise ValueError('Image graph output tensor layout changed')
        for i, leaf in enumerate(leaves):
            if i not in self.tensor_indices and leaf != self.literals[i]:
                raise ValueError('Image graph output metadata changed')
        return tuple(leaves[i] for i in self.tensor_indices)

    def restore(self, tensors):
        if len(tensors) != len(self.tensor_indices):
            raise ValueError('Image graph output tensor count changed')
        leaves = self.literals.copy()
        for i, tensor, needs_grad in zip(self.tensor_indices, tensors, self.requires_grad):
            leaves[i] = tensor if needs_grad else tensor.detach()
        return tree_unflatten(leaves, self.spec)


class OriginalImageObjectiveGraph:
    """One original ATen graph pair per actual requires-grad layout.

    Image shapes, strides, dtypes and device must stay fixed. The original
    before/after geometry schedule may change which inputs need gradients, so
    those layouts receive separate captures without changing tensor values.
    The serial caller must consume each forward with backward before reuse.
    """

    def __init__(self):
        self.calls = 0
        self.backward_calls = 0
        self.captures = 0
        self._input_spec = None
        self._metadata = None
        self._graphs = {}

    def __call__(self, *values):
        if not torch.is_grad_enabled():
            raise ValueError('Image graph is only for the original training consumer')
        flat, spec = tree_flatten(values)
        if not flat or any(not isinstance(value, torch.Tensor) for value in flat):
            raise TypeError('Image graph inputs must all be tensors')
        if any(not value.is_cuda for value in flat):
            raise ValueError('Image graph inputs must be CUDA tensors')
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in flat):
            raise ValueError('Image graph preserves the original float32 profile')
        metadata = tuple((tuple(v.shape), tuple(v.stride()), v.dtype, v.device) for v in flat)
        if self._input_spec is None:
            self._input_spec, self._metadata = spec, metadata
        elif spec != self._input_spec or metadata != self._metadata:
            raise ValueError('Image graph input structure, image shape or tensor metadata changed')
        gradient_layout = tuple(value.requires_grad for value in flat)
        if gradient_layout not in self._graphs:
            sample = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in flat)
            output_layout = [None]

            def original_tensor_objective(*tensors):
                original_values = tree_unflatten(list(tensors), spec)
                result = joint_image_objective(*original_values)
                if output_layout[0] is None:
                    output_layout[0] = TensorOutputLayout(result)
                return output_layout[0].tensors(result)

            graph = torch.cuda.make_graphed_callables(
                original_tensor_objective, sample, num_warmup_iters=3,
                allow_unused_input=True)
            self._graphs[gradient_layout] = (graph, output_layout[0])
            self.captures += 1
        graph, layout = self._graphs[gradient_layout]
        result = layout.restore(graph(*flat))
        color = result[0]['_regional_color']
        if not color.requires_grad:
            raise RuntimeError('Original image graph lost its photometric derivative')

        def backward_started(gradient):
            self.backward_calls += 1
            return gradient

        color.register_hook(backward_started)
        self.calls += 1
        return result
