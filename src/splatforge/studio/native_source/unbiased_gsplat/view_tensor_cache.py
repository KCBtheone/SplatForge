"""Optional exact memoization of immutable converted original-view inputs."""
from __future__ import annotations

import torch


class _PaletteTensor:
    """Store original scalar values and their indices without rounding."""
    def __init__(self, values, indices):
        self.values=values;self.indices=indices

    def decode(self):
        return self.values[self.indices.to(torch.int64)]

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.values,self.indices))

    @property
    def storage_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in (self.values,self.indices))


def _palette(value):
    if value.dtype!=torch.float32:return value
    if value.requires_grad:raise ValueError('View palettes require immutable input tensors')
    values,indices=torch.unique(value,sorted=True,return_inverse=True)
    if values.numel()>torch.iinfo(torch.uint8).max+1:return value
    if values.numel()*values.element_size()+value.numel()>=value.numel()*value.element_size():return value
    return _PaletteTensor(values.clone(),indices.to(torch.uint8).reshape(value.shape))


class ViewTensorCache:
    def __init__(self, loader, converter, *, enabled=False, palette=False):
        if palette and not enabled:raise ValueError('View palettes require the enabled input cache')
        self.loader=loader;self.converter=converter;self.enabled=enabled;self.palette=palette;self.entries={}

    def _store(self,index):
        if index not in self.entries:
            item=self.converter(self.loader(index))
            if self.palette:
                item={key:_palette(value) if key in ('rgb','foreground','hair','body','detail') else value
                      for key,value in item.items()}
            self.entries[index]=item

    def load(self,index):
        if not self.enabled:return self.converter(self.loader(index))
        self._store(index)
        item=self.entries[index]
        if self.palette:return {key:value.decode() if isinstance(value,_PaletteTensor) else value for key,value in item.items()}
        return item

    def preload(self,indices):
        if self.enabled:
            for index in indices:self._store(index)

    @property
    def bytes(self):
        return sum(t.bytes if isinstance(t,_PaletteTensor) else t.numel()*t.element_size()
                   for item in self.entries.values() for t in item.values())

    @property
    def palette_tensors(self):
        return sum(isinstance(t,_PaletteTensor) for item in self.entries.values() for t in item.values())

    @property
    def storage_bytes(self):
        return sum(t.storage_bytes if isinstance(t,_PaletteTensor) else t.untyped_storage().nbytes()
                   for item in self.entries.values() for t in item.values())
