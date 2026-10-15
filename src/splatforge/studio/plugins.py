"""Independent, extensible component registries."""

from __future__ import annotations

from importlib.metadata import entry_points


class ComponentRegistry:
    def __init__(self, group):
        self.group = group
        self._items = {}
        self._loaded = False

    def register(self, name, component):
        if name in self._items:
            raise ValueError(f"duplicate {self.group} component: {name}")
        self._items[name] = component
        return component

    def load_plugins(self):
        if not self._loaded:
            self._loaded = True
            for point in entry_points(group=self.group):
                self.register(point.name, point.load())

    def get(self, name):
        self.load_plugins()
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown {self.group} component {name}; choices: {', '.join(self.names())}"
            ) from exc

    def names(self):
        self.load_plugins()
        return sorted(self._items)
