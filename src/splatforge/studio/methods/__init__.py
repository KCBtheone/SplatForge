"""Methods are recipes and factories, not runtime or trainer branches."""

from ..plugins import ComponentRegistry
from .ambisur import AmbiSuR
from .gof import GOF
from .radegs import RaDeGS

methods = ComponentRegistry("splatforge.methods")
methods.register("radegs", RaDeGS)
methods.register("ambisur", AmbiSuR)
methods.register("gof", GOF)

__all__ = ["methods", "RaDeGS", "AmbiSuR", "GOF"]
