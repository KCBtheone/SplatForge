"""Methods are recipes and factories, not runtime or trainer branches."""

from ..plugins import ComponentRegistry
from .ambisur import AmbiSuR
from .gof import GOF
from .radegs import RaDeGS
from .milo import MILo
from .vcrgaus import VCRGauS
from .native import PGSR, UnbiasedSurfel

methods = ComponentRegistry("splatforge.methods")
methods.register("radegs", RaDeGS)
methods.register("ambisur", AmbiSuR)
methods.register("gof", GOF)
methods.register("milo", MILo)
methods.register("vcrgaus", VCRGauS)
methods.register("pgsr", PGSR)
methods.register("unbiased-surfel", UnbiasedSurfel)

__all__ = ["methods", "RaDeGS", "AmbiSuR", "GOF", "MILo", "VCRGauS", "PGSR", "UnbiasedSurfel"]
