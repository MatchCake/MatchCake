"""
MatchCake is a Python library for simulating non-interacting systems of fermions using Matchgate Circuits.
"""

import importlib_metadata

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2023, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/MatchCake"
__package__ = "matchcake"
__version__ = importlib_metadata.version(__package__)

import warnings

from . import ml, utils
from .devices import NIFDevice, NonInteractingFermionicDevice
from .observables import BatchHamiltonian
from .operations import MatchgateOperation
from .utils import math

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
