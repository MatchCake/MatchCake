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

from . import matchgate_parameter_sets  # Alias
from . import matchgate_parameter_sets as mgp
from . import ml, utils
from .devices import NIFDevice, NonInteractingFermionicDevice
from .matchgate_parameter_sets import matchgate_params
from .matchgate_parameter_sets.matchgate_polar_params import MatchgatePolarParams
from .matchgate_parameter_sets.matchgate_standard_params import MatchgateStandardParams
from .observables import BatchHamiltonian
from .operations import MatchgateOperation
from .utils import math

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
