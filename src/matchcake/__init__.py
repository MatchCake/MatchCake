"""
MatchCake is a Python library for simulating non-interacting systems of fermions using Matchgate Circuits.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2023, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/MatchCake"
__version__ = "0.0.4-beta3"

from .base import Matchgate
from .matchgate_parameter_sets import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from matchcake import matchgate_parameter_sets as mps  # Alias
from .operations import (
    MatchgateOperation,
)
from .devices import (
    NonInteractingFermionicDevice,
    NIFDevice,
)
from . import utils
from . import ml
from .utils import math

from .observables import (
    BatchHamiltonian,
)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
