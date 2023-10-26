"""
MSim is a Python library for simulating non-interacting fermionic systems using the Matchgates.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2023, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/JeremieGince/FermionicSimulation"
__version__ = "0.0.1-beta0"

from .matchgate import Matchgate
from .matchgate_parameter_sets import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianParams,
    MatchgateComposedHamiltonianParams,
)
from .matchgate_operator import MatchgateOperator
from .nif_device import NonInteractingFermionicDevice

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
