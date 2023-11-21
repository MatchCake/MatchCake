from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial

from msim.utils import PAULI_Z, PAULI_X
from .test_single_matchgate_circuit import single_matchgate_circuit


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    nif_device = NonInteractingFermionicDevice(
        wires=kwargs.get("wires", 2), prob_strategy=kwargs.get("prob_strategy", "lookup_table")
    )
    qubit_device = qml.device('default.qubit', wires=kwargs.get("wires", 2), shots=kwargs.get("shots", 1))
    qubit_device.operations.add(MatchgateOperator)
    return nif_device, qubit_device

