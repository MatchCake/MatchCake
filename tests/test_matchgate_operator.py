from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from pfapack import pfaffian

from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial

from msim.utils import PAULI_Z, PAULI_X
from .configs import N_RANDOM_TESTS_PER_CASE
from .test_nif_device import single_matchgate_circuit

np.random.seed(42)


@pytest.mark.parametrize(
    "params,k,binary_state,observable",
    [
        (  # fSWAP
            mps.MatchgateStandardParams(
                a=PAULI_Z[0, 0], b=PAULI_Z[0, 1], c=PAULI_Z[1, 0], d=PAULI_Z[1, 1],
                w=PAULI_X[0, 0], x=PAULI_X[0, 1], y=PAULI_X[1, 0], z=PAULI_X[1, 1]
            ),
            0, "01",
            np.array([
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [-1, -1, 0, 0],
                [-1, 0, 0, 0]
            ])
        )
    ]
)
def test_single_matchgate_obs_on_specific_cases(params, k, binary_state, observable):
    nif_device = NonInteractingFermionicDevice(wires=2)
    nif_device.apply(MatchgateOperator(params, wires=[0, 1]))
    state = utils.binary_state_to_state(binary_state)
    pred_obs = nif_device.lookup_table.get_observable(k, state)
    pred_pf = pfaffian.pfaffian(pred_obs)
    pf = pfaffian.pfaffian(observable)
    assert np.allclose(pred_obs, observable), (
        f"The observable is not the correct one. Got \n{pred_obs} instead of \n{observable}"
    )


