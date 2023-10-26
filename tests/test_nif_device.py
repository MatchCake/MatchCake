import numpy as np
import pytest
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import utils


@pytest.mark.parametrize(
    "gate,target_expectation_value",
    [
        (np.eye(4), 0.0)
    ]
)
def test_single_gate_circuit_expectation_value(gate, target_expectation_value):
    device = NonInteractingFermionicDevice(wires=2)
    mg_params = Matchgate.from_matrix(gate).polar_params
    op = MatchgateOperator(mg_params, wires=[0, 1])
    device.apply(op)
    expectation_value = device.analytic_probability(0)
    check = np.isclose(expectation_value, target_expectation_value)
    assert check, (f"The expectation value is not the correct one. "
                   f"Got {expectation_value} instead of {target_expectation_value}")







