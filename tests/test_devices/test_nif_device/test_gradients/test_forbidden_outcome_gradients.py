import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import CompRxRx
from matchcake.operations.single_particle_transition_matrices import SptmCompRxRx

from ....configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)


@pytest.mark.parametrize(
    "contraction_strategy",
    ["none", "forward", "horizontal", "vertical", "neighbours"],
)
class TestForbiddenOutcomeGradients:
    """The probability of a parity-forbidden outcome is zero up to round-off; it must not leak an
    O(1) gradient into the Jacobian of the measured probabilities."""

    PARAMS = np.array([1.53546487, 2.85139109])
    XY_ANGLES = [0.4, 0.9, 1.7, 0.6]

    @classmethod
    def _make_circuit(cls, device, gate_cls, n_wires):
        @qml.qnode(device, interface="torch")
        def circuit(params):
            # An alternating basis state has a fixed parity, so the outcomes of the other parity have
            # zero probability.
            qml.BasisState(np.arange(n_wires) % 2 == 0, wires=range(n_wires))
            for wire in range(n_wires - 1):
                qml.IsingXY(cls.XY_ANGLES[wire], wires=[wire, wire + 1])
            gate_cls(params, wires=[0, 1])
            qml.IsingXY(0.3, wires=[0, 1])
            return qml.probs(wires=range(n_wires))

        return circuit

    @classmethod
    def _reference_jacobian(cls, n_wires):
        circuit = cls._make_circuit(qml.device("default.qubit", wires=n_wires), CompRxRx, n_wires)
        return torch.autograd.functional.jacobian(circuit, torch.tensor(cls.PARAMS)).numpy()

    @pytest.mark.parametrize("n_wires", [3, 4])
    @pytest.mark.parametrize("gate_cls", [SptmCompRxRx, CompRxRx])
    @pytest.mark.parametrize(
        "r_dtype, c_dtype",
        [(torch.float64, torch.complex128), (torch.float32, torch.complex64)],
    )
    def test_jacobian_matches_default_qubit(self, contraction_strategy, gate_cls, r_dtype, c_dtype, n_wires):
        device = NonInteractingFermionicDevice(
            wires=n_wires,
            contraction_strategy=contraction_strategy,
            r_dtype=r_dtype,
            c_dtype=c_dtype,
        )
        circuit = self._make_circuit(device, gate_cls, n_wires)
        jacobian = torch.autograd.functional.jacobian(circuit, torch.tensor(self.PARAMS, dtype=r_dtype))
        np.testing.assert_allclose(
            jacobian.double().numpy(),
            self._reference_jacobian(n_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
