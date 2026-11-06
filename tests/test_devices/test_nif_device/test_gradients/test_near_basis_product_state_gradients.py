import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import CompRxRx
from matchcake.operations.state_preparation import ProductState

from ....configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)


@pytest.mark.parametrize("n_wires", [3, 6])
class TestNearBasisProductStateGradients:
    """A product state close to a basis state has rare outcomes whose Pfaffians are near the float32
    round-off level; their gradients must be kept so the normalized probabilities stay differentiable."""

    ANGLE_SCALE = 1e-3

    @staticmethod
    def _make_circuit(device, n_wires):
        @qml.qnode(device, interface="torch")
        def circuit(thetas, gate_params):
            states = torch.stack([torch.cos(thetas / 2), torch.sin(thetas / 2)], dim=-1)
            ProductState(states, wires=list(range(n_wires)), validate_norm=False)
            for wire in range(n_wires - 1):
                CompRxRx(gate_params[2 * wire : 2 * wire + 2], wires=[wire, wire + 1])
            return qml.probs(wires=list(range(n_wires)))

        return circuit

    @classmethod
    def _inputs(cls, n_wires):
        generator = np.random.default_rng(n_wires)
        thetas = cls.ANGLE_SCALE * generator.uniform(0.5, 1.5, n_wires)
        gate_params = generator.uniform(0.1, 0.5, 2 * (n_wires - 1))
        return torch.tensor(thetas), torch.tensor(gate_params)

    def test_float32_jacobian_matches_float64(self, n_wires):
        thetas, gate_params = self._inputs(n_wires)
        reference_device = NonInteractingFermionicDevice(wires=n_wires)
        reference_circuit = self._make_circuit(reference_device, n_wires)
        reference = torch.autograd.functional.jacobian(lambda t: reference_circuit(t, gate_params), thetas)
        device = NonInteractingFermionicDevice(wires=n_wires, r_dtype=torch.float32, c_dtype=torch.complex64)
        circuit = self._make_circuit(device, n_wires)
        jacobian = torch.autograd.functional.jacobian(
            lambda t: circuit(t, gate_params.to(torch.float32)), thetas.to(torch.float32)
        )
        dominant = int(torch.argmax(reference_circuit(thetas, gate_params)))
        np.testing.assert_allclose(
            jacobian[dominant].double().numpy(),
            reference[dominant].numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
