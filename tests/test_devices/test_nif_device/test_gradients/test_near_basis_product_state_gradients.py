import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice, utils
from matchcake.operations import CompRxRx
from matchcake.operations.state_preparation import ProductState


@pytest.mark.parametrize("n_wires", [3, 6])
@pytest.mark.parametrize(
    "r_dtype, c_dtype",
    [(torch.float64, torch.complex128), (torch.float32, torch.complex64)],
)
class TestNearBasisProductStateGradients:
    """A product state close to a basis state has rare outcomes whose Pfaffians are near the float32
    round-off level. They are genuine, so the product-state strategy must never request the round-off
    gradient mask, which would zero their gradients. The float32 values themselves are at round-off
    there and differ between platforms, so the test checks the request rather than the values."""

    ANGLE_SCALE = 1e-3

    @staticmethod
    def _make_circuit(device, n_wires, product_state):
        @qml.qnode(device, interface="torch")
        def circuit(thetas, gate_params):
            if product_state:
                states = torch.stack([torch.cos(thetas / 2), torch.sin(thetas / 2)], dim=-1)
                ProductState(states, wires=list(range(n_wires)), validate_norm=False)
            else:
                qml.BasisState(np.arange(n_wires) % 2 == 0, wires=range(n_wires))
            for wire in range(n_wires - 1):
                CompRxRx(gate_params[2 * wire : 2 * wire + 2], wires=[wire, wire + 1])
            return qml.probs(wires=list(range(n_wires)))

        return circuit

    @staticmethod
    def _spy_mask_requests(monkeypatch):
        requests = []
        original = utils.pfaffian

        def spy(*args, **kwargs):
            requests.append(bool(kwargs.get("mask_roundoff_gradient", False)))
            return original(*args, **kwargs)

        monkeypatch.setattr(utils, "pfaffian", spy)
        return requests

    @classmethod
    def _inputs(cls, n_wires, r_dtype):
        generator = np.random.default_rng(n_wires)
        thetas = cls.ANGLE_SCALE * generator.uniform(0.5, 1.5, n_wires)
        gate_params = generator.uniform(0.1, 0.5, 2 * (n_wires - 1))
        return torch.tensor(thetas, dtype=r_dtype), torch.tensor(gate_params, dtype=r_dtype)

    @classmethod
    def _jacobian(cls, n_wires, r_dtype, c_dtype, product_state):
        device = NonInteractingFermionicDevice(wires=n_wires, r_dtype=r_dtype, c_dtype=c_dtype)
        circuit = cls._make_circuit(device, n_wires, product_state)
        thetas, gate_params = cls._inputs(n_wires, r_dtype)
        return torch.autograd.functional.jacobian(lambda t: circuit(t, gate_params), thetas)

    def test_product_state_does_not_request_roundoff_mask(self, monkeypatch, n_wires, r_dtype, c_dtype):
        requests = self._spy_mask_requests(monkeypatch)
        jacobian = self._jacobian(n_wires, r_dtype, c_dtype, product_state=True)
        assert len(requests) > 0
        assert not any(requests)
        assert torch.all(torch.isfinite(jacobian))

    def test_basis_state_requests_roundoff_mask(self, monkeypatch, n_wires, r_dtype, c_dtype):
        requests = self._spy_mask_requests(monkeypatch)
        self._jacobian(n_wires, r_dtype, c_dtype, product_state=False)
        assert len(requests) > 0
        assert all(requests)
