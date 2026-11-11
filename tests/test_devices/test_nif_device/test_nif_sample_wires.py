import numpy as np
import pennylane as qml
import pytest

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import SptmCompRxRx
from matchcake.utils import torch_utils

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_SHOTS = 32
N_STATISTICAL_SHOTS = 8192


class TestNonInteractingFermionicDeviceSampleWires:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "basis_state, sample_wires",
        [
            ([1, 0, 1, 1], [0]),
            ([1, 0, 1, 1], [1]),
            ([1, 0, 1, 1], [3, 1]),
            ([1, 0, 1, 1], [0, 3, 1]),
            ([0, 1, 1, 0, 1], [4, 0, 2]),
            ([0, 1, 1, 0, 1], [0, 1, 2, 3, 4]),
            ([0, 1, 1, 0, 1], [4, 3, 2, 1, 0]),
        ],
    )
    def test_sample_wires_on_basis_state(self, basis_state, sample_wires):
        num_wires = len(basis_state)
        device = NonInteractingFermionicDevice(wires=num_wires, shots=N_SHOTS)

        @qml.qnode(device)
        def circuit():
            qml.BasisState(np.array(basis_state), wires=range(num_wires))
            return qml.sample(wires=sample_wires)

        samples = torch_utils.to_numpy(circuit()).astype(int)
        expected = np.tile(np.asarray(basis_state)[sample_wires], (N_SHOTS, 1))
        np.testing.assert_array_equal(samples, expected)

    def test_sample_without_wires_returns_all_wires(self):
        basis_state = np.array([1, 0, 1, 1])
        device = NonInteractingFermionicDevice(wires=4, shots=N_SHOTS)

        @qml.qnode(device)
        def circuit():
            qml.BasisState(basis_state, wires=range(4))
            return qml.sample()

        samples = torch_utils.to_numpy(circuit()).astype(int)
        np.testing.assert_array_equal(samples, np.tile(basis_state, (N_SHOTS, 1)))

    @pytest.mark.parametrize("sample_wires", [[0], [2, 0], [3, 1, 2]])
    def test_sample_wires_frequencies_match_probs(self, sample_wires):
        num_wires = 4
        params = np.random.uniform(0, 2 * np.pi, size=(num_wires - 1, 2))

        def circuit(measurement):
            qml.BasisState(np.array([1, 0, 1, 0]), wires=range(num_wires))
            for wire0, angles in zip(range(num_wires - 1), params):
                SptmCompRxRx(angles, wires=[wire0, wire0 + 1])
            return measurement(wires=sample_wires)

        sampled_device = NonInteractingFermionicDevice(wires=num_wires, shots=N_STATISTICAL_SHOTS)
        analytic_device = NonInteractingFermionicDevice(wires=num_wires)
        samples = torch_utils.to_numpy(qml.QNode(circuit, sampled_device)(qml.sample)).astype(int)
        probs = torch_utils.to_numpy(qml.QNode(circuit, analytic_device)(qml.probs))

        assert samples.shape == (N_STATISTICAL_SHOTS, len(sample_wires))
        powers_of_two = 2 ** np.arange(len(sample_wires))[::-1]
        state_indices = samples @ powers_of_two
        frequencies = np.bincount(state_indices, minlength=2 ** len(sample_wires)) / N_STATISTICAL_SHOTS
        np.testing.assert_allclose(frequencies, probs, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_sample_wires_on_batched_samples(self):
        num_wires = 3
        batch_size = 4
        device = NonInteractingFermionicDevice(wires=num_wires, shots=N_SHOTS)
        device._samples = np.random.randint(0, 2, size=(N_SHOTS, batch_size, num_wires))
        sample_wires = [2, 0]

        samples = device._process_measurement(qml.sample(wires=sample_wires))
        # PennyLane layout: (batch, shots, len(sample_wires))
        np.testing.assert_array_equal(samples, np.moveaxis(device._samples, 0, -2)[..., sample_wires])
