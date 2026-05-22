import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.sampling_strategies.qubit_by_qubit_sampling import QubitByQubitSampling

from ...configs import TEST_SEED, set_seed
from .. import init_nif_device


def _setup_zero_state_device(n_wires: int, shots: int):
    device = init_nif_device(wires=n_wires, shots=shots)
    state_prep_op = qml.BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires))
    device.apply_state_prep(state_prep_op)
    return device


class TestQubitByQubitSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_generate_samples_shape(self, n_wires):
        shots = 32
        device = _setup_zero_state_device(n_wires, shots)
        strategy = QubitByQubitSampling()
        samples = strategy.generate_samples(device, device.get_state_probability)
        assert samples.shape == (shots, n_wires)

    def test_generate_samples_all_zeros_for_zero_state(self):
        n_wires = 2
        shots = 64
        device = _setup_zero_state_device(n_wires, shots)
        strategy = QubitByQubitSampling()
        samples = strategy.generate_samples(device, device.get_state_probability)
        np.testing.assert_array_equal(samples, np.zeros((shots, n_wires), dtype=int))
