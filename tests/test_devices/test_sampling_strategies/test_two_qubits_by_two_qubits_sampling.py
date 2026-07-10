import numpy as np
import pennylane as qml
import pytest

from matchcake.circuits import random_sptm_operations_generator
from matchcake.devices.sampling_strategies.two_qubits_by_two_qubits_sampling import (
    TwoQubitsByTwoQubitsSampling,
)
from matchcake.operations import SptmCompRxRx

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import init_nif_device


def _setup_zero_state_device(n_wires: int, shots: int):
    device = init_nif_device(wires=n_wires, shots=shots)
    state_prep_op = qml.BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires))
    device.apply_state_prep(state_prep_op)
    return device


class TestTwoQubitsByTwoQubitsSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_k_is_two(self):
        assert TwoQubitsByTwoQubitsSampling.K == 2

    @pytest.mark.parametrize("n_wires", [2, 4])
    def test_generate_samples_shape(self, n_wires):
        shots = 16
        device = _setup_zero_state_device(n_wires, shots)
        strategy = TwoQubitsByTwoQubitsSampling()
        samples = strategy.generate_samples(device, device.get_state_probability)
        assert samples.shape == (shots, n_wires)

    def test_generate_samples_all_zeros_for_zero_state(self):
        n_wires = 2
        shots = 16
        device = _setup_zero_state_device(n_wires, shots)
        strategy = TwoQubitsByTwoQubitsSampling()
        samples = strategy.generate_samples(device, device.get_state_probability)
        np.testing.assert_array_equal(samples, np.zeros((shots, n_wires), dtype=int))

    @pytest.mark.parametrize("num_wires", [3, 5])
    def test_samples_match_exact_distribution(self, num_wires):
        set_seed(TEST_SEED)
        operations = list(random_sptm_operations_generator(15, np.arange(num_wires), op_types=[SptmCompRxRx]))
        device = init_nif_device(wires=num_wires, shots=int(20000))
        device.execute_generator((op for op in operations), output_type="expval")

        all_states = device.states_to_binary(np.arange(2**num_wires), num_wires)
        exact = np.asarray(device.get_states_probability(all_states, np.arange(num_wires))).reshape(-1)
        exact = exact / exact.sum()

        samples = TwoQubitsByTwoQubitsSampling().batch_generate_samples(device, device.get_states_probability)
        flat = np.asarray(samples).astype(int).reshape(-1, num_wires)
        indices = flat.dot(2 ** np.arange(num_wires)[::-1])
        empirical = np.bincount(indices, minlength=2**num_wires) / flat.shape[0]

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_sampler_unbiased_at_large_n(self):
        """Regression for the large-N autoregressive sampling bias.

        At many wires the deep-prefix conditional joint masses are ~2**-n. A normalisation
        that floored such masses (an additive epsilon in ``random_index``) collapsed the
        inverse-CDF onto the last block, biasing samples toward ``|1...1>``. The sampler's
        mean Hamming weight must instead match ``sum_k P(x_k = 1)`` from the (correct)
        one-wire marginals. With the bug present this drifts by tens of percent.
        """
        set_seed(TEST_SEED)
        num_wires = 44
        shots = 2000
        operations = list(random_sptm_operations_generator(20, np.arange(num_wires), op_types=[SptmCompRxRx]))

        device = init_nif_device(wires=num_wires, shots=shots)
        device.execute_generator((op for op in operations), output_type="expval")
        one_wire_p1 = np.array(
            [
                float(np.real(np.asarray(device.get_states_probability(np.array([1]), np.array([k])))))
                for k in range(num_wires)
            ]
        )
        expected_ones = one_wire_p1.sum()

        device.execute_generator((op for op in operations), output_type="expval")
        samples = TwoQubitsByTwoQubitsSampling().batch_generate_samples(device, device.get_states_probability)
        sampled_ones = np.asarray(samples).astype(int).reshape(-1, num_wires).sum(axis=1).mean()

        assert abs(sampled_ones - expected_ones) / expected_ones < 0.05
