import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.circuits import random_sptm_operations_generator
from matchcake.devices.sampling_strategies.k_qubits_by_k_qubits_sampling import (
    KQubitsByKQubitsSampling,
)
from matchcake.operations import SptmCompRxRx
from matchcake.operations.state_preparation.product_state import ProductState

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import init_nif_device


class _KSampling(KQubitsByKQubitsSampling):
    NAME = "TestKSampling"
    K = 2


def _exact_distribution(device, num_wires):
    all_states = device.states_to_binary(np.arange(2**num_wires), num_wires)
    probs = np.asarray(device.get_states_probability(all_states, np.arange(num_wires))).reshape(-1)
    return probs / probs.sum()


def _empirical_distribution(samples, num_wires):
    flat = np.asarray(samples).astype(int).reshape(-1, num_wires)
    indices = flat.dot(2 ** np.arange(num_wires)[::-1])
    counts = np.bincount(indices, minlength=2**num_wires)
    return counts / flat.shape[0]


def _non_basis_product_state(num_wires, rng, batch_size=None):
    shape = (num_wires,) if batch_size is None else (batch_size, num_wires)
    angles = rng.uniform(0.2, np.pi - 0.2, shape)
    amplitudes = np.stack([np.cos(angles / 2), np.sin(angles / 2)], axis=-1).astype(complex)
    return ProductState(amplitudes, wires=range(num_wires))


def _exact_distribution_batched(device, num_wires):
    all_states = device.states_to_binary(np.arange(2**num_wires), num_wires)
    probs = np.asarray(device.get_states_probability(all_states, np.arange(num_wires)))  # (2 ** n, batch)
    return probs / probs.sum(axis=0, keepdims=True)


def _empirical_distribution_batched(samples, num_wires, batch_size):
    flat = np.asarray(samples).astype(int)  # (shots, batch, n)
    empirical = np.zeros((2**num_wires, batch_size))
    powers = 2 ** np.arange(num_wires)[::-1]
    for b in range(batch_size):
        indices = flat[:, b, :].dot(powers)
        empirical[:, b] = np.bincount(indices, minlength=2**num_wires) / flat.shape[0]
    return empirical


class TestKQubitsByKQubitsSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "num_wires, k, expected",
        [
            (4, 2, [2]),
            (6, 2, [2, 2]),
            (5, 2, [2, 1]),
            (3, 2, [1]),
            (4, 1, [1, 1, 1]),
            (7, 3, [3, 1]),
            (3, 3, []),
        ],
    )
    def test_build_subset_sizes(self, num_wires, k, expected):
        assert KQubitsByKQubitsSampling.build_subset_sizes(num_wires, k) == expected
        assert k + sum(KQubitsByKQubitsSampling.build_subset_sizes(num_wires, k)) == num_wires

    def test_extend_states_ordering(self):
        prefixes = np.array([[0, 1], [1, 0]])
        added_states = np.array([[0], [1]])
        extended = KQubitsByKQubitsSampling.extend_states(prefixes, added_states, unique=True)
        # Block-major: row = block_idx * n_prefixes + prefix_idx.
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(extended, expected)

    def test_extend_states_deduplicates_prefixes(self):
        prefixes = np.array([[0, 0], [0, 0], [1, 1]])
        added_states = np.array([[0], [1]])
        extended = KQubitsByKQubitsSampling.extend_states(prefixes, added_states, unique=True)
        assert extended.shape == (2 * 2, 3)

    def test_extend_with_inverse_maps_each_prefix(self):
        prefixes = np.array([[0, 0], [1, 1], [0, 0]])
        added_states = np.array([[0], [1]])
        extended, inverse, n_prefixes = KQubitsByKQubitsSampling._extend_with_inverse(prefixes, added_states)
        assert n_prefixes == 2
        unique_prefixes = np.unique(prefixes, axis=0)
        np.testing.assert_array_equal(unique_prefixes[inverse], prefixes)
        assert extended.shape == (2 * n_prefixes, 3)

    def test_extend_with_inverse_without_unique(self):
        prefixes = np.array([[0, 0], [0, 0]])
        added_states = np.array([[0], [1]])
        extended, inverse, n_prefixes = KQubitsByKQubitsSampling._extend_with_inverse(
            prefixes, added_states, unique=False
        )
        assert n_prefixes == 2
        np.testing.assert_array_equal(inverse, np.array([0, 1]))

    def test_scatter_extended_probs_unbatched(self):
        # Two unique prefixes, two candidate blocks. grid[block, prefix].
        n_prefixes, n_blocks = 2, 2
        extended_probs = np.array([0.4, 0.1, 0.6, 0.9])  # row = block * n_prefixes + prefix
        prefix_inverse = np.array([0, 1, 0])  # three samples mapped to prefixes
        probs = KQubitsByKQubitsSampling.scatter_extended_probs(
            extended_probs, prefix_inverse, n_prefixes, n_blocks, batch_shape=(3,)
        )
        expected = np.array([[0.4, 0.6], [0.1, 0.9], [0.4, 0.6]])
        np.testing.assert_allclose(probs, expected, atol=ATOL_APPROX_COMPARISON)

    def test_scatter_extended_probs_batched(self):
        n_prefixes, n_blocks, batch = 2, 2, 3
        # extended_probs shape (n_blocks * n_prefixes, batch)
        extended_probs = np.arange(n_blocks * n_prefixes * batch, dtype=float).reshape(n_blocks * n_prefixes, batch)
        prefix_inverse = np.array([0, 1, 1, 0, 0, 1])  # shots=2, batch=3 flattened
        probs = KQubitsByKQubitsSampling.scatter_extended_probs(
            extended_probs, prefix_inverse, n_prefixes, n_blocks, batch_shape=(2, batch)
        )
        grid = extended_probs.reshape(n_blocks, n_prefixes, batch)
        inverse = prefix_inverse.reshape(2, batch)
        for shot in range(2):
            for b in range(batch):
                for block in range(n_blocks):
                    assert probs[shot, b, block] == grid[block, inverse[shot, b], b]

    def test_scatter_extended_probs_preserves_torch_backend(self):
        extended_probs = torch.tensor([0.4, 0.1, 0.6, 0.9], dtype=torch.float64)
        prefix_inverse = np.array([0, 1])
        probs = KQubitsByKQubitsSampling.scatter_extended_probs(extended_probs, prefix_inverse, 2, 2, batch_shape=(2,))
        assert isinstance(probs, torch.Tensor)

    @pytest.mark.parametrize("num_wires", [2, 4, 5])
    def test_samples_match_exact_distribution(self, num_wires):
        set_seed(TEST_SEED)
        operations = list(random_sptm_operations_generator(15, np.arange(num_wires), op_types=[SptmCompRxRx]))
        device = init_nif_device(wires=num_wires, shots=int(20000))
        device.execute_generator((op for op in operations), output_type="expval")
        exact = _exact_distribution(device, num_wires)

        samples = _KSampling().batch_generate_samples(device, device.get_states_probability)
        empirical = _empirical_distribution(samples, num_wires)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_samples_match_exact_distribution_with_torch_probs(self):
        set_seed(TEST_SEED)
        num_wires = 4
        operations = list(random_sptm_operations_generator(15, np.arange(num_wires), op_types=[SptmCompRxRx]))
        device = init_nif_device(wires=num_wires, shots=int(20000))
        device.execute_generator((op for op in operations), output_type="expval")
        exact = _exact_distribution(device, num_wires)

        def torch_states_prob_func(states, wires):
            probs = device.get_states_probability(np.asarray(states).astype(int), np.asarray(wires))
            return torch.as_tensor(np.asarray(probs), dtype=torch.float64)

        samples = _KSampling().batch_generate_samples(device, torch_states_prob_func)
        assert isinstance(samples, np.ndarray)
        empirical = _empirical_distribution(samples, num_wires)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("num_wires", [3, 4])
    def test_samples_match_exact_distribution_from_product_state(self, num_wires):
        set_seed(TEST_SEED)
        rng = np.random.default_rng(TEST_SEED)
        device = init_nif_device(wires=num_wires, shots=int(40000))
        device.apply_state_prep(_non_basis_product_state(num_wires, rng))
        assert not device.state_prep_op.is_basis_state
        exact = _exact_distribution(device, num_wires)

        samples = _KSampling().batch_generate_samples(device, device.get_states_probability)
        empirical = _empirical_distribution(samples, num_wires)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_samples_match_exact_distribution_product_state_with_gates(self):
        set_seed(TEST_SEED)
        num_wires = 4
        rng = np.random.default_rng(TEST_SEED)
        device = init_nif_device(wires=num_wires, shots=int(40000))
        device.apply_state_prep(_non_basis_product_state(num_wires, rng))
        device.apply([SptmCompRxRx(rng.uniform(0, np.pi, 2), wires=[i, i + 1]) for i in range(num_wires - 1)])
        exact = _exact_distribution(device, num_wires)

        samples = _KSampling().batch_generate_samples(device, device.get_states_probability)
        empirical = _empirical_distribution(samples, num_wires)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("num_wires", [3, 4])
    def test_samples_match_exact_distribution_from_batched_product_state(self, num_wires):
        set_seed(TEST_SEED)
        batch_size = 3
        rng = np.random.default_rng(TEST_SEED)
        device = init_nif_device(wires=num_wires, shots=int(40000))
        device.apply_state_prep(_non_basis_product_state(num_wires, rng, batch_size=batch_size))
        assert device.state_prep_op.batch_size == batch_size
        exact = _exact_distribution_batched(device, num_wires)

        samples = _KSampling().batch_generate_samples(device, device.get_states_probability)
        assert np.asarray(samples).shape == (40000, batch_size, num_wires)
        empirical = _empirical_distribution_batched(samples, num_wires, batch_size)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_samples_match_exact_distribution_batched_product_state_with_gates(self):
        set_seed(TEST_SEED)
        num_wires = 4
        batch_size = 3
        rng = np.random.default_rng(TEST_SEED)
        device = init_nif_device(wires=num_wires, shots=int(40000))
        device.apply_state_prep(_non_basis_product_state(num_wires, rng, batch_size=batch_size))
        device.apply([SptmCompRxRx(rng.uniform(0, np.pi, 2), wires=[i, i + 1]) for i in range(num_wires - 1)])
        exact = _exact_distribution_batched(device, num_wires)

        samples = _KSampling().batch_generate_samples(device, device.get_states_probability)
        assert np.asarray(samples).shape == (40000, batch_size, num_wires)
        empirical = _empirical_distribution_batched(samples, num_wires, batch_size)

        np.testing.assert_allclose(empirical, exact, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_generate_samples_single_state_callback(self):
        set_seed(TEST_SEED)
        num_wires = 4
        device = init_nif_device(wires=num_wires, shots=64)
        device.apply_state_prep(qml.BasisState(np.zeros(num_wires, dtype=int), wires=range(num_wires)))
        samples = _KSampling().generate_samples(device, device.get_state_probability)
        np.testing.assert_array_equal(samples, np.zeros((64, num_wires), dtype=int))
