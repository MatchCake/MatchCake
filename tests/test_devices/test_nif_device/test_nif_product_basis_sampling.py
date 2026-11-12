import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.sampling_strategies import ProductBasisSampling

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_SHOTS = 20_000
N_WIRES = 3


class TestNonInteractingFermionicDeviceProductBasisSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def setup_method(self):
        # The statistical tests must not depend on the tests run before them on the same worker.
        set_seed(TEST_SEED)

    @staticmethod
    def operations() -> list:
        return [
            qml.BasisState(np.array([0, 1, 0]), wires=range(N_WIRES)),
            qml.SingleExcitation(0.7, wires=[0, 1]),
            qml.IsingXX(0.4, wires=[1, 2]),
            qml.IsingYY(0.9, wires=[0, 1]),
        ]

    @classmethod
    def circuit(cls, measurement_fn):
        # Building the operations inside the QNode queues them.
        cls.operations()
        return measurement_fn()

    @classmethod
    def run(cls, measurement_fn, device):
        return qml.QNode(lambda: cls.circuit(measurement_fn), device)()

    @pytest.mark.parametrize(
        "observable_fn",
        [
            lambda: qml.X(0),
            lambda: qml.Y(1),
            lambda: qml.X(0) @ qml.X(1),
            lambda: qml.Y(0) @ qml.Y(1) @ qml.Z(2),
            lambda: qml.Z(0) @ qml.Y(1) @ qml.X(2),
            lambda: qml.Hamiltonian([0.5, 2.0, -1.0], [qml.X(0) @ qml.X(1), qml.Z(0), qml.Y(1) @ qml.Y(2)]),
        ],
    )
    def test_expval_matches_default_qubit(self, observable_fn):
        expected = self.run(lambda: qml.expval(observable_fn()), qml.device("default.qubit", wires=N_WIRES))
        estimate = self.run(
            lambda: qml.expval(observable_fn()), NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS)
        )
        np.testing.assert_allclose(
            float(estimate), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_several_expvals_match_default_qubit(self):
        def measurements():
            return [qml.expval(qml.X(0)), qml.expval(qml.Y(1) @ qml.Y(2)), qml.expval(qml.Z(0))]

        expected = self.run(measurements, qml.device("default.qubit", wires=N_WIRES))
        estimate = self.run(measurements, NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS))
        np.testing.assert_allclose(
            np.asarray(estimate, dtype=float),
            np.asarray(expected, dtype=float),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_sample_of_pauli_product(self):
        expected = self.run(lambda: qml.expval(qml.X(0) @ qml.Y(1)), qml.device("default.qubit", wires=N_WIRES))
        samples = np.asarray(
            self.run(
                lambda: qml.sample(qml.X(0) @ qml.Y(1)), NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS)
            )
        )
        assert samples.shape == (N_SHOTS,)
        np.testing.assert_array_equal(np.unique(samples), np.array([-1.0, 1.0]))
        np.testing.assert_allclose(
            np.mean(samples), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_sample_with_wires_next_to_rotated_wire(self):
        def measurements():
            return [qml.sample(qml.X(0)), qml.sample(wires=[1, 2])]

        x_samples, bits = self.run(measurements, NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS))
        expected = self.run(
            lambda: [qml.expval(qml.X(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))],
            qml.device("default.qubit", wires=N_WIRES),
        )
        estimate = [np.mean(x_samples), *np.mean(1 - 2 * np.asarray(bits), axis=0)]
        np.testing.assert_allclose(
            estimate, np.asarray(expected, dtype=float), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_generate_samples_in_mixed_bases_match_default_qubit(self):
        set_seed(TEST_SEED)
        bloch_vector = np.array([0.6, 0.0, 0.8])

        @qml.qnode(qml.device("default.qubit", wires=N_WIRES))
        def rotated_circuit():
            self.operations()
            qml.Hadamard(wires=0)
            qml.adjoint(qml.S(wires=1))
            qml.Hadamard(wires=1)
            qml.RY(-np.arctan2(bloch_vector[0], bloch_vector[2]), wires=2)
            return qml.probs()

        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS)
        device.apply(self.operations())
        samples = device.generate_samples(bases={0: "X", 1: "y", 2: bloch_vector})
        assert samples is device.samples
        indices = samples @ (1 << np.arange(N_WIRES)[::-1])
        empirical = np.bincount(indices, minlength=2**N_WIRES) / N_SHOTS
        np.testing.assert_allclose(empirical, np.asarray(rotated_circuit()), atol=ATOL_APPROX_COMPARISON)

    def test_generate_samples_in_computational_basis_uses_default_strategy(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply(self.operations())
        samples = device.generate_samples(bases={0: "Z", 1: [0.0, 0.0, 2.0]})
        assert samples.shape == (10, N_WIRES)
        assert device._samples_bases == {}

    def test_expval_after_generate_samples(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS)
        device.apply(self.operations())
        device.generate_samples(bases={0: "X"})
        expected = self.run(
            lambda: [qml.expval(qml.X(0) @ qml.Z(1)), qml.expval(qml.Z(2))], qml.device("default.qubit", wires=N_WIRES)
        )
        estimate = [float(device.expval(qml.X(0) @ qml.Z(1))), float(device.expval(qml.Z(2)))]
        np.testing.assert_allclose(
            estimate, np.asarray(expected, dtype=float), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize(
        "observable",
        [qml.Y(0), qml.Z(0), qml.X(0) @ qml.X(1), qml.Hermitian(np.diag([1.0, -1.0]), wires=0)],
    )
    def test_observable_not_matching_sampled_bases_raises(self, observable):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply(self.operations())
        device.generate_samples(bases={0: "X"})
        with pytest.raises(DeviceError, match="does not match the bases"):
            device.expval(observable)

    def test_reset_clears_sampled_bases(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply(self.operations())
        device.generate_samples(bases={0: "X"})
        assert list(device._samples_bases) == [0]
        device.reset()
        assert device._samples_bases == {}

    def test_non_product_state_preparation_raises(self):
        state = np.zeros(2**N_WIRES)
        state[[1, 2]] = 1 / np.sqrt(2)
        tape = qml.tape.QuantumScript([qml.StatePrep(state, wires=range(N_WIRES))], [qml.sample(qml.X(0))], shots=10)
        with pytest.raises(DeviceError, match="requires a product state"):
            NonInteractingFermionicDevice(wires=N_WIRES).execute(tape)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(qml.X(0)), qml.sample(qml.Z(0))],
            [qml.expval(qml.X(0)), qml.sample()],
            [qml.expval(qml.Y(1)), qml.expval(qml.Hermitian(np.diag([1.0, 2.0]), wires=1))],
        ],
    )
    def test_measurements_needing_several_bases_on_a_wire_raise(self, measurements):
        tape = qml.tape.QuantumScript(self.operations(), measurements, shots=10)
        with pytest.raises(DeviceError, match="need several bases on the same wires"):
            NonInteractingFermionicDevice(wires=N_WIRES).execute(tape)

    @pytest.mark.parametrize(
        "measurements, expected",
        [
            ([qml.expval(qml.X(0) @ qml.Y(2)), qml.expval(qml.Z(1))], {0: "X", 2: "Y"}),
            ([qml.sample(wires=[0]), qml.expval(qml.Hermitian(np.diag([1.0, 2.0]), wires=1))], {}),
            ([qml.expval(2.0 * qml.Y(1) + qml.Z(0))], {1: "Y"}),
        ],
    )
    def test_measurement_bases(self, measurements, expected):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        assert device._measurement_bases(measurements) == expected

    @pytest.mark.parametrize(
        "bases, match",
        [
            ({5: "X"}, "is not on the device"),
            ({0: "W"}, "Unknown Pauli bases"),
            ({0: [0.0, 0.0, 0.0]}, "non-zero Bloch vector of length 3"),
            ({0: [1.0, 0.0]}, "non-zero Bloch vector of length 3"),
        ],
    )
    def test_invalid_bases_raise(self, bases, match):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply(self.operations())
        with pytest.raises(ValueError, match=match):
            device.generate_samples(bases=bases)

    def test_bases_to_bloch_vectors(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        bloch_vectors = device._bases_to_bloch_vectors({1: "x", 2: [0.0, 3.0, 0.0]})
        np.testing.assert_allclose(
            bloch_vectors, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=ATOL_MATRIX_COMPARISON
        )

    def test_observable_in_sampled_frame(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply(self.operations())
        device.generate_samples(bases={0: "X", 1: "Y"})
        unchanged = qml.Z(2)
        assert device._observable_in_sampled_frame(unchanged) is unchanged
        rewritten = device._observable_in_sampled_frame(2.0 * qml.X(0) @ qml.Y(1) @ qml.Z(2))
        assert rewritten.pauli_rep == (2.0 * qml.Z(0) @ qml.Z(1) @ qml.Z(2)).pauli_rep
        assert isinstance(device._build_sampled_measurement(ExpectationMP, qml.X(0)), ExpectationMP)

    def test_product_basis_sampling_strategy_attribute(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        assert isinstance(device.product_basis_sampling_strategy, ProductBasisSampling)

    def test_default_sampling_strategy_is_product_basis(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        assert isinstance(device.sampling_strategy, ProductBasisSampling)

    def test_computational_basis_samples_use_product_basis_sampling(self, monkeypatch):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=N_SHOTS)
        device.apply(self.operations())
        monkeypatch.setattr(
            device.fallback_sampling_strategy,
            "batch_generate_samples",
            lambda *args, **kwargs: pytest.fail("The fallback strategy must not be used for a product state."),
        )
        samples = device.generate_samples()
        empirical = np.bincount(samples @ (1 << np.arange(N_WIRES)[::-1]), minlength=2**N_WIRES) / N_SHOTS
        np.testing.assert_allclose(empirical, device.analytic_probability().numpy(), atol=ATOL_APPROX_COMPARISON)

    def test_non_product_state_falls_back_to_fallback_strategy(self, monkeypatch):
        state = np.zeros(2**N_WIRES)
        state[[1, 2]] = 1 / np.sqrt(2)
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10)
        device.apply([qml.StatePrep(state, wires=range(N_WIRES))])
        fallback_samples = np.zeros((10, N_WIRES), dtype=np.int64)
        monkeypatch.setattr(
            device.fallback_sampling_strategy, "batch_generate_samples", lambda *args, **kwargs: fallback_samples
        )
        assert device.generate_samples() is fallback_samples

    def test_explicit_sampling_strategy_is_kept(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10, sampling_strategy="QubitByQubitSampling")
        device.apply(self.operations())
        assert not isinstance(device.sampling_strategy, ProductBasisSampling)
        assert device.generate_samples().shape == (10, N_WIRES)

    def test_sampling_strategy_instance_is_used_for_every_basis(self):
        strategy = ProductBasisSampling(shots_chunk_size=7)
        device = NonInteractingFermionicDevice(wires=N_WIRES, shots=10, sampling_strategy=strategy)
        assert device.sampling_strategy is strategy
        assert device.product_basis_sampling_strategy is strategy
        device.apply(self.operations())
        assert device.generate_samples(bases={0: "X"}).shape == (10, N_WIRES)
