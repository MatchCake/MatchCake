import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError
from pennylane.wires import Wires

import matchcake as mc
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.state_preparation import (
    GaussianStatePrep,
    PlusState,
    ProductState,
)
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)

N_WIRES = 5
BASIS_STATE = np.array([1, 0, 1, 1, 0])


class TestNonInteractingFermionicDeviceReducedState:
    @staticmethod
    def brick_wall(wires, depth: int, seed: int):
        """Build a random brick-wall matchgate circuit.

        :param wires: Wires of the circuit, in register order.
        :param depth: Number of layers.
        :type depth: int
        :param seed: Seed of the random angles.
        :type seed: int
        :return: The list of operations.
        """
        rng = np.random.default_rng(seed)
        wires = list(wires)
        operations = []
        for _ in range(depth):
            for position in range(len(wires) - 1):
                pair = [wires[position], wires[position + 1]]
                operations.append(CompRxRx(rng.normal(size=2), wires=pair))
                operations.append(CompRyRy(rng.normal(size=2), wires=pair))
                operations.append(CompRzRz(rng.normal(size=2), wires=pair))
            operations.append(fSWAP(wires=[wires[0], wires[1]]))
        return operations

    @staticmethod
    def run(device_name: str, measurement, operations=None, state_prep=None, shots=None):
        """Execute a circuit on ``N_WIRES`` wires and return the result of a measurement.

        :param device_name: Name of the PennyLane device.
        :type device_name: str
        :param measurement: Callable that returns the measurement process(es).
        :param operations: Operations of the circuit. Defaults to the reference brick-wall circuit.
        :param state_prep: Callable that queues the state preparation. Defaults to ``BASIS_STATE``.
        :param shots: Number of shots, or ``None`` for an analytic execution.
        :return: The result of the measurement.
        """
        if operations is None:
            operations = TestNonInteractingFermionicDeviceReducedState.brick_wall(range(N_WIRES), 2, TEST_SEED)

        @qml.set_shots(shots)
        @qml.qnode(qml.device(device_name, wires=N_WIRES))
        def circuit():
            if state_prep is None:
                qml.BasisState(BASIS_STATE, wires=range(N_WIRES))
            else:
                state_prep()
            for operation in operations:
                qml.apply(operation)
            return measurement()

        return circuit()

    @pytest.fixture
    def prepared_device(self) -> mc.NonInteractingFermionicDevice:
        """A device holding the final state of the reference circuit."""
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply(
            [qml.BasisState(BASIS_STATE, wires=range(N_WIRES))] + self.brick_wall(range(N_WIRES), 2, TEST_SEED)
        )
        return device

    def assert_matches_default_qubit(self, measurement, **kwargs):
        np.testing.assert_allclose(
            qml.math.toarray(self.run("nif.qubit", measurement, **kwargs)),
            qml.math.toarray(self.run("default.qubit", measurement, **kwargs)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0], [4], [0, 1], [1, 2, 3], [3, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    @pytest.mark.parametrize(
        "measurement",
        [mc.covariance_matrix, qml.vn_entropy, qml.purity, qml.density_matrix],
        ids=["covariance_matrix", "vn_entropy", "purity", "density_matrix"],
    )
    def test_measurements_on_contiguous_wires(self, measurement, wires):
        self.assert_matches_default_qubit(lambda: measurement(wires=wires))

    def test_vn_entropy_log_base(self):
        self.assert_matches_default_qubit(lambda: qml.vn_entropy(wires=[0, 1], log_base=2))

    def test_entropies_of_complementary_cuts_are_equal(self):
        left, right = self.run("nif.qubit", lambda: (qml.vn_entropy(wires=[0, 1]), qml.vn_entropy(wires=[2, 3, 4])))
        np.testing.assert_allclose(left, right, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_several_reduced_measurements_at_once(self):
        def measurement():
            return (
                mc.covariance_matrix(wires=[1, 2]),
                qml.vn_entropy(wires=[1, 2]),
                qml.purity(wires=[0, 1, 2]),
                qml.expval(qml.Z(0) @ qml.Z(1)),
            )

        nif_results = self.run("nif.qubit", measurement)
        dense_results = self.run("default.qubit", measurement)
        for nif_result, dense_result in zip(nif_results, dense_results):
            np.testing.assert_allclose(
                qml.math.toarray(nif_result),
                qml.math.toarray(dense_result),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix, mc.covariance_matrix])
    @pytest.mark.parametrize("wires", [[0, 2], [1, 0], [0, 1, 3]])
    def test_non_contiguous_wires_are_rejected(self, measurement, wires):
        with pytest.raises(ValueError, match="fermionic=True"):
            self.run("nif.qubit", lambda: measurement(wires=wires))

    @pytest.mark.parametrize("wires", [[0, 2], [1, 3], [0, 2, 4], [2, 0]])
    def test_fermionic_covariance_matrix_matches_dense(self, wires):
        self.assert_matches_default_qubit(lambda: mc.covariance_matrix(wires=wires, fermionic=True))

    def test_fermionic_reduced_state_differs_from_the_qubit_one(self):
        fermionic_covariance = self.run("nif.qubit", lambda: mc.covariance_matrix(wires=[0, 2], fermionic=True))
        qubit_density = self.run("default.qubit", lambda: qml.density_matrix(wires=[0, 2]))
        difference = np.abs(CovarianceMatrixToolbox.to_density_matrix(fermionic_covariance) - qubit_density).max()
        assert difference > ATOL_MATRIX_COMPARISON

    @pytest.mark.parametrize("measurement", [mc.covariance_matrix, qml.vn_entropy, qml.purity, qml.density_matrix])
    def test_reduced_measurements_are_rejected_with_shots(self, measurement):
        with pytest.raises(DeviceError, match="analytic"):
            self.run("nif.qubit", lambda: measurement(wires=[0, 1]), operations=[], shots=10)

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix])
    def test_superposed_product_state_is_rejected(self, measurement):
        with pytest.raises(NotImplementedError, match="without displacement"):
            self.run(
                "nif.qubit",
                lambda: measurement(wires=[0, 1]),
                state_prep=lambda: PlusState(wires=range(N_WIRES)),
            )

    def test_non_gaussian_state_preparation_is_rejected(self, prepared_device):
        state = np.zeros(2**N_WIRES)
        state[0] = 1.0
        prepared_device.apply_state_prep(qml.StatePrep(state, wires=range(N_WIRES)), index=0)
        with pytest.raises(ValueError, match="Covariance matrix can only be computed"):
            prepared_device.reduced_covariance_matrix([0, 1])

    def test_reduced_state_of_a_gaussian_state_prep(self):
        covariance = qml.math.toarray(self.run("nif.qubit", lambda: mc.covariance_matrix(wires=[0, 1, 2, 3])))
        follow_up = self.brick_wall(range(4), 1, TEST_SEED + 1)

        def measurement():
            return mc.covariance_matrix(wires=[1, 2]), qml.vn_entropy(wires=[0, 1])

        def run(device_name):
            @qml.qnode(qml.device(device_name, wires=4))
            def circuit():
                if device_name == "nif.qubit":
                    GaussianStatePrep(covariance, wires=range(4))
                else:
                    qml.QubitDensityMatrix(CovarianceMatrixToolbox.to_density_matrix(covariance), wires=range(4))
                for operation in follow_up:
                    qml.apply(operation)
                return measurement()

            return circuit()

        for nif_result, dense_result in zip(run("nif.qubit"), run("default.mixed")):
            np.testing.assert_allclose(
                nif_result, dense_result, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    @pytest.mark.parametrize("kept_wires", [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4]])
    @pytest.mark.parametrize(
        "observable_builder",
        [
            lambda wires: qml.Z(wires[0]),
            lambda wires: qml.X(wires[0]) @ qml.X(wires[1]),
            lambda wires: qml.Y(wires[0]) @ qml.X(wires[-1]),
            lambda wires: qml.Z(wires[0]) @ qml.Z(wires[-1]),
        ],
    )
    def test_follow_up_circuit_on_the_reduced_state(self, kept_wires, observable_builder):
        r"""
        Reduce the state to the kept wires, run a second circuit on a smaller device from the reduced
        covariance matrix, and compare with the same second circuit applied to the kept wires of the
        full system.
        """
        n_kept = len(kept_wires)
        reduced_covariance = self.run("nif.qubit", lambda: mc.covariance_matrix(wires=kept_wires))
        follow_up_seed = TEST_SEED + 2
        follow_up_operations = self.brick_wall(range(n_kept), 1, follow_up_seed)

        @qml.qnode(qml.device("nif.qubit", wires=n_kept))
        def follow_up():
            GaussianStatePrep(reduced_covariance, wires=range(n_kept))
            for operation in follow_up_operations:
                qml.apply(operation)
            return qml.expval(observable_builder(list(range(n_kept)))), qml.probs()

        full_operations = self.brick_wall(range(N_WIRES), 2, TEST_SEED) + self.brick_wall(kept_wires, 1, follow_up_seed)
        expected_expval, expected_probabilities = self.run(
            "default.qubit",
            lambda: (qml.expval(observable_builder(kept_wires)), qml.probs(wires=kept_wires)),
            operations=full_operations,
        )
        expval, probabilities = follow_up()
        np.testing.assert_allclose(expval, expected_expval, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            probabilities, expected_probabilities, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_reduced_covariance_matrix_is_the_principal_submatrix(self, prepared_device):
        full_covariance = qml.math.toarray(prepared_device.covariance_matrix)
        indices = [2, 3, 4, 5]
        np.testing.assert_allclose(
            prepared_device.reduced_covariance_matrix([1, 2]),
            full_covariance[np.ix_(indices, indices)],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires, message", [([0, 7], "not wires of the device"), ([], "At least one wire")])
    def test_reduced_covariance_matrix_rejects_bad_wires(self, prepared_device, wires, message):
        with pytest.raises(ValueError, match=message):
            prepared_device.reduced_covariance_matrix(wires)

    def test_reduced_state_returns_a_state_preparation(self, prepared_device):
        reduced_state = prepared_device.reduced_state([1, 2, 3])
        assert isinstance(reduced_state, GaussianStatePrep)
        assert reduced_state.wires == Wires([0, 1, 2])
        np.testing.assert_allclose(
            reduced_state.covariance_matrix,
            prepared_device.reduced_covariance_matrix([1, 2, 3]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.to_density_matrix(reduced_state.covariance_matrix),
            self.run("default.qubit", lambda: qml.density_matrix(wires=[1, 2, 3])),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_reduced_state_options(self, prepared_device):
        reduced_state = prepared_device.reduced_state([0, 2], fermionic=True, reduced_wires=["a", "b"])
        assert reduced_state.wires == Wires(["a", "b"])
        with pytest.raises(ValueError, match="fermionic=True"):
            prepared_device.reduced_state([0, 2])

    def test_partial_trace_is_the_complement_of_reduced_state(self, prepared_device):
        np.testing.assert_allclose(
            prepared_device.partial_trace([3, 4]).covariance_matrix,
            prepared_device.reduced_state([0, 1, 2]).covariance_matrix,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        assert prepared_device.partial_trace([0], reduced_wires=[5, 6, 7, 8]).wires == Wires([5, 6, 7, 8])
        fermionic_state = prepared_device.partial_trace([1], fermionic=True)
        assert fermionic_state.wires == Wires([0, 1, 2, 3])

    def test_partial_trace_rejects_bad_wires(self, prepared_device):
        with pytest.raises(ValueError, match="not wires of the device"):
            prepared_device.partial_trace([9])
        with pytest.raises(ValueError, match="fermionic=True"):
            prepared_device.partial_trace([1])
        with pytest.raises(ValueError, match="At least one wire"):
            prepared_device.partial_trace(range(N_WIRES))

    def test_reduced_state_of_a_fresh_device_is_the_zero_state(self):
        device = mc.NonInteractingFermionicDevice(wires=3)
        expected = ProductState.from_basis_state(np.zeros(2, dtype=int), wires=[0, 1]).covariance_matrix
        np.testing.assert_allclose(
            device.reduced_state([0, 1]).covariance_matrix, np.asarray(expected), atol=ATOL_MATRIX_COMPARISON
        )

    def test_number_of_mixed_modes_is_bounded_by_the_traced_wires(self, prepared_device):
        polarizations = mc.utils.CovarianceMatrixToolbox.mode_polarizations(
            prepared_device.reduced_covariance_matrix([0, 1, 2, 3])
        )
        n_mixed_modes = int(np.sum(np.asarray(polarizations) < 1.0 - ATOL_SCALAR_COMPARISON))
        assert n_mixed_modes <= 1

    def test_batched_circuit(self):
        batch_size = 3
        rng = np.random.default_rng(TEST_SEED)
        angles = torch.tensor(rng.normal(size=(batch_size, 2)))

        def operations(batch_angles):
            return [
                CompRxRx(batch_angles, wires=[0, 1]),
                CompRyRy(0.5 * batch_angles, wires=[1, 2]),
                CompRxRx(1.5 * batch_angles, wires=[2, 3]),
                CompRyRy(batch_angles, wires=[3, 4]),
                CompRxRx(batch_angles, wires=[1, 2]),
            ]

        def measurement():
            return (
                mc.covariance_matrix(wires=[1, 2]),
                qml.vn_entropy(wires=[1, 2]),
                qml.purity(wires=[1, 2]),
                qml.density_matrix(wires=[1, 2]),
            )

        batched_results = self.run("nif.qubit", measurement, operations=operations(angles))
        expected_shapes = [(batch_size, 4, 4), (batch_size,), (batch_size,), (batch_size, 4, 4)]
        for result, expected_shape in zip(batched_results, expected_shapes):
            assert tuple(qml.math.shape(result)) == expected_shape
        for index in range(batch_size):
            dense_results = self.run("default.qubit", measurement, operations=operations(angles[index]))
            for result, dense_result in zip(batched_results, dense_results):
                np.testing.assert_allclose(
                    qml.math.toarray(result[index]),
                    qml.math.toarray(dense_result),
                    atol=ATOL_MATRIX_COMPARISON,
                    rtol=RTOL_MATRIX_COMPARISON,
                )

    def test_batched_basis_states(self):
        bits = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
        entropies = self.run(
            "nif.qubit",
            lambda: qml.vn_entropy(wires=[0, 1]),
            state_prep=lambda: ProductState.from_basis_state(bits, wires=range(N_WIRES)),
        )
        for index in range(len(bits)):
            expected = self.run(
                "default.qubit",
                lambda: qml.vn_entropy(wires=[0, 1]),
                state_prep=lambda: qml.BasisState(bits[index], wires=range(N_WIRES)),
            )
            np.testing.assert_allclose(
                entropies[index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    @pytest.mark.parametrize("r_dtype, c_dtype", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)])
    def test_density_matrix_uses_the_device_complex_dtype(self, r_dtype, c_dtype):
        device = mc.NonInteractingFermionicDevice(wires=3, r_dtype=r_dtype, c_dtype=c_dtype)

        @qml.qnode(device, interface="torch")
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=range(3))
            CompRxRx(torch.tensor([0.3, 0.7], dtype=r_dtype), wires=[1, 2])
            return qml.density_matrix(wires=[0, 1])

        assert circuit().dtype == c_dtype
