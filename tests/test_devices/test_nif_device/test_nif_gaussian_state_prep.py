import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError

import matchcake as mc
from matchcake.operations import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import GaussianStatePrep, ProductState
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_WIRES = 4
N_SHOTS = 20_000


class TestNonInteractingFermionicDeviceGaussianStatePrep:
    @staticmethod
    def gates():
        """Queue a matchgate circuit on ``N_WIRES`` wires."""
        qml.IsingXY(0.5, wires=[0, 1])
        qml.IsingXY(0.9, wires=[1, 2])
        mc.operations.CompRxRx(np.array([0.3, -0.7]), wires=[2, 3])
        mc.operations.CompRyRy(np.array([-1.1, 0.2]), wires=[1, 2])
        qml.IsingXY(-0.4, wires=[0, 1])

    @staticmethod
    def random_covariance(n_wires: int, batch_size=None, seed: int = TEST_SEED) -> np.ndarray:
        """Build a random mixed-state covariance matrix by rotating random polarizations.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :param seed: Seed of the random generator.
        :type seed: int
        :return: Real antisymmetric matrix of shape ``(2 * n_wires, 2 * n_wires)`` or batched.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(seed)
        shape = (n_wires,) if batch_size is None else (batch_size, n_wires)
        polarizations = rng.uniform(-0.9, 0.9, size=shape)
        rotation = SingleParticleTransitionMatrixOperation.random_params(
            batch_size=batch_size, wires=list(range(n_wires)), seed=seed
        ).real
        return np.einsum(
            "...ij,...ik,...kl->...jl",
            rotation,
            np.asarray(CovarianceMatrixToolbox.from_polarizations(polarizations)),
            rotation,
        )

    @staticmethod
    def run(device, covariance, wires, measurement, shots=None):
        """Execute the test circuit from a Gaussian state on ``nif.qubit`` or on a dense mixed simulator.

        The dense simulator receives the decomposition of the state preparation, a density matrix.

        :param device: A device name or a device instance.
        :param covariance: Covariance matrix of the initial state on ``wires``.
        :param wires: Wires of the initial state.
        :param measurement: Callable that returns the measurement process.
        :param shots: Number of shots, or ``None`` for an analytic execution.
        :return: The result of the measurement.
        """
        if isinstance(device, str):
            device = qml.device(device, wires=N_WIRES)

        @qml.set_shots(shots)
        @qml.qnode(device)
        def circuit():
            if device.name == "default.mixed":
                qml.QubitDensityMatrix(CovarianceMatrixToolbox.to_density_matrix(covariance), wires=wires)
            else:
                GaussianStatePrep(covariance, wires=wires)
            TestNonInteractingFermionicDeviceGaussianStatePrep.gates()
            return measurement()

        return circuit()

    def setup_method(self):
        set_seed(TEST_SEED)

    def assert_matches_default_mixed(self, covariance, wires, measurement):
        np.testing.assert_allclose(
            qml.math.toarray(self.run("nif.qubit", covariance, wires, measurement)),
            qml.math.toarray(self.run("default.mixed", covariance, wires, measurement)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable",
        [
            qml.Z(0),
            qml.X(0) @ qml.X(1),
            qml.X(0) @ qml.Y(1),
            qml.Y(1) @ qml.Z(2) @ qml.X(3),
            qml.Z(0) @ qml.Z(3),
            qml.X(0) @ qml.Z(1) @ qml.Z(2) @ qml.Y(3),
            qml.X(1),
            0.3 * qml.Z(0) @ qml.Z(1) - 1.2 * qml.X(2) @ qml.X(3) + 0.5 * qml.Y(0) @ qml.Z(1) @ qml.Y(2),
            qml.Projector([0, 1], wires=[1, 2]),
        ],
    )
    def test_expval(self, observable):
        self.assert_matches_default_mixed(
            self.random_covariance(N_WIRES), range(N_WIRES), lambda: qml.expval(observable)
        )

    @pytest.mark.parametrize("probs_wires", [None, [0], [1, 2], [3, 0], [0, 1, 2, 3]])
    def test_probs(self, probs_wires):
        self.assert_matches_default_mixed(
            self.random_covariance(N_WIRES), range(N_WIRES), lambda: qml.probs(wires=probs_wires)
        )

    def test_simultaneous_measurements(self):
        covariance = self.random_covariance(N_WIRES)

        def measurement():
            return qml.expval(qml.Z(0) @ qml.Z(1)), qml.probs(wires=[2, 3]), qml.purity(wires=range(N_WIRES))

        nif_results = self.run("nif.qubit", covariance, range(N_WIRES), measurement)
        mixed_results = self.run("default.mixed", covariance, range(N_WIRES), measurement)
        for nif_result, mixed_result in zip(nif_results, mixed_results):
            np.testing.assert_allclose(
                qml.math.toarray(nif_result),
                qml.math.toarray(mixed_result),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 1, 3], [2]])
    def test_state_on_part_of_the_wires(self, wires):
        covariance = self.random_covariance(len(wires))
        self.assert_matches_default_mixed(covariance, wires, qml.probs)
        self.assert_matches_default_mixed(covariance, wires, lambda: qml.expval(qml.X(0) @ qml.Z(1) @ qml.Y(2)))

    def test_state_on_reordered_wires_is_rejected(self):
        with pytest.raises(ValueError, match="same order"):
            self.run("nif.qubit", self.random_covariance(2), [1, 0], qml.probs)

    def test_pure_gaussian_state_matches_basis_state(self):
        bits = np.array([1, 0, 0, 1])
        covariance = np.asarray(ProductState.from_basis_state(bits, wires=range(N_WIRES)).covariance_matrix)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES))
        def reference():
            qml.BasisState(bits, wires=range(N_WIRES))
            self.gates()
            return qml.probs()

        np.testing.assert_allclose(
            self.run("nif.qubit", covariance, range(N_WIRES), qml.probs),
            reference(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_maximally_mixed_state_is_uniform(self):
        probabilities = self.run("nif.qubit", np.zeros((2 * N_WIRES, 2 * N_WIRES)), range(N_WIRES), qml.probs)
        np.testing.assert_allclose(
            probabilities, np.full(2**N_WIRES, 2.0**-N_WIRES), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_batched_covariance(self):
        batch_size = 3
        covariances = self.random_covariance(N_WIRES, batch_size)

        def measurement():
            return qml.expval(qml.X(0) @ qml.X(1)), qml.probs(wires=[1, 2])

        expvals, probabilities = self.run("nif.qubit", covariances, range(N_WIRES), measurement)
        assert qml.math.shape(expvals) == (batch_size,)
        assert qml.math.shape(probabilities) == (batch_size, 4)
        for index in range(batch_size):
            expected_expval, expected_probabilities = self.run(
                "default.mixed", covariances[index], range(N_WIRES), measurement
            )
            np.testing.assert_allclose(
                expvals[index], expected_expval, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )
            np.testing.assert_allclose(
                probabilities[index], expected_probabilities, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    @pytest.mark.parametrize("r_dtype, c_dtype", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)])
    def test_device_precision(self, r_dtype, c_dtype):
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES, r_dtype=r_dtype, c_dtype=c_dtype)
        covariance = torch.tensor(self.random_covariance(N_WIRES), dtype=r_dtype)
        result = self.run(device, covariance, range(N_WIRES), lambda: qml.expval(qml.X(0) @ qml.Y(1)))
        assert device.extended_covariance_matrix.dtype == r_dtype
        np.testing.assert_allclose(
            qml.math.toarray(result),
            self.run(
                "default.mixed",
                covariance.numpy().astype(float),
                range(N_WIRES),
                lambda: qml.expval(qml.X(0) @ qml.Y(1)),
            ),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_covariance_matrix_properties(self):
        covariance = self.random_covariance(N_WIRES)
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply([GaussianStatePrep(covariance, wires=range(N_WIRES))])
        np.testing.assert_allclose(
            device.covariance_matrix, covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        extended = qml.math.toarray(device.extended_covariance_matrix)
        np.testing.assert_allclose(
            extended[:-1, :-1], covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_array_equal(extended[-1], 0.0)
        np.testing.assert_array_equal(extended[:, -1], 0.0)

    def test_state_prep_after_a_gate_is_rejected(self):
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        with pytest.raises(DeviceError, match="cannot be used after other Operations"):
            device.apply(
                [
                    mc.operations.CompRxRx(np.array([0.3, -0.7]), wires=[0, 1]),
                    GaussianStatePrep(self.random_covariance(N_WIRES), wires=range(N_WIRES)),
                ]
            )

    @pytest.mark.parametrize(
        "sampling_strategy", ["ProductBasisSampling", "2QubitBy2QubitSampling", "QubitByQubitSampling"]
    )
    def test_samples_follow_probabilities(self, sampling_strategy):
        covariance = self.random_covariance(N_WIRES)
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES, sampling_strategy=sampling_strategy)
        estimated = self.run(device, covariance, range(N_WIRES), qml.probs, shots=N_SHOTS)
        np.testing.assert_allclose(
            estimated,
            self.run("default.mixed", covariance, range(N_WIRES), qml.probs),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("observable", [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Z(2) @ qml.Y(3), qml.Z(0) @ qml.Z(2)])
    def test_shot_estimates_in_rotated_bases(self, observable):
        covariance = self.random_covariance(N_WIRES)
        estimated = self.run("nif.qubit", covariance, range(N_WIRES), lambda: qml.expval(observable), shots=N_SHOTS)
        np.testing.assert_allclose(
            estimated,
            self.run("default.mixed", covariance, range(N_WIRES), lambda: qml.expval(observable)),
            atol=ATOL_APPROX_COMPARISON,
        )
