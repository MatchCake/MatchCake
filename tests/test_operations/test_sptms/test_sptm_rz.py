import numpy as np
import pennylane as qml
import pytest

from matchcake.operations.single_particle_transition_matrices import SptmRz
from matchcake.utils import make_single_particle_transition_matrix_from_gate

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestSptmRz:
    """Single-qubit ``R_Z`` single-particle transition matrix (``SptmRz``)."""

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, num=15))
    def test_sptm_matches_block_formula(self, theta):
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        expected = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        sptm = np.asarray(SptmRz(theta, wires=[0]).matrix())
        np.testing.assert_allclose(sptm, expected, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("wire, theta", [(wire, theta) for wire in [0, 1] for theta in [0.3, 1.7, -2.4]])
    def test_sptm_matches_rz_gate_via_full_transition_matrix(self, wire, theta):
        # Ground truth: build R_Z(theta) (x) I as a two-qubit matchgate and take its full 4x4 SPTM.
        gate = np.kron(qml.RZ(theta, wires=0).matrix(), np.eye(2))
        if wire == 1:
            gate = np.kron(np.eye(2), qml.RZ(theta, wires=0).matrix())
        full_sptm = make_single_particle_transition_matrix_from_gate(gate).real
        block = full_sptm[2 * wire : 2 * wire + 2, 2 * wire : 2 * wire + 2]
        sptm = np.asarray(SptmRz(theta, wires=[wire]).matrix())
        np.testing.assert_allclose(sptm, block, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batched_params(self, batch_size):
        thetas = np.linspace(0.1, 2.0, batch_size)
        sptm = np.asarray(SptmRz(thetas, wires=[0]).matrix())
        assert sptm.shape == (batch_size, 2, 2)
        for index, theta in enumerate(thetas):
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            expected = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
            np.testing.assert_allclose(sptm[index], expected, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("theta", [0.0, 0.6, 2.3, -1.1])
    def test_sptm_is_in_so2(self, theta):
        sptm = np.asarray(SptmRz(theta, wires=[0]).matrix())
        np.testing.assert_allclose(np.linalg.det(sptm), 1.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(sptm @ sptm.T, np.eye(2), atol=ATOL_MATRIX_COMPARISON)

    def test_adjoint_is_negative_angle(self):
        theta = 0.83
        adjoint = np.asarray(SptmRz(theta, wires=[0]).adjoint().matrix())
        negated = np.asarray(SptmRz(-theta, wires=[0]).matrix())
        np.testing.assert_allclose(adjoint, negated, atol=ATOL_MATRIX_COMPARISON)

    def test_invalid_param_shape_raises(self):
        with pytest.raises(ValueError):
            SptmRz(np.zeros((2, 2)), wires=[0])

    def test_random_params_shape(self):
        assert np.shape(SptmRz.random_params()) == (1,)
        assert np.shape(SptmRz.random_params(batch_size=3)) == (3, 1)
