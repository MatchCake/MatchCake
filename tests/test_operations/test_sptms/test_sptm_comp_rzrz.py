import numpy as np
import pytest

from matchcake.operations import CompRzRz
from matchcake.operations.single_particle_transition_matrices import (
    SptmCompRzRz,
)
from matchcake.utils import make_single_particle_transition_matrix_from_gate

from ...configs import (
    ATOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestSptmRzRz:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "theta, phi",
        [
            (np.full(batch_size, theta), np.full(batch_size, phi))
            for batch_size in [1, 3]
            for theta in SptmCompRzRz.ALLOWED_ANGLES
            for phi in SptmCompRzRz.ALLOWED_ANGLES
        ]
        + [
            (np.full(batch_size, theta), np.full(batch_size, theta))
            for batch_size in [1, 3]
            for theta in SptmCompRzRz.EQUAL_ALLOWED_ANGLES
        ],
    )
    def test_matchgate_equal_to_sptm_rzrz(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        params = SptmCompRzRz.clip_angles(params)
        matchgate = CompRzRz(params, wires=[0, 1])
        m_sptm = make_single_particle_transition_matrix_from_gate(matchgate.matrix())
        sptm = SptmCompRzRz(params, wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=1,
        )

    @pytest.mark.parametrize(
        "theta, phi",
        [
            (np.full(batch_size, theta), np.full(batch_size, phi))
            for batch_size in [1, 3]
            for theta in SptmCompRzRz.ALLOWED_ANGLES
            for phi in SptmCompRzRz.ALLOWED_ANGLES
        ],
    )
    def test_sptm_rzrz_is_so4(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1])
        assert sptm_obj.check_is_in_so4()

    @pytest.mark.parametrize(
        "theta",
        [np.full(batch_size, theta) for batch_size in [1, 4] for theta in SptmCompRzRz.EQUAL_ALLOWED_ANGLES],
    )
    def test_sptm_rzrz_is_so4_equal_angles(self, theta):
        params = np.asarray([theta, theta]).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1])
        assert sptm_obj.check_is_in_so4()

    @pytest.mark.parametrize(
        "theta, phi",
        [
            (
                np.linspace(-4 * np.pi, 4 * np.pi, batch_size).squeeze(),
                np.linspace(-4 * np.pi, 4 * np.pi, batch_size).squeeze(),
            )
            for batch_size in [1, 3]
        ],
    )
    def test_sptm_rzrz_is_so4_out_of_angles(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1], check_angles=False, clip_angles=True)
        assert sptm_obj.check_is_in_so4()

    @pytest.mark.parametrize(
        "params",
        [
            [
                [np.pi, np.pi],
                [np.pi, 0],
                [0, np.pi],
                [0, 0],
            ],
            [np.pi, np.pi],
            [np.pi, 0],
            [0, np.pi],
            [0, 0],
        ],
    )
    def test_sptm_rzrz_is_so4_with_params(self, params):
        params = np.asarray(params).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1], check_angles=False, clip_angles=True)
        assert sptm_obj.check_is_in_so4()

    @pytest.mark.parametrize(
        "batch_size",
        [1, 3, (3, 2)],
    )
    def test_sptm_rzrz_clip_angles(self, batch_size):
        angles = np.random.uniform(-10 * np.pi, 10 * np.pi, size=batch_size)
        new_angles = SptmCompRzRz.clip_angles(angles)
        assert new_angles.shape == angles.shape
        assert type(new_angles) == type(angles)
        assert SptmCompRzRz.check_angles(new_angles)

    @pytest.mark.parametrize(
        "theta, phi",
        [
            (np.full(batch_size, theta), np.full(batch_size, phi))
            for batch_size in [1, 3]
            for theta in SptmCompRzRz.ALLOWED_ANGLES
            for phi in SptmCompRzRz.ALLOWED_ANGLES
        ],
    )
    def test_matchgate_equal_to_sptm_rzrz_adjoint(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        params = SptmCompRzRz.clip_angles(params)
        matchgate = CompRzRz(params, wires=[0, 1]).adjoint()
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompRzRz(params, wires=[0, 1]).adjoint().matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=1,
        )
