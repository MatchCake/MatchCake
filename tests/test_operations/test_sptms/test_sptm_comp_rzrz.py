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
            for theta in np.linspace(0, 2 * np.pi, num=10)
            for phi in np.linspace(0, 2 * np.pi, num=10)
        ]
        + [
            (np.full(batch_size, theta), np.full(batch_size, theta))
            for batch_size in [1, 3]
            for theta in np.linspace(0, 2 * np.pi, num=10)
        ],
    )
    def test_matchgate_equal_to_sptm_rzrz(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
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
            for theta in np.linspace(0, 2 * np.pi, num=10)
            for phi in np.linspace(0, 2 * np.pi, num=10)
        ],
    )
    def test_sptm_rzrz_is_so4(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1])
        assert sptm_obj.check_is_in_so4()

    @pytest.mark.parametrize(
        "theta",
        [np.full(batch_size, theta) for batch_size in [1, 4] for theta in np.linspace(0, 2 * np.pi, num=10)],
    )
    def test_sptm_rzrz_is_so4_equal_angles(self, theta):
        params = np.asarray([theta, theta]).reshape(-1, 2).squeeze()
        sptm_obj = SptmCompRzRz(params, wires=[0, 1])
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
            for theta in np.linspace(0, 2 * np.pi, num=10)
            for phi in np.linspace(0, 2 * np.pi, num=10)
        ],
    )
    def test_matchgate_equal_to_sptm_rzrz_adjoint(self, theta, phi):
        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        params = SptmCompRzRz.clip_angles(params)
        matchgate = CompRzRz(params, wires=[0, 1]).adjoint()
        m_sptm = make_single_particle_transition_matrix_from_gate(matchgate.matrix())
        sptm = SptmCompRzRz(params, wires=[0, 1]).adjoint().matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=1,
        )
