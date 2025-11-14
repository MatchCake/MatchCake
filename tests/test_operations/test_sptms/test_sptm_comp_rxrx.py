import numpy as np
import pytest
import torch

from matchcake.operations import CompRxRx
from matchcake.operations.single_particle_transition_matrices import SptmCompRxRx
from matchcake.utils import (
    make_single_particle_transition_matrix_from_gate,
    torch_utils,
)
from matchcake.utils.math import circuit_matmul

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)


@pytest.mark.parametrize("batch_size", [1, 4])
class TestCompRxRx:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_matchgate_equal_to_sptm(self, batch_size):
        theta = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        phi = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()

        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        params = SptmCompRxRx.clip_angles(params)
        matchgate = CompRxRx(params, wires=[0, 1])
        m_sptm = make_single_particle_transition_matrix_from_gate(matchgate.matrix())
        sptm = SptmCompRxRx(params, wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_multiple(self, batch_size):
        theta = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        phi = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()

        params = np.asarray([theta, phi]).reshape(-1, 2)
        params = SptmCompRxRx.clip_angles(params)

        matchgates = [CompRxRx(p, wires=[0, 1]) for p in params]
        matchgate = matchgates[0]
        for mg in matchgates[1:]:
            matchgate = circuit_matmul(matchgate, mg)

        m_sptm = matchgate.single_particle_transition_matrix
        sptms = [SptmCompRxRx(p, wires=[0, 1]).matrix() for p in params]
        sptm = sptms[0]
        for s in sptms[1:]:
            sptm = circuit_matmul(sptm, s, operator="einsum")
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_adjoint(self, batch_size):
        theta = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        phi = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()

        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        matchgate = CompRxRx(params, wires=[0, 1]).adjoint()
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompRxRx(params, wires=[0, 1]).adjoint().matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_sptm_unitary(self, batch_size):
        theta = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        phi = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()

        params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
        sptm = SptmCompRxRx(params, wires=[0, 1])
        assert sptm.check_is_unitary()

    def test_sptm_sum_gradient_check(self, batch_size):
        theta = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        phi = np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()

        def sptm_sum(p):
            return torch.sum(SptmCompRxRx(p, wires=[0, 1]).matrix())

        torch.autograd.gradcheck(
            sptm_sum,
            torch_utils.to_tensor(np.c_[theta, phi], torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )
