import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import fSWAP
from matchcake.operations.single_particle_transition_matrices import SptmCompZX
from matchcake.utils import make_single_particle_transition_matrix_from_gate

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestSptmFSwap:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_matchgate_equal_to_sptm(self):
        matchgate = fSWAP(wires=[0, 1])
        m_sptm = make_single_particle_transition_matrix_from_gate(matchgate.matrix())
        sptm = SptmCompZX(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_adjoint(self):
        matchgate = fSWAP(wires=[0, 1]).adjoint()
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompZX(wires=[0, 1]).adjoint().matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_sptm_fswap_chain_equal_to_sptm_fswap(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))

        def _gen():
            for tmp_wire0 in range(wire0, wire1):
                yield SptmCompZX(wires=[tmp_wire0, tmp_wire0 + 1])
            return

        device = mc.NIFDevice(wires=all_wires)
        device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
        chain_sptm = device.apply_metadata["global_sptm"]
        sptm = SptmCompZX(wires=[wire0, wire1]).matrix(all_wires)

        np.testing.assert_allclose(
            sptm,
            chain_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_fswap_chain_equal_to_sptm_fswap(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))

        def _gen():
            for tmp_wire0 in range(wire0, wire1):
                yield fSWAP(wires=[tmp_wire0, tmp_wire0 + 1])
            return

        device = mc.NIFDevice(wires=all_wires)
        device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
        chain_sptm = device.apply_metadata["global_sptm"]
        sptm = SptmCompZX(wires=[wire0, wire1]).matrix(all_wires)

        np.testing.assert_allclose(
            sptm,
            chain_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_sptm_fswap_chain_equal_to_sptm_fswap_reverse(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))

        def _gen():
            for tmp_wire0 in reversed(range(wire0, wire1)):
                yield SptmCompZX(wires=[tmp_wire0, tmp_wire0 + 1])
            return

        device = mc.NIFDevice(wires=all_wires)
        device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
        chain_sptm = device.apply_metadata["global_sptm"]
        sptm = SptmCompZX(wires=[wire1, wire0]).matrix(all_wires)

        np.testing.assert_allclose(
            sptm,
            chain_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_fswap_chain_equal_to_sptm_fswap_reverse(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))

        def _gen():
            for tmp_wire0 in reversed(range(wire0, wire1)):
                yield fSWAP(wires=[tmp_wire0, tmp_wire0 + 1])
            return

        device = mc.NIFDevice(wires=all_wires)
        device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
        chain_sptm = device.apply_metadata["global_sptm"]
        sptm = SptmCompZX(wires=[wire1, wire0]).matrix(all_wires)

        np.testing.assert_allclose(
            sptm,
            chain_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_sptm_fswap_in_so4(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))
        sptm = SptmCompZX(wires=[wire0, wire1]).pad(all_wires)
        assert sptm.check_is_in_so4()
        sptm_matrix = sptm.matrix()
        sptm_dagger = np.einsum("...ij->...ji", sptm_matrix).conj()
        expected_eye = np.einsum("...ij,...jk->...ik", sptm_matrix, sptm_dagger)
        np.testing.assert_allclose(
            expected_eye,
            np.eye(2 * len(all_wires)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire0, wire1, all_wires",
        [
            (wire0, wire1, n_wires)
            for n_wires in range(2, 16)
            for wire0 in range(n_wires - 1)
            for wire1 in range(wire0 + 1, n_wires)
        ],
    )
    def test_sptm_fswap_unitary(self, wire0, wire1, all_wires):
        all_wires = list(range(all_wires))
        sptm = SptmCompZX(wires=[wire0, wire1]).pad(all_wires)
        assert sptm.check_is_unitary(atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
