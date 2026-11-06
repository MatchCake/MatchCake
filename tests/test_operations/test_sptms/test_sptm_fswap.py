import numpy as np
import pennylane as qml
import pytest

import matchcake as mc
from matchcake.operations import fSWAP
from matchcake.operations.fermionic_swap import CompZX
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompZX,
)
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

    def test_to_matchgate(self):
        sptm = SptmCompZX(wires=[0, 1])
        mg = sptm.to_matchgate()
        assert isinstance(mg, CompZX)

    def test_matchgate_equal_to_sptm(self):
        matchgate = fSWAP(wires=[0, 1])
        m_sptm = make_single_particle_transition_matrix_from_gate(matchgate.matrix())
        sptm = SptmCompZX(wires=[0, 1]).sptm_matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], [0, 2], [2, 0], [1, 4], [0, 5], [3, 0]])
    def test_adjoint_is_inverse(self, wires):
        all_wires = list(range(max(wires) + 1))
        sptm = SptmCompZX(wires=wires)
        product = sptm.adjoint().sptm_matrix(all_wires) @ sptm.sptm_matrix(all_wires)
        np.testing.assert_allclose(
            product,
            np.eye(2 * len(all_wires)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0]])
    def test_adjoint_on_adjacent_wires_is_self(self, wires):
        sptm = SptmCompZX(wires=wires)
        assert sptm.adjoint() is sptm

    @pytest.mark.parametrize("device_name", ["nif.qubit", "default.qubit"])
    @pytest.mark.parametrize("wires", [[0, 2], [2, 0], [1, 4], [0, 5], [3, 0]])
    def test_qml_adjoint_matches_explicit_inverse(self, wires, device_name):
        n_wires = max(wires) + 1
        params = np.array([0.3, 0.8])
        inverse_matrix = SptmCompZX(wires=wires).sptm_matrix().T
        initial_state = np.arange(n_wires) % 2 == 0

        def circuit(use_adjoint):
            qml.BasisState(initial_state.astype(int), wires=range(n_wires))
            SptmCompRxRx(params, wires=[0, 1])
            if use_adjoint:
                qml.adjoint(SptmCompZX(wires=wires))
            else:
                SingleParticleTransitionMatrixOperation(inverse_matrix, wires=wires)
            SptmCompRxRx(params[::-1], wires=[n_wires - 2, n_wires - 1])
            return qml.probs(wires=range(n_wires))

        qnode = qml.QNode(circuit, qml.device(device_name, wires=n_wires))
        np.testing.assert_allclose(
            qml.math.to_numpy(qnode(True)),
            qml.math.to_numpy(qnode(False)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_adjoint(self):
        matchgate = fSWAP(wires=[0, 1]).adjoint()
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompZX(wires=[0, 1]).adjoint().sptm_matrix()
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
        sptm = SptmCompZX(wires=[wire0, wire1]).sptm_matrix(all_wires)

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
        sptm = SptmCompZX(wires=[wire0, wire1]).sptm_matrix(all_wires)

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
        sptm = SptmCompZX(wires=[wire1, wire0]).sptm_matrix(all_wires)

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
        sptm = SptmCompZX(wires=[wire1, wire0]).sptm_matrix(all_wires)

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
        sptm_matrix = sptm.sptm_matrix()
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
