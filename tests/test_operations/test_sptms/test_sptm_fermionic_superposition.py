from copy import deepcopy

import numpy as np
import pennylane as qml
import pytest

import matchcake as mc
from matchcake.operations import (
    CompHH,
    FermionicSuperposition,
    SptmCompHH,
    SptmCompZX,
    fSWAP,
)
from matchcake.operations.single_particle_transition_matrices.sptm_fermionic_superposition import (
    SptmFermionicSuperposition,
)
from matchcake.utils import make_single_particle_transition_matrix_from_gate
from matchcake.utils.torch_utils import to_numpy

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestSptmFermionicSuperposition:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_sptm_fswap_matmul_comp_hh(self):
        wires = [0, 1]
        initial_state = np.zeros(len(wires))

        nif_device = mc.NIFDevice(wires=wires, contraction_strategy=None)
        nif_device_none = mc.NIFDevice(
            wires=wires,
            contraction_strategy=None,
        )

        def circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield fSWAP(wires=wires)
            yield CompHH(wires=wires)
            return

        def sptm_circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield SptmCompZX(wires=wires)
            yield SptmCompHH(wires=wires)
            return

        nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

        nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"]).round(3)

        np.testing.assert_allclose(
            sptm_global_sptm,
            global_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_sptm_fswap_matmul_comp_hh_probs(self):
        wires = [0, 1]
        initial_state = np.zeros(len(wires))
        nif_device = mc.NIFDevice(wires=wires, contraction_strategy="neighbours")

        def circuit():
            qml.BasisState(initial_state, wires=wires)
            fSWAP(wires=wires)
            CompHH(wires=wires)
            return qml.probs(wires=wires)

        def sptm_circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield SptmCompZX(wires=wires)
            yield SptmCompHH(wires=wires)
            return

        qubit_device = qml.device("default.qubit", wires=len(wires), shots=None)
        qubit_qnode = qml.QNode(circuit, qubit_device)
        qubit_probs = to_numpy(qubit_qnode()).squeeze()

        nif_probs = nif_device.execute_generator(sptm_circuit_gen(), output_type="probs", apply=True, reset=True)
        nif_probs = to_numpy(nif_probs).squeeze()

        np.testing.assert_allclose(
            nif_probs,
            qubit_probs,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wires, is_odd",
        [(np.arange(n), is_odd) for n in range(2, 12) for is_odd in [False, True]],
    )
    def test_sptm_fermionic_superposition_decomposition(self, wires, is_odd):
        initial_state = np.zeros(len(wires))
        initial_state[0] = int(is_odd)

        nif_device = mc.NIFDevice(
            wires=wires,
            contraction_strategy=None,
        )
        nif_device_none = mc.NIFDevice(
            wires=wires,
            contraction_strategy=None,
        )

        def circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield from FermionicSuperposition.compute_decomposition(wires=wires)
            return

        def sptm_circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield from SptmFermionicSuperposition.compute_decomposition(wires=wires)
            return

        nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

        nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"]).round(3)

        np.testing.assert_allclose(
            sptm_global_sptm,
            global_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wires, is_odd",
        [(np.arange(n), is_odd) for n in range(2, 12) for is_odd in [False, True]],
    )
    def test_sptm_fermionic_superposition(self, wires, is_odd):
        initial_state = np.zeros(len(wires))
        initial_state[0] = int(is_odd)

        nif_device = mc.NIFDevice(
            wires=wires,
            contraction_strategy=None,
        )
        nif_device_none = mc.NIFDevice(
            wires=wires,
            contraction_strategy=None,
        )

        def circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield from FermionicSuperposition.compute_decomposition(wires=wires)
            return

        def sptm_circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield SptmFermionicSuperposition(wires=wires)
            return

        nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

        nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
        sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"])

        np.testing.assert_allclose(
            sptm_global_sptm,
            global_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wires, is_odd",
        [(np.arange(n), is_odd) for n in range(2, 12) for is_odd in [False, True]],
    )
    def test_sptm_fermionic_superposition_probs(self, wires, is_odd):
        initial_state = np.zeros(len(wires))
        initial_state[0] = int(is_odd)

        nif_device = mc.NIFDevice(wires=wires, contraction_strategy="neighbours")

        def circuit():
            qml.BasisState(initial_state, wires=wires)
            FermionicSuperposition(wires=wires)
            return qml.probs(wires=wires)

        def sptm_circuit_gen():
            yield qml.BasisState(initial_state, wires=wires)
            yield SptmFermionicSuperposition(wires=wires)
            return

        qubit_device = qml.device("default.qubit", wires=len(wires), shots=None)
        qubit_qnode = qml.QNode(circuit, qubit_device)
        qubit_probs = to_numpy(qubit_qnode()).squeeze()

        nif_probs = nif_device.execute_generator(sptm_circuit_gen(), output_type="probs", apply=True, reset=True)
        nif_probs = to_numpy(nif_probs).squeeze()

        np.testing.assert_allclose(
            nif_probs,
            qubit_probs,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [np.arange(n) for n in range(2, 12)])
    def test_sptm_fermionic_superposition_unitary(self, wires):
        sptm = SptmFermionicSuperposition(wires=wires)
        assert sptm.check_is_unitary(atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("wires", [np.arange(n) for n in range(2, 6)])
    def test_adjoint_is_transpose(self, wires):
        sptm = SptmFermionicSuperposition(wires=wires)
        np.testing.assert_allclose(
            to_numpy(sptm.adjoint().matrix()),
            to_numpy(sptm.matrix()).T,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [np.arange(n) for n in range(2, 6)])
    def test_adjoint_inverts_matrix(self, wires):
        sptm = SptmFermionicSuperposition(wires=wires)
        product = to_numpy(sptm.matrix()) @ to_numpy(sptm.adjoint().matrix())
        np.testing.assert_allclose(
            product,
            np.eye(product.shape[-1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_adjoint_matches_sptm_of_unitary_dagger(self):
        unitary = to_numpy(qml.matrix(FermionicSuperposition(wires=[0, 1])))
        expected = to_numpy(make_single_particle_transition_matrix_from_gate(unitary.conj().T))
        sptm = SptmFermionicSuperposition(wires=[0, 1])
        np.testing.assert_allclose(
            to_numpy(sptm.adjoint().matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("theta", [0.4, 1.1])
    def test_apply_then_adjoint_is_identity_in_circuit(self, theta):
        wires = [0, 1]
        forward = SptmFermionicSuperposition(wires=wires)
        backward = SptmFermionicSuperposition(wires=wires).adjoint()

        @qml.qnode(qml.device("nif.qubit", wires=len(wires)))
        def circuit(apply_pair: bool):
            qml.BasisState(np.array([1, 0]), wires=wires)
            qml.IsingXY(theta, wires=wires)
            if apply_pair:
                qml.apply(forward)
                qml.apply(backward)
            return qml.probs(wires=wires)

        np.testing.assert_allclose(
            to_numpy(circuit(True)),
            to_numpy(circuit(False)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_to_matchgate(self):
        sptm = SptmFermionicSuperposition(wires=[0, 1])
        mg = sptm.to_matchgate()
        assert isinstance(mg, FermionicSuperposition)
