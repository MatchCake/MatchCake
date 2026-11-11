import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestSptmFromCircuit:
    @staticmethod
    def layered_block(params, wires):
        for left, right in zip(wires[:-1], wires[1:]):
            mc.operations.CompRxRx(params, wires=[left, right])
            mc.operations.CompRzRz(params, wires=[left, right])
            mc.operations.CompRyRy(params, wires=[left, right])

    @staticmethod
    def swap_block(params, wires):
        mc.operations.CompRyRy(params, wires=wires)
        mc.operations.fSWAP(wires=wires)
        mc.operations.CompHH(wires=wires)

    @staticmethod
    def qubit_probs(qfunc, n_wires):
        @qml.qnode(qml.device("default.qubit", wires=n_wires))
        def circuit():
            qfunc()
            return qml.probs(wires=range(n_wires))

        return qml.math.to_numpy(circuit())

    @staticmethod
    def nif_probs(qfunc, n_wires):
        @qml.qnode(qml.device("nif.qubit", wires=n_wires))
        def circuit():
            qfunc()
            return qml.probs(wires=range(n_wires))

        return qml.math.to_numpy(circuit())

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize("n_wires", [2, 3, 4, 5])
    def test_matches_qubit_simulation(self, n_wires):
        params = np.random.uniform(-np.pi, np.pi, size=2)
        wires = list(range(n_wires))
        sptm = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=wires)

        np.testing.assert_equal(sptm.wires.tolist(), wires)
        np.testing.assert_allclose(
            self.nif_probs(lambda: qml.apply(sptm), n_wires),
            self.qubit_probs(lambda: self.layered_block(params, wires=wires), n_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_matches_device_global_sptm(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)
        wires = [0, 1, 2, 3]
        sptm = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=wires)
        nif_device = mc.NonInteractingFermionicDevice(wires=len(wires))
        with qml.QueuingManager.stop_recording():
            tape = qml.tape.make_qscript(self.layered_block)(params, wires=wires)
        nif_device.apply(tape.operations)

        np.testing.assert_allclose(
            qml.math.to_numpy(sptm.sptm_matrix()),
            qml.math.to_numpy(nif_device.global_sptm.sptm_matrix()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_merge_disjoint_circuits_then_next_block(self):
        params_a, params_b, params_next = np.random.uniform(-np.pi, np.pi, size=(3, 2))
        merged_sptm = SingleParticleTransitionMatrixOperation.from_operations(
            [
                SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params_a, wires=[0, 1]),
                SingleParticleTransitionMatrixOperation.from_circuit(self.swap_block, params_b, wires=[2, 3]),
            ]
        )

        def merged_qfunc():
            qml.apply(merged_sptm)
            self.layered_block(params_next, wires=[0, 1, 2, 3])

        def reference_qfunc():
            self.layered_block(params_a, wires=[0, 1])
            self.swap_block(params_b, wires=[2, 3])
            self.layered_block(params_next, wires=[0, 1, 2, 3])

        np.testing.assert_allclose(
            self.nif_probs(merged_qfunc, 4),
            self.qubit_probs(reference_qfunc, 4),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_chain_with_matmul(self):
        params_first, params_second = np.random.uniform(-np.pi, np.pi, size=(2, 2))
        first = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params_first, wires=[0, 1, 2])
        second = SingleParticleTransitionMatrixOperation.from_circuit(self.swap_block, params_second, wires=[1, 2])
        chained = first @ second

        def reference_qfunc():
            self.layered_block(params_first, wires=[0, 1, 2])
            self.swap_block(params_second, wires=[1, 2])

        np.testing.assert_allclose(
            self.nif_probs(lambda: qml.apply(chained), 3),
            self.qubit_probs(reference_qfunc, 3),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_wires_with_gap_use_continuous_span(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)

        def gapped_circuit():
            mc.operations.CompRxRx(params, wires=[0, 1])
            mc.operations.CompRyRy(params, wires=[3, 4])

        sptm = SingleParticleTransitionMatrixOperation.from_circuit(gapped_circuit)

        np.testing.assert_equal(sptm.wires.tolist(), [0, 1, 2, 3, 4])
        np.testing.assert_equal(sptm.shape, (10, 10))

    def test_measurements_and_identities_are_ignored(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)

        def circuit_with_extras(wires):
            qml.Identity(wires=wires[0])
            self.layered_block(params, wires=wires)
            return qml.probs(wires=wires)

        sptm = SingleParticleTransitionMatrixOperation.from_circuit(circuit_with_extras, wires=[0, 1, 2])
        expected = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=[0, 1, 2])

        np.testing.assert_allclose(
            qml.math.to_numpy(sptm.sptm_matrix()),
            qml.math.to_numpy(expected.sptm_matrix()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_queues_only_the_contracted_operation(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)
        with qml.queuing.AnnotatedQueue() as queue:
            sptm = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=[0, 1, 2])

        np.testing.assert_equal(len(queue.queue), 1)
        np.testing.assert_equal(queue.queue[0] is sptm, True)

    def test_inside_qnode_applies_circuit_once(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)
        wires = [0, 1, 2]

        np.testing.assert_allclose(
            self.nif_probs(
                lambda: SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=wires), 3
            ),
            self.qubit_probs(lambda: self.layered_block(params, wires=wires), 3),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_batched_params(self):
        batch_size = 3
        params = np.random.uniform(-np.pi, np.pi, size=(batch_size, 2))
        sptm = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=[0, 1, 2])

        np.testing.assert_equal(sptm.batch_size, batch_size)
        for batch_index in range(batch_size):
            single = SingleParticleTransitionMatrixOperation.from_circuit(
                self.layered_block, params[batch_index], wires=[0, 1, 2]
            )
            np.testing.assert_allclose(
                qml.math.to_numpy(sptm.sptm_matrix()[batch_index]),
                qml.math.to_numpy(single.sptm_matrix()),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_torch_params_keep_interface(self):
        params = torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), requires_grad=True)
        sptm = SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=[0, 1, 2])

        np.testing.assert_equal(qml.math.get_interface(sptm.sptm_matrix()), "torch")
        np.testing.assert_equal(sptm.sptm_matrix().requires_grad, True)

    def test_gradcheck_sptm_matrix(self):
        params = torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64, requires_grad=True)

        def sptm_matrix(circuit_params):
            return SingleParticleTransitionMatrixOperation.from_circuit(
                self.layered_block, circuit_params, wires=[0, 1, 2]
            ).sptm_matrix()

        torch.autograd.gradcheck(sptm_matrix, (params,))

    def test_gradcheck_merged_circuits_expval(self):
        params_a, params_b = (
            torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64, requires_grad=True)
            for _ in range(2)
        )
        params_next = torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64)

        def expval(circuit_params_a, circuit_params_b):
            merged_sptm = SingleParticleTransitionMatrixOperation.from_operations(
                [
                    SingleParticleTransitionMatrixOperation.from_circuit(
                        self.layered_block, circuit_params_a, wires=[0, 1]
                    ),
                    SingleParticleTransitionMatrixOperation.from_circuit(
                        self.swap_block, circuit_params_b, wires=[2, 3]
                    ),
                ]
            )

            @qml.qnode(qml.device("nif.qubit", wires=4), interface="torch")
            def circuit():
                qml.apply(merged_sptm)
                self.layered_block(params_next, wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(1) @ qml.Z(2))

            return circuit()

        torch.autograd.gradcheck(expval, (params_a, params_b))

    def test_qnode_gradient_matches_qubit_simulation(self):
        wires = [0, 1, 2]
        initial_params = np.random.uniform(-np.pi, np.pi, size=2)
        gradients = {}
        for device_name in ["nif.qubit", "default.qubit"]:
            params = torch.tensor(initial_params, dtype=torch.float64, requires_grad=True)

            @qml.qnode(qml.device(device_name, wires=len(wires)), interface="torch")
            def circuit():
                if device_name == "nif.qubit":
                    SingleParticleTransitionMatrixOperation.from_circuit(self.layered_block, params, wires=wires)
                else:
                    self.layered_block(params, wires=wires)
                return qml.expval(qml.Z(0) @ qml.Z(2))

            circuit().backward()
            gradients[device_name] = params.grad

        torch.testing.assert_close(
            gradients["nif.qubit"],
            gradients["default.qubit"],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "state_prep",
        [
            lambda: qml.BasisState(np.array([1, 0]), wires=[0, 1]),
            lambda: mc.operations.state_preparation.ProductState(np.array([[1.0, 0.0], [0.0, 1.0]]), wires=[0, 1]),
        ],
    )
    def test_raises_on_state_prep(self, state_prep):
        params = np.random.uniform(-np.pi, np.pi, size=2)

        def circuit_with_state_prep():
            state_prep()
            self.layered_block(params, wires=[0, 1])

        with pytest.raises(ValueError, match="state preparation"):
            SingleParticleTransitionMatrixOperation.from_circuit(circuit_with_state_prep)

    def test_raises_on_empty_circuit(self):
        def empty_circuit():
            return qml.probs(wires=[0, 1])

        with pytest.raises(ValueError, match="without operations"):
            SingleParticleTransitionMatrixOperation.from_circuit(empty_circuit)
