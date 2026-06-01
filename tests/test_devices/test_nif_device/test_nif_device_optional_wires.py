import numpy as np
import pennylane as qml
import pytest

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import (
    SingleParticleTransitionMatrixOperation,
)


class TestNonInteractingFermionicDeviceOptionalWires:
    def test_device_created_without_wires(self):
        dev = NonInteractingFermionicDevice()
        assert dev.wires is None
        assert dev.num_wires is None

    def test_device_created_with_explicit_wires_unchanged(self):
        dev = NonInteractingFermionicDevice(wires=4)
        assert len(dev.wires) == 4
        assert dev.num_wires == 4

    def test_wires_inferred_from_circuit_via_qnode(self):
        dev = NonInteractingFermionicDevice()

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 0, 0, 0]), wires=range(4))
            return qml.expval(qml.PauliZ(0))

        circuit()
        assert dev.wires is not None
        assert len(dev.wires) == 4

    def test_expval_correct_with_inferred_wires(self):
        dev_inferred = NonInteractingFermionicDevice()
        dev_explicit = NonInteractingFermionicDevice(wires=2)

        @qml.qnode(dev_inferred)
        def circuit_inferred():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_explicit)
        def circuit_explicit():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        np.testing.assert_allclose(float(circuit_inferred()), float(circuit_explicit()), atol=1e-6)

    def test_probs_correct_with_inferred_wires(self):
        dev_inferred = NonInteractingFermionicDevice()
        dev_explicit = NonInteractingFermionicDevice(wires=2)

        @qml.qnode(dev_inferred)
        def circuit_inferred():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            return qml.probs(wires=[0, 1])

        @qml.qnode(dev_explicit)
        def circuit_explicit():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            return qml.probs(wires=[0, 1])

        np.testing.assert_allclose(circuit_inferred(), circuit_explicit(), atol=1e-6)

    def test_sampling_with_inferred_wires(self):
        shots = 32
        dev = NonInteractingFermionicDevice(shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            return qml.sample()

        samples = circuit()
        assert samples.shape == (shots, 2)

    def test_device_reused_across_circuits_with_same_wires(self):
        dev = NonInteractingFermionicDevice()

        @qml.qnode(dev)
        def circuit(basis):
            qml.BasisState(np.array(basis), wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result_0 = float(circuit([0, 0]))
        result_1 = float(circuit([1, 0]))
        np.testing.assert_allclose(result_0, 1.0, atol=1e-6)
        np.testing.assert_allclose(result_1, -1.0, atol=1e-6)

    def test_apply_generator_infers_wires(self):
        dev = NonInteractingFermionicDevice()
        assert dev.wires is None

        op = SingleParticleTransitionMatrixOperation(np.eye(4), wires=[0, 1])
        dev.apply_generator(iter([op]))

        assert dev.wires is not None
        assert len(dev.wires) == 2

    def test_reset_with_no_wires_does_not_raise(self):
        dev = NonInteractingFermionicDevice()
        dev.reset()
        assert dev._state_prep_op is None
        assert dev._global_sptm is None

    def test_two_wire_minimum_enforced_without_explicit_wires(self):
        dev = NonInteractingFermionicDevice()

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0]), wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(AssertionError):
            circuit()
