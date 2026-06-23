import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError

import matchcake as mc
from matchcake import NonInteractingFermionicDevice

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestNifRzSupport:
    """Single-qubit ``R_Z`` support on the NIF device (``R_Z`` is Gaussian; ``R_X``/``R_Y`` are not)."""

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @staticmethod
    def _circuit(x):
        # Index the last axis so the body works for both 1-D params (7,) and batched params (B, 7).
        qml.BasisState(np.array([1, 0, 1]), wires=range(3))
        qml.RZ(x[..., 0], wires=0)
        mc.operations.CompRxRx(qml.math.stack([x[..., 1], x[..., 2]], axis=-1), wires=[0, 1])
        qml.RZ(x[..., 3], wires=2)
        mc.operations.CompRyRy(qml.math.stack([x[..., 4], x[..., 5]], axis=-1), wires=[1, 2])
        qml.RZ(x[..., 6], wires=1)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_probs_match_default_qubit(self, seed):
        x = np.random.default_rng(seed).uniform(-3, 3, size=7)

        @qml.qnode(qml.device("default.qubit", wires=3))
        def ref(x):
            self._circuit(x)
            return qml.probs(wires=range(3))

        @qml.qnode(NonInteractingFermionicDevice(wires=3))
        def got(x):
            self._circuit(x)
            return qml.probs(wires=range(3))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_expval_match_default_qubit(self):
        x = np.random.default_rng(3).uniform(-3, 3, size=7)
        observable = qml.PauliZ(0) @ qml.PauliZ(2) + 0.5 * qml.PauliZ(1)

        @qml.qnode(qml.device("default.qubit", wires=3))
        def ref(x):
            self._circuit(x)
            return qml.expval(observable)

        @qml.qnode(NonInteractingFermionicDevice(wires=3))
        def got(x):
            self._circuit(x)
            return qml.expval(observable)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_batched_params_match_default_qubit(self):
        x = np.random.default_rng(4).uniform(-3, 3, size=(5, 7))

        @qml.qnode(qml.device("default.qubit", wires=3))
        def ref(x):
            self._circuit(x)
            return qml.probs(wires=range(3))

        @qml.qnode(NonInteractingFermionicDevice(wires=3))
        def got(x):
            self._circuit(x)
            return qml.probs(wires=range(3))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_lone_rz_between_matchgates(self):
        # A standalone R_Z (no neighbouring matchgate on its wire) still contracts correctly.
        @qml.qnode(qml.device("default.qubit", wires=2))
        def ref():
            qml.BasisState(np.array([1, 1]), wires=range(2))
            qml.RZ(0.9, wires=0)
            qml.RZ(-1.4, wires=1)
            return qml.probs(wires=range(2))

        @qml.qnode(NonInteractingFermionicDevice(wires=2))
        def got():
            qml.BasisState(np.array([1, 1]), wires=range(2))
            qml.RZ(0.9, wires=0)
            qml.RZ(-1.4, wires=1)
            return qml.probs(wires=range(2))

        np.testing.assert_allclose(np.asarray(got()), np.asarray(ref()), atol=ATOL_MATRIX_COMPARISON)

    def test_rz_is_listed_as_supported(self):
        assert "RZ" in NonInteractingFermionicDevice._supported_ops

    def test_single_qubit_rx_is_rejected_as_non_gaussian(self):
        # R_X on one qubit is not a matchgate (its generator is a single Majorana with a JW string),
        # so it cannot be simulated and must raise rather than silently produce a wrong answer.
        @qml.qnode(NonInteractingFermionicDevice(wires=2))
        def got():
            qml.RX(0.5, wires=0)
            return qml.probs(wires=range(2))

        with pytest.raises(DeviceError):
            got()
