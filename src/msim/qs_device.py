import pennylane as qml
from pennylane import numpy as np


class FermionicDevice(qml.QubitDevice):
    name = 'Fermionic Simulator'
    short_name = "gince.qubit"
    pennylane_requires = ">=0.32"
    version = "0.0.1"
    author = "Jérémie Gince"

    operations = {"RX", "RY", "RZ", "PhaseShift"}
    observables = {"PauliX", "PauliZ"}

    def __init__(self):
        super().__init__(wires=1, shots=None)

        # create the initial state
        self._state = np.array([0, 1])

        # create a variable for future copies of the state
        self._pre_rotated_state = None

    @property
    def state(self):
        return self._pre_rotated_state

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=True,
            supports_finite_shots=False,
            supports_tensor_observables=False
        )
        return capabilities

    def apply(self, operations, rotations=None, **kwargs):
        for op in operations:
            # We update the state by applying the matrix representation of the gate
            self._state = qml.matrix(op) @ self._state

        # store the pre-rotated state
        self._pre_rotated_state = self._state.copy()

        # apply the circuit rotations
        for rot in rotations or []:
            self._state = qml.matrix(rot) @ self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        real = self._real(self._state)
        imag = self._imag(self._state)
        prob = self.marginal_prob(real ** 2 + imag ** 2, wires)
        return prob

    def reset(self):
        """Reset the device"""
        self._state = np.array([0, 1])

