from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.wires import Wires
from .matchgate_operator import MatchgateOperator
from .lookup_table import NonInteractingFermionicLookupTable
from . import utils
from pfapack import pfaffian


class NonInteractingFermionicDevice(qml.QubitDevice):
    name = 'Non-Interacting Fermionic Simulator'
    short_name = "nif.qubit"
    pennylane_requires = ">=0.32"
    version = "0.0.1"
    author = "Jérémie Gince"

    operations = {"MatchgateOperator"}
    observables = {"PauliZ", "Identity"}

    def __init__(self, wires=2):
        assert wires > 1, "At least two wires are required for this device."
        super().__init__(wires=wires, shots=None)

        # create the initial state
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # create a variable for future copies of the state
        self._transition_matrix = None
        self._block_diagonal_matrix = None
        self._lookup_table = None
    
    @property
    def state(self):
        """
        Return the state of the device.
        
        :return: state vector of the device
        :rtype: array[complex]
        
        :Note: This function comes from the ``default.qubit`` device.
        """
        dim = 2**self.num_wires
        batch_size = self._get_batch_size(self._pre_rotated_state, (2,) * self.num_wires, dim)
        # Do not flatten the state completely but leave the broadcasting dimension if there is one
        shape = (batch_size, dim) if batch_size is not None else (dim,)
        return self._reshape(self._pre_rotated_state, shape)
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def block_diagonal_matrix(self):
        if self._block_diagonal_matrix is None:
            self._block_diagonal_matrix = utils.get_block_diagonal_matrix(self.num_wires)
        return self._block_diagonal_matrix
    
    @property
    def lookup_table(self):
        if self.transition_matrix is None:
            return None
        if self._lookup_table is None:
            self._lookup_table = NonInteractingFermionicLookupTable(
                self.transition_matrix,
                self.block_diagonal_matrix
            )
        return self._lookup_table

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=False,
            supports_finite_shots=False,
            supports_tensor_observables=False
        )
        return capabilities
    
    def _create_basis_state(self, index):
        """
        Create a computational basis state over all wires.
        
        :param index: integer representing the computational basis state
        :type index: int
        :return: complex array of shape ``[2]*self.num_wires`` representing the statevector of the basis state
        
        :Note: This function does not support broadcasted inputs yet.
        :Note: This function comes from the ``default.qubit`` device.
        """
        state = np.zeros(2**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [2] * self.num_wires)

    def apply(self, operations, rotations=None, **kwargs):
        if not isinstance(operations, Iterable):
            operations = [operations]
        
        action_matrix = pnp.eye(2 * self.num_wires, dtype=complex)
        for op in operations:
            assert isinstance(op, MatchgateOperator), "Only MatchgateOperator is supported"
            action_matrix = qml.math.dot(op.single_transition_particle_matrix, action_matrix)
            # action_matrix = qml.math.dot(action_matrix, op.single_transition_particle_matrix)

        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(action_matrix)
        
        # store the pre-rotated state
        # self._pre_rotated_state = self._state.copy()

        # apply the circuit rotations
        # for rot in rotations or []:
        #     self._state = qml.matrix(rot) @ self._state
        assert rotations is None or np.asarray([rotations]).size == 0, "Rotations are not supported"

    def analytic_probability(self, wires=None):
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)
        
        if self.state is None:
            return None
        state_hamming_weight = utils.get_hamming_weight(self.state)

        # assert num_wires == 1, "Only one wire is supported for now."
        probs = pnp.zeros((num_wires, 2))
        for wire in wires:
            obs = self.lookup_table.get_observable(wire, state_hamming_weight)
            prob1 = pnp.real(pfaffian.pfaffian(obs))
            prob0 = 1.0 - prob1
            probs[wire] = pnp.array([prob0, prob1])
        return probs.flatten()

    def reset(self):
        """Reset the device"""
        self._state = pnp.array([0, 1])

