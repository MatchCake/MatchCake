from typing import Iterable

import pennylane as qml
from pennylane import numpy as pnp
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
    observables = {"PauliZ"}

    def __init__(self, wires=1):
        super().__init__(wires=wires, shots=None)

        # create the initial state
        self._state = pnp.array([0, 1])

        # create a variable for future copies of the state
        self._pre_rotated_state = None
        self._transition_matrix = None
        self._block_diagonal_matrix = None
        self._lookup_table = None

    @property
    def state(self):
        return self._pre_rotated_state
    
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

    def apply(self, operations, rotations=None, **kwargs):
        if not isinstance(operations, Iterable):
            operations = [operations]
        
        action_matrix = pnp.eye(2 * self.num_wires, dtype=complex)
        for op in operations:
            assert isinstance(op, MatchgateOperator), "Only MatchgateOperator is supported"
            action_matrix = op.action_matrix @ action_matrix
        
        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(action_matrix)
        
        # store the pre-rotated state
        self._pre_rotated_state = self._state.copy()

        # apply the circuit rotations
        # for rot in rotations or []:
        #     self._state = qml.matrix(rot) @ self._state
        assert rotations is None, "Rotations are not supported"

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None
        state_hamming_weight = utils.get_hamming_weight(self._state)
        
        if isinstance(wires, int):
            wires = [wires]
        assert len(wires) == 1, "Only one wire is supported for now."
        wire = wires[0]
        
        obs = self.lookup_table.get_observable(wire, state_hamming_weight)
        prob = pfaffian.pfaffian(obs)
        return prob

    def reset(self):
        """Reset the device"""
        self._state = pnp.array([0, 1])

