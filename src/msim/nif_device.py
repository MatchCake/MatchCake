import itertools
from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
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

    operations = {"MatchgateOperator", "BasisEmbedding", "StatePrep", "BasisState", "Snapshot"}
    observables = {"PauliZ", "Identity"}

    prob_strategies = {"lookup_table", "explicit_sum"}

    def __init__(self, wires=2, **kwargs):
        assert wires > 1, "At least two wires are required for this device."
        super().__init__(wires=wires, shots=None)

        self.prob_strategy = kwargs.get("prob_strategy", "lookup_table").lower()
        assert self.prob_strategy in self.prob_strategies, (
            f"The probability strategy must be one of {self.prob_strategies}. "
            f"Got {self.prob_strategy} instead."
        )
        self._debugger = kwargs.get("debugger", None)

        self.single_transition_particle_matrices = []

        # create the initial state
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # create a variable for future copies of the state
        self._transition_matrix = None
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
    def lookup_table(self):
        if self.transition_matrix is None:
            return None
        if self._lookup_table is None:
            self._lookup_table = NonInteractingFermionicLookupTable(self.transition_matrix)
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

    @property
    def global_single_transition_particle_matrix(self):
        if len(self.single_transition_particle_matrices) > 0:
            if len(self.single_transition_particle_matrices) == 1:
                global_single_transition_particle_matrix = self.single_transition_particle_matrices[0]
            else:
                global_single_transition_particle_matrix = utils.recursive_2in_operator(
                    qml.math.dot, self.single_transition_particle_matrices
                )
        else:
            global_single_transition_particle_matrix = pnp.eye(2*self.num_wires)
        return global_single_transition_particle_matrix
    
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

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)
        dim = 2 ** len(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        batch_size = self._get_batch_size(state, (dim,), dim)
        output_shape = [2] * self.num_wires
        if batch_size is not None:
            output_shape.insert(0, batch_size)

        if len(device_wires) == self.num_wires and sorted(device_wires) == device_wires:
            # Initialize the entire device state with the input state
            self._state = self._reshape(state, output_shape)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        if batch_size is not None:
            state = self._scatter(
                (slice(None), ravelled_indices), state, [batch_size, 2**self.num_wires]
            )
        else:
            state = self._scatter(ravelled_indices, state, [2**self.num_wires])
        state = self._reshape(state, output_shape)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    def _apply_parametrized_evolution(self, state: TensorLike, operation: ParametrizedEvolution):
        """Applies a parametrized evolution to the input state.

        Args:
            state (array[complex]): input state
            operation (ParametrizedEvolution): operation to apply on the state
        """
        raise NotImplementedError(
            f"The device {self.short_name} cannot execute a ParametrizedEvolution operation. "
            "Please use the jax interface."
        )

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []
        if not isinstance(operations, Iterable):
            operations = [operations]

        # apply the circuit operations
        for i, op in enumerate(operations):
            if i > 0 and isinstance(op, (qml.StatePrep, qml.BasisState)):
                raise qml.DeviceError(
                    f"Operation {op.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            if isinstance(op, qml.StatePrep):
                self._apply_state_vector(op.parameters[0], op.wires)
            elif isinstance(op, qml.BasisState):
                self._apply_basis_state(op.parameters[0], op.wires)
            elif isinstance(op, qml.Snapshot):
                if self._debugger and self._debugger.active:
                    state_vector = np.array(self._flatten(self._state))
                    if op.tag:
                        self._debugger.snapshots[op.tag] = state_vector
                    else:
                        self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
            elif isinstance(op, qml.pulse.ParametrizedEvolution):
                self._state = self._apply_parametrized_evolution(self._state, op)
            else:
                assert op.name in self.operations, f"Operation {op.name} is not supported."
                if isinstance(op, MatchgateOperator):
                    self.single_transition_particle_matrices.append(op.single_transition_particle_matrix)

        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(
            self.global_single_transition_particle_matrix
        )

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        assert rotations is None or np.asarray([rotations]).size == 0, "Rotations are not supported"
        # apply the circuit rotations
        # for operation in rotations:
        #     self._state = self._apply_operation(self._state, operation)

    def analytic_probability(self, wires=None):
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)

        if self.state is None:
            return None

        if self.prob_strategy == "lookup_table":
            return self.compute_probability_using_lookup_table(wires)
        elif self.prob_strategy == "explicit_sum":
            return self.compute_probability_using_explicit_sum(wires)
        else:
            raise NotImplementedError(f"Probability strategy {self.prob_strategy} is not implemented.")

    def compute_probability_using_lookup_table(self, wires=None):
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        if self.state is None:
            return None

        # assert num_wires == 1, "Only one wire is supported for now."
        probs = pnp.zeros((num_wires, 2))
        for wire in wires:
            obs = self.lookup_table.get_observable(wire, self.state)
            prob1 = pnp.real(pfaffian.pfaffian(obs))
            prob0 = 1.0 - prob1
            probs[wire] = pnp.array([prob0, prob1])
        return probs.flatten()

    def compute_probability_using_explicit_sum(self, wires=None):
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        if self.state is None:
            return None

        ket_majorana_indexes = utils.decompose_state_into_majorana_indexes(self.state)
        ket_majorana_list = [utils.get_majorana(i, self.num_wires) for i in ket_majorana_indexes]
        if ket_majorana_list:
            ket_op = utils.recursive_2in_operator(pnp.matmul, ket_majorana_list)
        else:
            ket_op = pnp.eye(2*self.num_wires)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))
        bra_majorana_list = [utils.get_majorana(i, self.num_wires) for i in bra_majorana_indexes]
        if bra_majorana_list:
            bra_op = utils.recursive_2in_operator(pnp.matmul, bra_majorana_list)
        else:
            bra_op = pnp.eye(2*self.num_wires)
        zero_state = self._create_basis_state(0).flatten()
        probs = pnp.zeros((num_wires, 2))
        for wire_idx, wire in enumerate(wires):
            p = 0.0
            for m, n in np.ndindex((2*self.num_wires, 2*self.num_wires)):
                c_m = utils.get_majorana(m, self.num_wires)
                c_n = utils.get_majorana(n, self.num_wires)
                inner_op = qml.math.dot(bra_op, qml.math.dot(c_n, qml.math.dot(c_m, ket_op)))
                inner_product = qml.math.dot(qml.math.dot(zero_state.T.conj(), inner_op), zero_state)
                t_wire_m = self.transition_matrix[wire, m]
                t_wire_n = pnp.conjugate(self.transition_matrix[wire, n])
                p += t_wire_m * t_wire_n * inner_product
            prob1 = pnp.real(p)
            prob0 = 1.0 - prob1
            probs[wire_idx] = pnp.array([prob0, prob1])
        return probs.flatten()

    def reset(self):
        """Reset the device"""
        self._state = self._create_basis_state(0)

