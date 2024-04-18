import itertools
import warnings
from copy import deepcopy
from typing import Iterable, Tuple, Union, Callable, Any, Optional, List

import numpy as np
import psutil
import tqdm
from scipy import sparse
import pythonbasictools as pbt
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.ops.qubit.observables import BasisStateProjector

from .device_utils import _VHMatchgatesContainer, _VerticalMatchgatesContainer, _HorizontalMatchgatesContainer
from ..operations.matchgate_operation import MatchgateOperation, _SingleParticleTransitionMatrix
from ..base.lookup_table import NonInteractingFermionicLookupTable
from .. import utils


class NonInteractingFermionicDevice(qml.QubitDevice):
    """
    The Non-Interacting Fermionic Simulator device.

    :param wires: The number of wires of the device
    :type wires: Union[int, Wires, List[int]]
    :param r_dtype: The data type for the real part of the state vector
    :type r_dtype: np.dtype
    :param c_dtype: The data type for the complex part of the state vector
    :type c_dtype: np.dtype

    :kwargs: Additional keyword arguments

    :keyword prob_strategy: The strategy to compute the probabilities. Can be either "lookup_table" or "explicit_sum".
        Defaults to "lookup_table".
    :type prob_strategy: str
    :keyword majorana_getter: The Majorana getter to use. Defaults to a new instance of MajoranaGetter.
    :type majorana_getter: MajoranaGetter
    :keyword contraction_method: The contraction method to use. Can be either None or "neighbours".
        Defaults to None.
    :type contraction_method: Optional[str]
    :keyword pfaffian_method: The method to compute the Pfaffian. Can be either "det" or "P". Defaults to "det".
    :type pfaffian_method: str
    :keyword n_workers: The number of workers to use for multiprocessing. Defaults to 0.
    :type n_workers: int


    :ivar prob_strategy: The strategy to compute the probabilities
    :vartype prob_strategy: str
    :ivar majorana_getter: The Majorana getter to use
    :vartype majorana_getter: MajoranaGetter
    :ivar contraction_method: The contraction method to use
    :vartype contraction_method: Optional[str]
    :ivar pfaffian_method: The method to compute the Pfaffian
    :vartype pfaffian_method: str
    :ivar n_workers: The number of workers to use for multiprocessing
    :vartype n_workers: int
    :ivar basis_state_index: The index of the basis state
    :vartype basis_state_index: int
    :ivar sparse_state: The sparse state of the device
    :vartype sparse_state: sparse.coo_array
    :ivar state: The state of the device
    :vartype state: np.ndarray
    :ivar is_state_initialized: Whether the state is initialized
    :vartype is_state_initialized: bool
    :ivar transition_matrix: The transition matrix of the device
    :vartype transition_matrix: np.ndarray
    :ivar lookup_table: The lookup table of the device
    :vartype lookup_table: NonInteractingFermionicLookupTable
    :ivar memory_usage: The memory usage of the device
    :vartype memory_usage: int


    :Note: This device is a simulator for non-interacting fermions. It is based on the ``default.qubit`` device.
    :Note: This device supports batch execution.
    :Note: This device is in development, and its API is subject to change.
    """
    name = 'Non-Interacting Fermionic Simulator'
    short_name = "nif.qubit"
    pennylane_requires = "==0.32"
    version = "0.0.1"
    author = "Jérémie Gince"

    operations = {
        MatchgateOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(MatchgateOperation)],
        "BasisEmbedding", "StatePrep", "BasisState", "Snapshot"
    }
    observables = {"BasisStateProjector", "Projector", "Identity"}

    prob_strategies = {"lookup_table", "explicit_sum"}
    contraction_methods = {None, "neighbours", "vertical", "horizontal"}
    pfaffian_methods = {"det", "P"}

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            supports_broadcasting=True,
            returns_state=False,
            supports_finite_shots=False,
            supports_tensor_observables=False,
            # passthru_devices={
            #     "tf": "default.qubit.tf",
            #     "torch": "default.qubit.torch",
            #     "autograd": "default.qubit.autograd",
            #     "jax": "default.qubit.jax",
            # },
        )
        return capabilities

    @classmethod
    def update_single_particle_transition_matrix(cls, single_transition_particle_matrix, other):
        l_interface = qml.math.get_interface(single_transition_particle_matrix)
        other_interface = qml.math.get_interface(other)
        l_priority = cls.casting_priorities.index(l_interface)
        other_priority = cls.casting_priorities.index(other_interface)
        if l_priority < other_priority:
            single_transition_particle_matrix = utils.math.convert_and_cast_like(
                single_transition_particle_matrix, other
            )
        elif l_priority > other_priority:
            other = utils.math.convert_and_cast_like(
                other, single_transition_particle_matrix
            )
        # single_transition_particle_matrix = utils.math.convert_and_cast_like(
        #     single_transition_particle_matrix, other
        # )
        single_transition_particle_matrix = qml.math.einsum(
            "...ij,...jl->...il",
            single_transition_particle_matrix, other
        )
        return single_transition_particle_matrix

    @classmethod
    def prod_single_particle_transition_matrices(cls, first, sptm_list):
        """
        Compute the product of the single particle transition matrices of a list of
        single particle transition matrix.

        :param first: The first single particle transition matrix
        :param sptm_list: The list of single particle transition matrices
        :return: The product of the single particle transition matrices with the first one
        """
        sptm = first
        for op_r in sptm_list:
            if op_r is not None:
                sptm = cls.update_single_particle_transition_matrix(sptm, op_r)
        return sptm

    def __init__(
            self,
            wires: Union[int, Wires, List[int]] = 2,
            *,
            r_dtype=np.float64,
            c_dtype=np.complex128,
            analytic=None,
            **kwargs
    ):
        if np.isscalar(wires):
            assert wires > 1, "At least two wires are required for this device."
        else:
            assert len(wires) > 1, "At least two wires are required for this device."
        super().__init__(wires=wires, shots=None, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)

        self.prob_strategy = kwargs.get("prob_strategy", "lookup_table").lower()
        assert self.prob_strategy in self.prob_strategies, (
            f"The probability strategy must be one of {self.prob_strategies}. "
            f"Got {self.prob_strategy} instead."
        )
        self._debugger = kwargs.get("debugger", None)

        # create the initial state
        self._basis_state_index = 0
        self._sparse_state = None
        self._pre_rotated_sparse_state = None
        self._state = None
        self._pre_rotated_state = None

        # create a variable for future copies of the state
        self._transition_matrix = None
        self._lookup_table = None

        self.majorana_getter = kwargs.get("majorana_getter", utils.MajoranaGetter(self.num_wires, maxsize=256))
        assert isinstance(self.majorana_getter, utils.MajoranaGetter), (
            f"The majorana_getter must be an instance of {utils.MajoranaGetter}. "
            f"Got {type(self.majorana_getter)} instead."
        )
        assert self.majorana_getter.n == self.num_wires, (
            f"The majorana_getter must be initialized with {self.num_wires} wires. "
            f"Got {self.majorana_getter.n} instead."
        )
        self.contraction_method = kwargs.get("contraction_method", None)
        assert self.contraction_method in self.contraction_methods, (
            f"The contraction method must be one of {self.contraction_methods}. "
            f"Got {self.contraction_method} instead."
        )
        self.pfaffian_method = kwargs.get("pfaffian_method", "det")
        assert self.pfaffian_method in self.pfaffian_methods, (
            f"The pfaffian method must be one of {self.pfaffian_methods}. "
            f"Got {self.pfaffian_method} instead."
        )
        self.n_workers = kwargs.get("n_workers", 0)
        self.p_bar: Optional[tqdm.tqdm] = kwargs.get("p_bar", None)
        self.show_progress = kwargs.get("show_progress", self.p_bar is not None)
    
    @property
    def basis_state_index(self) -> int:
        if self._basis_state_index is None:
            if self._sparse_state is not None:
                return self._sparse_state.indices[0]
            assert self._state is not None, "The state is not initialized."
            return np.argmax(np.abs(self.state))
        return self._basis_state_index
    
    @property
    def sparse_state(self) -> sparse.coo_array:
        if self._sparse_state is None:
            if self._basis_state_index is not None:
                return self._create_basis_sparse_state(self._basis_state_index)
            assert self._state is not None, "The state is not initialized."
            return sparse.coo_array(self.state)
        self._sparse_state.reshape((-1, 2**self.num_wires))
        # if self._sparse_state.shape[0] == 1:
        #     self._sparse_state = self._sparse_state.reshape(-1)
        return self._sparse_state

    @property
    def state(self) -> np.ndarray:
        """
        Return the state of the device.

        :return: state vector of the device
        :rtype: array[complex]

        :Note: This function comes from the ``default.qubit`` device.
        """
        if self._state is None:
            if self._basis_state_index is not None:
                pre_state = self._create_basis_state(self._basis_state_index)
            else:
                assert self._sparse_state is not None, "The sparse state is not initialized."
                pre_state = self.sparse_state.toarray()
        else:
            pre_state = self._pre_rotated_state
        dim = 2**self.num_wires
        batch_size = self._get_batch_size(pre_state, (2,) * self.num_wires, dim)
        # Do not flatten the state completely but leave the broadcasting dimension if there is one
        shape = (batch_size, dim) if batch_size is not None else (dim,)
        return self._reshape(pre_state, shape)
    
    @property
    def is_state_initialized(self) -> bool:
        return self._state is not None or self._sparse_state is not None or self._basis_state_index is not None

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
    
    @property
    def memory_usage(self):
        mem = 0
        if self._basis_state_index is not None:
            arr = np.asarray([self._basis_state_index])
            mem += arr.size * arr.dtype.itemsize
        if self._state is not None:
            mem += self._state.size * self._state.dtype.itemsize
        if self._sparse_state is not None:
            mem += self._sparse_state.data.size * self._sparse_state.data.dtype.itemsize
            mem += self._sparse_state.indices.size * self._sparse_state.indices.dtype.itemsize
            mem += self._sparse_state.indptr.size * self._sparse_state.indptr.dtype.itemsize
        if self._pre_rotated_state is not None:
            mem += self._pre_rotated_state.size * self._pre_rotated_state.dtype.itemsize
        if self._pre_rotated_sparse_state is not None:
            mem += self._pre_rotated_sparse_state.data.size * self._pre_rotated_sparse_state.data.dtype.itemsize
            mem += self._pre_rotated_sparse_state.indices.size * self._pre_rotated_sparse_state.indices.dtype.itemsize
            mem += self._pre_rotated_sparse_state.indptr.size * self._pre_rotated_sparse_state.indptr.dtype.itemsize
        if self._transition_matrix is not None:
            size = qml.math.prod(qml.math.shape(self._transition_matrix))
            mem += size * self._transition_matrix.dtype.itemsize
        if self._lookup_table is not None:
            mem += self._lookup_table.memory_usage
        return mem
    
    def get_sparse_or_dense_state(self) -> Union[int, sparse.coo_array, np.ndarray]:
        if self._basis_state_index is not None:
            return self.basis_state_index
        elif self._state is not None:
            return self.state
        elif self._sparse_state is not None:
            return self.sparse_state
        else:
            raise RuntimeError("The state is not initialized.")

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
    
    def _create_basis_sparse_state(self, index) -> sparse.coo_array:
        """
        Create a computational basis state over all wires.

        :param index: integer representing the computational basis state
        :type index: int
        :return: complex coo_array of shape ``[2]*self.num_wires`` representing the statevector of the basis state

        :Note: This function does not support broadcasted inputs yet.
        :Note: This function comes from the ``default.qubit`` device.
        """
        sparse_state = sparse.coo_array(([1], ([index], [0])), shape=(2**self.num_wires, 1), dtype=self.C_DTYPE)
        return sparse_state

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
        self._sparse_state = None

    def _apply_basis_state(self, state, wires):
        """Initialize the sparse state vector in a specified computational basis state.

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
        
        self._basis_state_index = num
        self._sparse_state = None
        self._state = None

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
    
    def contract_operations(self, operations, contraction_method=None):
        contraction_method = contraction_method or self.contraction_method
        if contraction_method is None:
            return operations
        elif contraction_method in ["neighbours", "vertical", "horizontal"]:
            return self.do_neighbours_contraction(operations)
        else:
            raise NotImplementedError(
                f"The contraction method {contraction_method} is not implemented."
            )

    def do_neighbours_contraction(self, operations):
        if len(operations) <= 1:
            return operations
        queue = operations.copy()
        new_operations = []
        if self.contraction_method == "neighbours":
            container = _VHMatchgatesContainer()
        elif self.contraction_method == "vertical":
            container = _VerticalMatchgatesContainer()
        elif self.contraction_method == "horizontal":
            container = _HorizontalMatchgatesContainer()
        else:
            raise ValueError(f"Contraction method {self.contraction_method} is not implemented.")

        self.initialize_p_bar(total=len(operations), initial=0, desc=f"{self.contraction_method} contraction")
        for i, op in enumerate(operations):
            if not isinstance(op, MatchgateOperation):
                new_operations.append(op)
                if container:
                    new_operations.append(container.contract())
                    container.clear()
                self.p_bar_set_n(i + 1)
                continue
            new_op = container.push_contract(op)
            if new_op is not None:
                new_operations.append(new_op)
            self.p_bar_set_n(i + 1)

        if container:
            new_operations.append(container.contract())
            container.clear()
        self.close_p_bar()
        return new_operations

    def batch_execute(self, circuits):
        """Execute a batch of quantum circuits on the device.

        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.

        For plugin developers: This function should be overwritten if the device can efficiently run multiple
        circuits on a backend, for example using parallel and/or asynchronous executions.

        Args:
            circuits (list[~.tape.QuantumTape]): circuits to execute on the device

        Returns:
            list[array[float]]: list of measured value(s)
        """
        if not qml.active_return():
            return self._batch_execute_legacy(circuits=circuits)

        results = []
        for circuit in circuits:
            # we need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()

            res = self.execute(circuit)
            results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results

    def apply_state_prep(self, operation, **kwargs) -> bool:
        """
        Apply a state preparation operation to the device. Will set the internal state of the device
        only if the operation is a state preparation operation and then return True. Otherwise, it will
        return False.

        :param operation: The operation to apply
        :param kwargs: Additional keyword arguments
        :return: True if the operation was applied, False otherwise
        :rtype: bool
        """
        is_applied = True
        if isinstance(operation, qml.StatePrep):
            self._apply_state_vector(operation.parameters[0], operation.wires)
        elif isinstance(operation, qml.BasisState):
            self._apply_basis_state(operation.parameters[0], operation.wires)
        elif isinstance(operation, qml.Snapshot):
            if self._debugger and self._debugger.active:
                state_vector = np.array(self._flatten(self._state))
                if operation.tag:
                    self._debugger.snapshots[operation.tag] = state_vector
                else:
                    self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
        elif isinstance(operation, qml.pulse.ParametrizedEvolution):
            self._state = self._apply_parametrized_evolution(self._state, operation)
        else:
            is_applied = False
        return is_applied

    def gather_single_particle_transition_matrix(self, operation):
        """
        Gather the single particle transition matrix of a given operation.

        :param operation: The operation to gather the single particle transition matrix from
        :return: The single particle transition matrix of the operation
        """
        if isinstance(operation, qml.Identity):
            return None

        if isinstance(operation, _SingleParticleTransitionMatrix):
            return operation.pad(self.wires).matrix

        assert operation.name in self.operations, f"Operation {operation.name} is not supported."
        op_r = None
        if isinstance(operation, MatchgateOperation):
            op_r = operation.get_padded_single_particle_transition_matrix(self.wires)
        return op_r

    def gather_single_particle_transition_matrices(self, operations) -> list:
        """
        Gather the single particle transition matrices of a list of operations.

        :param operations: The operations to gather the single particle transition matrices from
        :return: The single particle transition matrices of the operations
        """
        single_particle_transition_matrices = []
        for operation in operations:
            op_r = self.gather_single_particle_transition_matrix(operation)
            if op_r is not None:
                single_particle_transition_matrices.append(op_r)
        return single_particle_transition_matrices

    def gather_single_particle_transition_matrices_mp(self, operations) -> list:
        """
        Gather the single particle transition matrices of a list of operations. Will use multiprocessing if the number
        of workers is different from 0.

        :param operations: The operations to gather the single particle transition matrices from
        :return: The single particle transition matrices of the operations
        """
        if len(operations) == 1:
            return [self.gather_single_particle_transition_matrix(operations[0])]
        # return pbt.apply_func_multiprocess(
        #     func=self.gather_single_particle_transition_matrix,
        #     iterable_of_args=[(op,) for op in operations],
        #     nb_workers=self.n_workers,
        #     verbose=False, n
        # )
        n_processes = self.n_workers
        if self.n_workers == -1:
            n_processes = psutil.cpu_count(logical=True)
        elif self.n_workers == -2:
            n_processes = psutil.cpu_count(logical=False)
        elif self.n_workers < 0:
            raise ValueError("The number of workers must be greater or equal than 0 or in [0, -2].")

        if n_processes == 0 or n_processes == 1:
            return self.gather_single_particle_transition_matrices(operations)

        op_splits = np.array_split(operations, n_processes)
        sptm_outputs = pbt.apply_func_multiprocess(
            func=self.gather_single_particle_transition_matrices,
            iterable_of_args=[(op_split,) for op_split in op_splits],
            nb_workers=n_processes,
            verbose=False,
        )
        return list(itertools.chain(*sptm_outputs))

    def apply(self, operations, rotations=None, **kwargs):
        """
        This method applies a list of operations to the device. It will update the ``_transition_matrix`` attribute
        of the device.

        :Note: if the number of workers is different from 0, this method will use multiprocessing method
            :py:meth:`apply_mp` to apply the operations.

        :param operations: The operations to apply
        :param rotations: The rotations to apply
        :param kwargs: Additional keyword arguments. The keyword arguments are passed to the :py:meth:`apply_mp` method.
        :return: None
        """
        if self.n_workers != 0:
            return self.apply_mp(operations, rotations, **kwargs)
        rotations = rotations or []
        if not isinstance(operations, Iterable):
            operations = [operations]
        global_single_transition_particle_matrix = pnp.eye(2 * self.num_wires)[None, ...]
        batched = False
        operations = self.contract_operations(operations, self.contraction_method)
        self.initialize_p_bar(total=len(operations), desc="Applying operations")
        # apply the circuit operations
        for i, op in enumerate(operations):
            op_r = None
            if i > 0 and isinstance(op, (qml.StatePrep, qml.BasisState)):
                raise qml.DeviceError(
                    f"Operation {op.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            if isinstance(op, qml.Identity):
                continue

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
            elif isinstance(op, _SingleParticleTransitionMatrix):
                self.p_bar_set_postfix_str(
                    f"Padding single particle transition matrix for {getattr(op, 'name', op.__class__.__name__)}"
                )
                op_r = op.pad(self.wires).matrix
            else:
                if isinstance(op, MatchgateOperation):
                    self.p_bar_set_postfix_str(
                        f"Computing single particle transition matrix for {getattr(op, 'name', op.__class__.__name__)}"
                    )
                    op_r = op.get_padded_single_particle_transition_matrix(self.wires)
                else:
                    assert op.name in self.operations, f"Operation {op.name} is not supported."
            if op_r is not None:
                batched = batched or (qml.math.ndim(op_r) > 2)
                self.p_bar_set_postfix_str(f"Applying operation {getattr(op, 'name', op.__class__.__name__)}")
                global_single_transition_particle_matrix = self.update_single_particle_transition_matrix(
                    global_single_transition_particle_matrix, op_r
                )
            self.p_bar_set_n(i + 1)

        if not batched:
            global_single_transition_particle_matrix = global_single_transition_particle_matrix[0]
        # store the pre-rotated state
        self._pre_rotated_sparse_state = self._sparse_state
        self._pre_rotated_state = self._state

        assert rotations is None or np.asarray([rotations]).size == 0, "Rotations are not supported"
        # apply the circuit rotations
        # for operation in rotations:
        #     self._state = self._apply_operation(self._state, operation)
        self.p_bar_set_postfix_str("Computing transition matrix")
        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(
            global_single_transition_particle_matrix
        )
        self.p_bar_set_postfix_str("Transition matrix computed")
        self.close_p_bar()

    def prod_single_particle_transition_matrices_mp(self, sptm_list):
        """
        Compute the product of the single particle transition matrices of a list of
        single particle transition matrix using multiprocessing.

        :param sptm_list: The list of single particle transition matrices
        :return: The product of the single particle transition matrices
        """
        if len(sptm_list) == 1:
            return sptm_list[0]

        sptm = pnp.eye(2 * self.num_wires)[None, ...]
        n_processes = self.n_workers
        if self.n_workers == -1:
            n_processes = psutil.cpu_count(logical=True)
        elif self.n_workers == -2:
            n_processes = psutil.cpu_count(logical=False)
        elif self.n_workers < 0:
            raise ValueError("The number of workers must be greater or equal than 0 or in [0, -2].")

        if n_processes == 0 or n_processes == 1:
            return self.prod_single_particle_transition_matrices(sptm, sptm_list)

        self.initialize_p_bar(total=len(sptm_list), desc="Computing single particle transition matrices")
        sptm_splits = np.array_split(sptm_list, n_processes)
        sptm_outputs = pbt.apply_func_multiprocess(
            func=self.prod_single_particle_transition_matrices,
            iterable_of_args=[(deepcopy(sptm), sptm_split) for sptm_split in sptm_splits],
            nb_workers=n_processes,
            verbose=False,
            callbacks=[self.update_p_bar],
        )
        self.close_p_bar()
        if len(sptm_outputs) == 1:
            return sptm_outputs[0]
        return self.prod_single_particle_transition_matrices(sptm, sptm_outputs)

    def apply_mp(self, operations, rotations=None, **kwargs):
        """
        Apply a list of operations to the device using multiprocessing. This method will update the
        ``_transition_matrix`` attribute of the device.

        :param operations: The operations to apply
        :param rotations: The rotations to apply
        :param kwargs: Additional keyword arguments

        :return: None
        """
        rotations = rotations or []
        if not isinstance(operations, Iterable):
            operations = [operations]

        operations = self.contract_operations(operations, self.contraction_method)

        remove_first = self.apply_state_prep(operations[0])
        if remove_first:
            operations = operations[1:]

        sptm_list = self.gather_single_particle_transition_matrices_mp(operations)
        batched = any([qml.math.ndim(op_r) > 2 for op_r in sptm_list])
        global_single_particle_transition_matrix = self.prod_single_particle_transition_matrices_mp(sptm_list)

        if not batched and qml.math.ndim(global_single_particle_transition_matrix) > 2:
            global_single_particle_transition_matrix = global_single_particle_transition_matrix[0]
        # store the pre-rotated state
        self._pre_rotated_sparse_state = self._sparse_state
        self._pre_rotated_state = self._state

        assert rotations is None or np.asarray([rotations]).size == 0, "Rotations are not supported"
        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(
            global_single_particle_transition_matrix
        )

    def get_prob_strategy_func(self) -> Callable[[Wires, Any], float]:
        if self.prob_strategy == "lookup_table":
            return self.compute_probability_of_target_using_lookup_table
        elif self.prob_strategy == "explicit_sum":
            return self.compute_probability_of_target_using_explicit_sum
        else:
            raise NotImplementedError(f"Probability strategy {self.prob_strategy} is not implemented.")

    def analytic_probability(self, wires=None):
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        wires_binary_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))
        prob_func = self.get_prob_strategy_func()
        return np.asarray([
            prob_func(wires, wires_binary_state)
            for wires_binary_state in wires_binary_states
        ])

    def expval(self, observable, shot_range=None, bin_size=None):
        if isinstance(observable, BasisStateProjector):
            wires = observable.wires
            prob_func = self.get_prob_strategy_func()
            return prob_func(wires, observable.parameters[0])
        elif isinstance(observable, qml.Identity):
            return 1.0
        else:
            raise NotImplementedError(f"Observable {observable.name} is not implemented.")

    def compute_probability_using_lookup_table(self, wires=None):
        warnings.warn(
            "This method is deprecated. Please use compute_probability_of_target_using_lookup_table instead.",
            DeprecationWarning
        )
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        # assert num_wires == 1, "Only one wire is supported for now."
        probs = pnp.zeros((num_wires, 2))
        for wire in wires:
            obs = self.lookup_table.get_observable(wire, self.get_sparse_or_dense_state())
            prob1 = pnp.real(utils.pfaffian(obs, method=self.pfaffian_method))
            prob0 = 1.0 - prob1
            probs[wire] = pnp.array([prob0, prob1])
        return probs.flatten()

    def compute_probability_of_target_using_lookup_table(self, wires=None, target_binary_state=None):
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        wires_indexes = wires.indices(wires)
        obs = self.lookup_table.get_observable_of_target_state(
            self.get_sparse_or_dense_state(), target_binary_state, wires_indexes,
            show_progress=self.show_progress,
        )
        return qml.math.real(utils.pfaffian(obs, method=self.pfaffian_method))

    def compute_probability_using_explicit_sum(self, wires=None):
        warnings.warn(
            "This method is deprecated. Please use compute_probability_of_target_using_explicit_sum instead.",
            DeprecationWarning
        )
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        ket_majorana_indexes = utils.decompose_state_into_majorana_indexes(
            self.get_sparse_or_dense_state(), n=self.num_wires
        )
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
                inner_op_list = [zero_state.T.conj(), bra_op, c_n, c_m, ket_op, zero_state]
                inner_product = utils.recursive_2in_operator(qml.math.dot, inner_op_list)
                t_wire_m = self.transition_matrix[wire, m]
                t_wire_n = pnp.conjugate(self.transition_matrix[wire, n])
                p += t_wire_m * t_wire_n * inner_product
            prob1 = pnp.real(p)
            prob0 = 1.0 - prob1
            probs[wire_idx] = pnp.array([prob0, prob1])
        return probs.flatten()

    def compute_probability_of_target_using_explicit_sum(self, wires=None, target_binary_state=None):
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        if target_binary_state is None:
            target_binary_state = np.ones(num_wires, dtype=int)
        if isinstance(target_binary_state, int):
            target_binary_state = np.array([target_binary_state])
        elif isinstance(target_binary_state, list):
            target_binary_state = np.array(target_binary_state)
        elif isinstance(target_binary_state, str):
            target_binary_state = utils.binary_string_to_vector(target_binary_state)
        else:
            target_binary_state = np.asarray(target_binary_state)
        assert len(target_binary_state) == num_wires, (
            f"The target binary state must have {num_wires} elements. "
            f"Got {len(target_binary_state)} instead."
        )
        if len(target_binary_state) > 4:
            warnings.warn(
                f"Computing the probability of a target state with more than 4 bits "
                f"may take a long time. Please consider using the lookup table strategy instead.",
                UserWarning,
            )

        ket_majorana_indexes = utils.decompose_state_into_majorana_indexes(
            self.get_sparse_or_dense_state(), n=self.num_wires
        )
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))
        zero_state = self._create_basis_state(0).flatten()
        # TODO: Dont compute the zero explicitly and only take the 00 element of the operator
        bra = utils.recursive_2in_operator(
            qml.math.dot, [zero_state.T.conj(), *[self.majorana_getter(i) for i in bra_majorana_indexes]]
        )
        ket = utils.recursive_2in_operator(
            qml.math.dot, [*[self.majorana_getter(i) for i in ket_majorana_indexes], zero_state]
        )

        np_iterator = np.ndindex(tuple([2*self.num_wires for _ in range(2 * len(target_binary_state))]))
        sum_elements = pbt.apply_func_multiprocess(
            func=self._compute_partial_prob_of_m_n_vector,
            iterable_of_args=[
                (m_n_vector, target_binary_state, wires, bra, ket)
                for m_n_vector in np_iterator
            ],
            nb_workers=self.n_workers,
            verbose=False,
        )
        target_prob = sum(sum_elements, start=0.0)
        # target_prob = sum(
        #     (
        #         self._compute_partial_prob_of_m_n_vector(
        #             m_n_vector=m_n_vector,
        #             target_binary_state=target_binary_state,
        #             wires=wires,
        #             bra=bra,
        #             ket=ket,
        #         )
        #         for m_n_vector in np_iterator
        #     ),
        #     start=0.0,
        # )
        return pnp.real(target_prob)

    def _compute_partial_prob_of_m_n_vector(
            self,
            m_n_vector,
            target_binary_state,
            wires,
            bra,
            ket,
    ):
        inner_op_list = [
            self.majorana_getter((1 - b) * i + b * j, (1 - b) * j + b * i)
            for i, j, b in zip(m_n_vector[::2], m_n_vector[1::2], target_binary_state)
        ]
        inner_product = utils.recursive_2in_operator(qml.math.dot, [bra, *inner_op_list, ket])
        t_wire_m = qml.math.prod(self.transition_matrix[wires, m_n_vector[::2]])
        t_wire_n = qml.math.prod(pnp.conjugate(self.transition_matrix[wires, m_n_vector[1::2]]))
        product_coeff = t_wire_m * t_wire_n
        return product_coeff * inner_product

    def _asarray(self, x, dtype=None):
        r"""
        Convert the input to an array of type ``dtype``.

        :Note: If the input is on cuda, it will be copied to cpu.

        :param x: input to be converted
        :param dtype: type of the output array
        :return: array of type ``dtype``
        """
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
        except ImportError:
            pass
        return qml.math.array(x, dtype=dtype)

    def reset(self):
        """Reset the device"""
        self._basis_state_index = 0
        self._sparse_state = None
        self._pre_rotated_sparse_state = self._sparse_state
        self._state = None
        self._pre_rotated_state = self._state
        self._transition_matrix = None
        self._lookup_table = None

    def update_p_bar(self, *args, **kwargs):
        if self.p_bar is None:
            return
        self.p_bar.update(*args, **kwargs)
        self.p_bar.refresh()

    def p_bar_set_n(self, n: int):
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def initialize_p_bar(self, *args, **kwargs):
        kwargs.setdefault("disable", not self.show_progress)
        if self.p_bar is None and not self.show_progress:
            return
        self.p_bar = tqdm.tqdm(*args, **kwargs)
        return self.p_bar

    def p_bar_set_postfix(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix(*args, **kwargs)
            self.p_bar.refresh()

    def p_bar_set_postfix_str(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix_str(*args, **kwargs)
            self.p_bar.refresh()

    def close_p_bar(self):
        if self.p_bar is not None:
            self.p_bar.close()
