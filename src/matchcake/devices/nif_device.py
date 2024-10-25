import itertools
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Iterable, Tuple, Union, Callable, Any, Optional, List, Literal

import numpy as np
import psutil
import tqdm
from pennylane.tape import QuantumTape
from scipy import sparse
import pythonbasictools as pbt
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.ops.qubit.observables import BasisStateProjector

from ..operations.matchgate_operation import MatchgateOperation
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    _SingleParticleTransitionMatrix,
    SingleParticleTransitionMatrixOperation,
)
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..base.lookup_table import NonInteractingFermionicLookupTable
from .. import utils
from .sampling_strategies import get_sampling_strategy, SamplingStrategy
from .probability_strategies import get_probability_strategy, ProbabilityStrategy
from .contraction_strategies import get_contraction_strategy, ContractionStrategy
from ..utils import torch_utils
from ..utils.math import convert_and_cast_like


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
        SingleParticleTransitionMatrixOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(SingleParticleTransitionMatrixOperation)],
        "BasisEmbedding", "StatePrep", "BasisState", "Snapshot"
    }
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "BatchHamiltonian",
        "Hermitian",
        "Identity",
        "Projector",
        "Sum",
        "Sprod",
        "Prod",
        BatchHamiltonian.__name__,
    }

    DEFAULT_PROB_STRATEGY = "LookupTable"
    DEFAULT_CONTRACTION_METHOD = "neighbours"
    DEFAULT_SAMPLING_STRATEGY = "QubitByQubitSampling"
    pfaffian_methods = {"det", "bLTL", "bH", "cuda_det"}
    DEFAULT_PFAFFIAN_METHOD = "det"

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            supports_broadcasting=True,
            returns_state=False,
            supports_finite_shots=True,
            supports_tensor_observables=True,
            passthru_interface="torch",
        )
        return capabilities

    @classmethod
    def update_single_particle_transition_matrix(cls, single_particle_transition_matrix, other):
        if single_particle_transition_matrix is None:
            return other
        l_interface = qml.math.get_interface(single_particle_transition_matrix)
        other_interface = qml.math.get_interface(other)
        l_priority = cls.casting_priorities.index(l_interface)
        other_priority = cls.casting_priorities.index(other_interface)
        if l_priority < other_priority:
            single_particle_transition_matrix = utils.math.convert_and_cast_like(
                single_particle_transition_matrix, other
            )
        elif l_priority > other_priority:
            other = utils.math.convert_and_cast_like(
                other, single_particle_transition_matrix
            )
        single_particle_transition_matrix = qml.math.einsum(
            "...ij,...jl->...il",
            single_particle_transition_matrix, other
        )
        return single_particle_transition_matrix

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
            r_dtype=float,
            c_dtype=complex,
            analytic=None,
            shots: Optional[int] = None,
            **kwargs
    ):
        if np.isscalar(wires):
            assert wires > 1, "At least two wires are required for this device."
        else:
            assert len(wires) > 1, "At least two wires are required for this device."
        super().__init__(wires=wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)

        self._debugger = kwargs.get("debugger", None)

        # create the initial state
        self._basis_state_index = 0
        self._sparse_state = None
        self._pre_rotated_sparse_state = None
        self._state = None
        self._pre_rotated_state = None

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
        self.pfaffian_method = kwargs.get("pfaffian_method", self.DEFAULT_PFAFFIAN_METHOD)
        assert self.pfaffian_method in self.pfaffian_methods, (
            f"The pfaffian method must be one of {self.pfaffian_methods}. "
            f"Got {self.pfaffian_method} instead."
        )
        self.sampling_strategy: SamplingStrategy = get_sampling_strategy(
            kwargs.get("sampling_strategy", self.DEFAULT_SAMPLING_STRATEGY)
        )
        self.prob_strategy: ProbabilityStrategy = get_probability_strategy(
            kwargs.get("prob_strategy", self.DEFAULT_PROB_STRATEGY)
        )
        self.contraction_strategy: ContractionStrategy = get_contraction_strategy(
            kwargs.get("contraction_strategy", self.DEFAULT_CONTRACTION_METHOD)
        )
        self.n_workers = kwargs.get("n_workers", 0)
        self.p_bar: Optional[tqdm.tqdm] = kwargs.get("p_bar", None)
        self.show_progress = kwargs.get("show_progress", self.p_bar is not None)
        self.apply_metadata = defaultdict()
    
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
    
    def get_sparse_or_dense_state(self) -> Union[int, sparse.coo_array, np.ndarray, TensorLike]:
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

    def batch_transform(self, circuit: QuantumTape):
        if len(circuit.observables) == 1:
            if isinstance(circuit.observables[0], BatchHamiltonian):
                circuits = [circuit]

                def hamiltonian_fn(res):
                    return res[0]

                if circuit.batch_size is None or self.capabilities().get("supports_broadcasting"):
                    return circuits, hamiltonian_fn
        return super().batch_transform(circuit)

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

        if kwargs.get("index", 0) > 0 and isinstance(operation, (qml.StatePrep, qml.BasisState)):
            raise qml.DeviceError(
                f"Operation {operation.name} cannot be used after other Operations have already been applied "
                f"on a {self.short_name} device."
            )

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
            return operation.pad(self.wires).matrix()

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
        n_processes = self.n_workers
        if self.n_workers == -1:
            n_processes = psutil.cpu_count(logical=True)
        elif self.n_workers == -2:
            n_processes = psutil.cpu_count(logical=False)
        elif self.n_workers < 0:
            raise ValueError("The number of workers must be greater or equal than 0 or in [0, -2].")

        if n_processes == 0 or n_processes == 1:
            return self.gather_single_particle_transition_matrices(operations)

        op_indices_splits = np.array_split(range(len(operations)), n_processes)
        op_splits = [
            [operations[i] for i in op_indices_split]
            for op_indices_split in op_indices_splits
            if len(op_indices_split) > 0
        ]
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
        global_single_particle_transition_matrix = None
        batched = False
        self.apply_metadata["n_operations"] = len(operations)
        operations = self.contraction_strategy(operations, p_bar=self.p_bar, show_progress=self.show_progress)
        self.apply_metadata["n_contracted_operations"] = len(operations)
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
                op_r = op.pad(self.wires).matrix()
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
                global_single_particle_transition_matrix = self.update_single_particle_transition_matrix(
                    global_single_particle_transition_matrix, op_r
                )
            self.p_bar_set_n(i + 1)

        if global_single_particle_transition_matrix is None:
            global_single_particle_transition_matrix = pnp.eye(2 * self.num_wires)[None, ...]
            if not batched:
                global_single_particle_transition_matrix = global_single_particle_transition_matrix[0]
        # store the pre-rotated state
        self._pre_rotated_sparse_state = self._sparse_state
        self._pre_rotated_state = self._state

        assert rotations is None or np.asarray([rotations]).size == 0, "Rotations are not supported"
        # apply the circuit rotations
        # for operation in rotations:
        #     self._state = self._apply_operation(self._state, operation)
        self.p_bar_set_postfix_str("Computing transition matrix")
        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(
            global_single_particle_transition_matrix
        )
        self.p_bar_set_postfix_str("Transition matrix computed")
        self.close_p_bar()

    def _apply_op(
            self, _op: qml.operation.Operation, _batched: bool, _global_sptm: TensorLike
    ) -> Tuple[TensorLike, bool]:
        _op_r = SingleParticleTransitionMatrixOperation.from_operation(_op).pad(self.wires).matrix()
        _batched = _batched or (qml.math.ndim(_op_r) > 2)
        return self.update_single_particle_transition_matrix(_global_sptm, _op_r), _batched

    def apply_generator(self, op_iterator: Iterable[qml.operation.Operation], **kwargs):
        """
        Apply a generator of gates to the device.

        :param op_iterator: The generator of operations to apply
        :type op_iterator: Iterable[qml.operation.Operation]
        :param kwargs: Additional keyword arguments
        :return: None
        """
        global_sptm, batched = None, False
        n_ops = kwargs.get("n_ops", getattr(op_iterator, "__len__", lambda: None)())
        total = n_ops or 0
        self.initialize_p_bar(total=total, desc="Applying operations", unit="op")

        for i, op in enumerate(op_iterator):
            self.apply_metadata["n_operations"] = i + 1
            self.p_bar_set_total(max(i + 1, total))
            if isinstance(op, qml.Identity):
                continue
            is_prep = self.apply_state_prep(op, index=i)
            if is_prep:
                continue
            op_list = self.contraction_strategy.get_next_operations(op)
            for j, op_j in enumerate(op_list):
                if op_j is None:
                    continue
                global_sptm, batched = self._apply_op(op_j, batched, global_sptm)
                self.apply_metadata["n_contracted_operations"] = (
                        self.apply_metadata.get("n_contracted_operations", 0) + 1
                )
                self.apply_metadata["percentage_contracted"] = (
                        100 * (i + 1 - self.apply_metadata["n_contracted_operations"]) / (i + 1)
                )
            self.p_bar_set_n(i + 1)
            self.p_bar_set_postfix_str(f"Compression: {self.apply_metadata.get('percentage_contracted', 0):.2f}%")
            if kwargs.get("gc_op", True):
                del op

        self.p_bar_set_total(self.apply_metadata["n_operations"])
        last_op = self.contraction_strategy.get_reminding()
        if last_op is not None:
            global_sptm, batched = self._apply_op(last_op, batched, global_sptm)
            self.apply_metadata["n_contracted_operations"] = self.apply_metadata.get("n_contracted_operations", 0) + 1
        self.apply_metadata["percentage_contracted"] = (
                100 * (
                    self.apply_metadata["n_operations"] - self.apply_metadata.get("n_contracted_operations", 0)
                ) / self.apply_metadata["n_operations"]
        )
        if global_sptm is None:
            global_sptm = pnp.eye(2 * self.num_wires)[None, ...]
            if not batched:
                global_sptm = global_sptm[0]

        self.p_bar_set_n(self.apply_metadata["n_operations"])
        self.p_bar_set_postfix_str("Computing transition matrix")
        self._transition_matrix = utils.make_transition_matrix_from_action_matrix(global_sptm)
        self.p_bar_set_postfix_str(
            f"Transition matrix computed. Compression: {self.apply_metadata.get('percentage_contracted', 0):.2f}%"
        )
        self.close_p_bar()
        return

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

        operations = self.contraction_strategy(operations, p_bar=self.p_bar, show_progress=self.show_progress)

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

    def get_state_probability(self, target_binary_state: TensorLike, wires: Optional[Wires] = None):
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires

        wires = Wires(wires)
        num_wires = len(wires)
        if isinstance(target_binary_state, int):
            target_binary_state = utils.binary_string_to_vector(
                utils.state_to_binary_string(target_binary_state, num_wires)
            )
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
        return self.prob_strategy(
            system_state=self.get_sparse_or_dense_state(),
            target_binary_state=target_binary_state,
            wires=wires,
            all_wires=self.wires,
            lookup_table=self.lookup_table,
            transition_matrix=self.transition_matrix,
            pfaffian_method=self.pfaffian_method,
            majorana_getter=self.majorana_getter,
            show_progress=self.show_progress,
        )

    def get_states_probability(
            self,
            target_binary_states: TensorLike,
            batch_wires: Optional[Wires] = None,
            **kwargs
    ):
        if not self.is_state_initialized:
            return None
        if batch_wires is None:
            batch_wires = [self.wires]

        batch_wires = np.asarray(batch_wires)
        wires_shape = batch_wires.shape
        if isinstance(target_binary_states, int):
            target_binary_state = utils.binary_string_to_vector(
                utils.state_to_binary_string(target_binary_states, wires_shape[-1])
            )
        elif isinstance(target_binary_states, list):
            target_binary_state = np.array(target_binary_states)
        elif isinstance(target_binary_states, str):
            target_binary_state = utils.binary_string_to_vector(target_binary_states)
        else:
            target_binary_state = np.asarray(target_binary_states)

        if len(target_binary_state.shape) == len(wires_shape) + 1:
            batch_wires = np.stack([batch_wires for _ in range(target_binary_state.shape[0])])
            wires_shape = batch_wires.shape

        assert target_binary_state.shape == wires_shape, (
            f"The target binary states must have the shape {wires_shape}. "
            f"Got {target_binary_state.shape} instead."
        )
        return self.prob_strategy.batch_call(
            system_state=self.get_sparse_or_dense_state(),
            target_binary_states=target_binary_states,
            batch_wires=batch_wires,
            all_wires=self.wires,
            lookup_table=self.lookup_table,
            transition_matrix=self.transition_matrix,
            pfaffian_method=self.pfaffian_method,
            majorana_getter=self.majorana_getter,
            show_progress=kwargs.pop("show_progress", self.show_progress),
            nb_workers=kwargs.pop("nb_workers", self.n_workers),
            **kwargs
        )

    def analytic_probability(self, wires=None):
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        wires_shape = wires.toarray().shape
        wires_binary_states = np.array(list(itertools.product([0, 1], repeat=wires_shape[-1])))

        if len(wires.toarray().shape) == 2:
            wires_batched = np.stack([wires.toarray() for _ in range(wires_binary_states.shape[-2])], axis=-2)
            wires_binary_states = np.stack([wires_binary_states for _ in range(wires_shape[0])])
            probs = self.get_states_probability(
                wires_binary_states.reshape(-1, wires_shape[-1]),
                wires_batched.reshape(-1, wires_shape[-1])
            )
            probs = probs.reshape(wires_shape[0], -1)
            probs = probs / qml.math.sum(probs, -1).reshape(-1, 1)
            return probs
        elif len(wires_shape) > 2:
            raise ValueError(f"The wires must be a 1D or 2D array. Got a {len(wires_shape)}D array instead.")

        probs = qml.math.stack([
            self.get_state_probability(wires_binary_state, wires)
            for wires_binary_state in wires_binary_states
        ])
        probs = probs / qml.math.sum(probs)
        # probs = self.get_states_probability(wires_binary_states, wires)
        return probs

    def generate_samples(self):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        if not self.is_state_initialized:
            return None
        # return self.sampling_strategy.generate_samples(self, self.get_state_probability)
        return self.sampling_strategy.batch_generate_samples(
            self,
            partial(self.get_states_probability, show_progress=False),
            nb_workers=self.n_workers,
            show_progress=self.show_progress
        )

    def expval(self, observable, shot_range=None, bin_size=None):
        if isinstance(observable, BasisStateProjector):
            return self.get_state_probability(observable.parameters[0], observable.wires)
        elif isinstance(observable, qml.Identity):
            t_shape = qml.math.shape(self.transition_matrix)
            if len(t_shape) == 2:
                return convert_and_cast_like(1, self.transition_matrix)
            return convert_and_cast_like(np.ones(t_shape[0]), self.transition_matrix)

        super_re = super().expval(observable, shot_range, bin_size)
        if isinstance(observable, BatchHamiltonian):
            return observable.reduce(super_re)
        return super_re

    def execute_generator(
            self,
            op_iterator: Iterable[qml.operation.Operation],
            observable: Optional = None,
            output_type: Optional[Literal["samples", "expval", "probs"]] = None,
            **kwargs
    ):
        if kwargs.get("reset", True):
            self.reset()
        self.apply_generator(op_iterator, **kwargs)
        if output_type is None:
            return
        if self.shots is not None and self._samples is None:
            self._samples = self.generate_samples()
        if output_type == "samples":
            if self.shots is None:
                raise ValueError("The number of shots must be specified to generate samples.")
            return self._samples
        if output_type == "expval":
            return self.expval(observable)
        if output_type == "probs":
            return self.probability(**kwargs)
        raise ValueError(f"Output type {output_type} is not supported.")

    def _asarray(self, x, dtype=None):
        r"""
        Convert the input to an array of type ``dtype``.

        :Note: If the input is on cuda, it will be copied to cpu.

        :param x: input to be converted
        :param dtype: type of the output array
        :return: array of type ``dtype``
        """
        return qml.math.cast(torch_utils.to_cpu(x), dtype=dtype)

    def _dot(self, a, b):
        r"""
        Compute the dot product of two arrays.

        :param a: input array
        :param b: input array
        :return: dot product of the input arrays
        """
        return qml.math.einsum("...i,...i->...", self._asarray(a), self._asarray(b))

    def reset(self):
        """Reset the device"""
        self._basis_state_index = 0
        self._sparse_state = None
        self._pre_rotated_sparse_state = self._sparse_state
        self._state = None
        self._pre_rotated_state = self._state
        self._transition_matrix = None
        self._lookup_table = None
        self.apply_metadata = defaultdict()

    def update_p_bar(self, *args, **kwargs):
        if self.p_bar is None:
            return
        self.p_bar.update(*args, **kwargs)
        self.p_bar.refresh()

    def p_bar_set_n(self, n: int):
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def p_bar_set_total(self, total: int):
        if self.p_bar is not None:
            self.p_bar.total = total
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
