from collections import defaultdict
from functools import partial
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import torch
import tqdm

from ..typing import TensorLike
from .expval_strategies.clifford_expval.clifford_expval_strategy import (
    CliffordExpvalStrategy,
)
from .expval_strategies.expval_from_probabilities import ExpvalFromProbabilitiesStrategy

try:
    from pennylane.ops import Hamiltonian
except ImportError:
    from pennylane.ops.op_math.linear_combination import Hamiltonian

import pennylane as qml
from pennylane.devices._legacy_device import _local_tape_expand
from pennylane.measurements import (
    ShadowExpvalMP,
)
from pennylane.operation import Operation, StatePrepBase
from pennylane.ops import LinearCombination, Prod, SProd, Sum
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires

from .. import __version__, utils
from ..base.lookup_table import NonInteractingFermionicLookupTable
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..observables.batch_projector import BatchProjector
from ..operations.matchgate_operation import MatchgateOperation
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..utils import (
    binary_string_to_vector,
    torch_utils,
)
from ..utils.math import (
    convert_and_cast_like,
    fermionic_operator_matmul,
)
from .contraction_strategies import ContractionStrategy, get_contraction_strategy
from .expval_strategies.terms_splitter import TermsSplitter
from .probability_strategies import ProbabilityStrategy, get_probability_strategy
from .sampling_strategies import SamplingStrategy, get_sampling_strategy
from .star_state_finding_strategies import (
    StarStateFindingStrategy,
    get_star_state_finding_strategy,
)


class NonInteractingFermionicDevice(qml.devices.QubitDevice):
    r"""
    The Non-Interacting Fermionic Simulator device. This device simulates non-interacting fermions using
    matchgate operations and single particle transition matrices.

    The initial state of the device is the Z-zero state :math:`|0\rangle_{Z}^{\otimes N}` unless a state preparation
    operation is applied at the beginning of the circuit.

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
    :keyword star_state_finding_strategy: The strategy to find the star state.
    :type star_state_finding_strategy: Union[str, StarStateFindingStrategy]

    :Note: This device is a simulator for non-interacting fermions. It is based on the ``default.qubit`` device.
    :Note: This device supports batch execution.
    :Note: This device is in development, and its API is subject to change.

    TODO: Remove all kind of state storage and only use the `state_prep_op`. Then add properties like: `binary_state`, `initial_product_state`.
    TODO: Dynamically update the wires when applying operations on different wires -> `wires` input becomes optional.
    """

    name = "Non-Interacting Fermionic Simulator"
    short_name = "nif.qubit"
    pennylane_requires = "==0.39"
    version = __version__
    author = "Jérémie Gince"

    operations = {
        MatchgateOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(MatchgateOperation)],
        SingleParticleTransitionMatrixOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(SingleParticleTransitionMatrixOperation)],
        "BasisEmbedding",
        "StatePrep",
        "BasisState",
        "Snapshot",
    }
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
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
    DEFAULT_SAMPLING_STRATEGY = "2QubitBy2QubitSampling"
    DEFAULT_STAR_STATE_FINDING_STRATEGY = "FromSampling"
    pfaffian_methods = {"det", "cuda_det", "PfaffianFDBPf"}
    DEFAULT_PFAFFIAN_METHOD = "det"

    casting_priorities = [
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ]  # greater index means higher priority

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
    def update_single_particle_transition_matrix(
        cls,
        old_sptm: Optional[Union[SingleParticleTransitionMatrixOperation, TensorLike]],
        new_sptm: Union[SingleParticleTransitionMatrixOperation, TensorLike],
    ) -> TensorLike:
        """
        Update the old single particle transition matrix by performing a matrix multiplication with the new single
        particle transition matrix.
        :param old_sptm: The old single particle transition matrix
        :param new_sptm: The new single particle transition matrix

        :return: The updated single particle transition matrix.
        """
        if isinstance(new_sptm, Operation):
            new_sptm = new_sptm.matrix()
        if old_sptm is None:
            return new_sptm
        if isinstance(old_sptm, Operation):
            old_sptm = old_sptm.matrix()
        old_interface = qml.math.get_interface(old_sptm)
        new_interface = qml.math.get_interface(new_sptm)
        old_priority = cls.casting_priorities.index(old_interface)
        new_priority = cls.casting_priorities.index(new_interface)
        if old_priority < new_priority:
            old_sptm = utils.math.convert_and_cast_like(old_sptm, new_sptm)
        elif old_priority > new_priority:
            new_sptm = utils.math.convert_and_cast_like(new_sptm, old_sptm)
        return fermionic_operator_matmul(old_sptm, new_sptm, operator="einsum")

    def __init__(
        self,
        wires: Union[int, Wires, List[int]] = 2,
        *,
        r_dtype=torch.float64,
        c_dtype=torch.complex128,
        analytic=None,
        shots: Optional[int] = None,
        **kwargs,
    ):
        if np.isscalar(wires):
            assert wires > 1, "At least two wires are required for this device."
        else:
            assert len(wires) > 1, "At least two wires are required for this device."
        super().__init__(
            wires=wires,
            shots=shots,
            r_dtype=r_dtype,
            c_dtype=c_dtype,
            analytic=analytic,
        )
        self._debugger = kwargs.get("debugger", None)

        # create the initial state
        self._basis_state_index = 0
        self._binary_state = None
        self._pre_rotated_sparse_state = None
        self._pre_rotated_state = None

        self._star_state = None
        self._star_probability = None

        self._transition_matrix = None
        self._lookup_table = None
        self._global_sptm = None
        self._batched = False

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
            f"The pfaffian method must be one of {self.pfaffian_methods}. " f"Got {self.pfaffian_method} instead."
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
        self.star_state_finding_strategy: StarStateFindingStrategy = get_star_state_finding_strategy(
            kwargs.get(
                "star_state_finding_strategy",
                self.DEFAULT_STAR_STATE_FINDING_STRATEGY,
            )
        )
        self.p_bar: Optional[tqdm.tqdm] = kwargs.get("p_bar", None)
        self.show_progress = kwargs.get("show_progress", self.p_bar is not None)
        self.apply_metadata = defaultdict()
        self.clifford_expval_strategy = CliffordExpvalStrategy()
        self.expval_from_probabilities_strategy = ExpvalFromProbabilitiesStrategy()
        self._state_prep_op = qml.BasisState(np.zeros(self.num_wires, dtype=int), wires=self.wires)

    def apply(self, operations, rotations=None, **kwargs):
        """
        This method applies a list of operations to the device. It will update the ``_transition_matrix`` attribute
        of the device.

        :Note: if the number of workers is different from 0, this method will use multiprocessing method
            :py:meth:`apply_mp` to apply the operations.

        :param operations: The operations to apply
        :param rotations: The rotations to apply
        :param kwargs: Additional keyword arguments.
        :return: None
        """
        if not isinstance(operations, Iterable):
            operations = [operations]
        kwargs.setdefault("n_ops", len(operations))
        return self.apply_generator(iter(operations), rotations=rotations, **kwargs)

    def execute_generator(
        self,
        op_iterator: Iterable[qml.operation.Operation],
        observable: Optional = None,
        output_type: Optional[Literal["samples", "expval", "probs", "star_state", "*state"]] = None,
        **kwargs,
    ):
        """
        Execute a generator of operations on the device and return the result in the specified output type.

        :param op_iterator: A generator of operations to execute
        :type op_iterator: Iterable[qml.operation.Operation]
        :param observable: The observable to measure
        :type observable: Optional
        :param output_type: The type of output to return. Supported types are "samples", "expval", and "probs"
        :type output_type: Optional[Literal["samples", "expval", "probs"]]
        :param kwargs: Additional keyword arguments

        :keyword reset: Whether to reset the device before applying the operations. Default is False.
        :keyword apply: Whether to apply the operations. Where "auto" will apply the operations if the transition matrix
            is None which means that no operations have been applied yet. Default is "auto".
        :keyword wires: The wires to measure the observable on. Default is None.
        :keyword shot_range: The range of shots to measure the observable on. Default is None.
        :keyword bin_size: The size of the bins to measure the observable on. Default is None.

        :return: The result of the execution in the specified output type
        """
        if kwargs.get("reset", False):
            self.reset()
        apply = kwargs.get("apply", "auto")
        if apply == "auto" and self._global_sptm is None:
            apply = True
        elif apply == "auto":
            apply = False
        if apply:
            self.apply_generator(op_iterator, **kwargs)
        return self.execute_output(observable=observable, output_type=output_type, **kwargs)

    def execute_output(
        self,
        observable: Optional = None,
        output_type: Optional[Literal["samples", "expval", "probs", "star_state", "*state"]] = None,
        **kwargs,
    ):
        """
        Return the result of the execution in the specified output type.

        :param observable: The observable to measure
        :type observable: Optional
        :param output_type: The type of output to return. Supported types are "samples", "expval", and "probs"
        :type output_type: Optional[Literal["samples", "expval", "probs"]]
        :param kwargs: Additional keyword arguments

        :keyword reset: Whether to reset the device before applying the operations. Default is False.
        :keyword apply: Whether to apply the operations. Where "auto" will apply the operations if the transition matrix
            is None which means that no operations have been applied yet. Default is "auto".
        :keyword wires: The wires to measure the observable on. Default is None.
        :keyword shot_range: The range of shots to measure the observable on. Default is None.
        :keyword bin_size: The size of the bins to measure the observable on. Default is None.

        :return: The result of the execution in the specified output type
        """
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
            return self.probability(
                wires=kwargs.get("wires", None),
                shot_range=kwargs.get("shot_range", None),
                bin_size=kwargs.get("bin_size", None),
            )
        if output_type in ["star_state", "*state"]:
            return self.compute_star_state(**kwargs)
        raise ValueError(f"Output type {output_type} is not supported.")

    def apply_generator(
        self, op_iterator: Iterable[qml.operation.Operation], **kwargs
    ) -> "NonInteractingFermionicDevice":
        """
        Apply a generator of gates to the device.

        :param op_iterator: The generator of operations to apply
        :type op_iterator: Iterable[qml.operation.Operation]
        :param kwargs: Additional keyword arguments
        :return: None
        """
        n_ops = kwargs.get("n_ops", getattr(op_iterator, "__len__", lambda: None)())
        total = n_ops or 0
        self.initialize_p_bar(total=total, desc="Applying operations", unit="op")
        self.apply_metadata["n_operations"] = self.apply_metadata.get("n_operations", 0)
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
                self.apply_op(op_j)
                self.apply_metadata["n_contracted_operations"] = (
                    self.apply_metadata.get("n_contracted_operations", 0) + 1
                )
                self.apply_metadata["percentage_contracted"] = (
                    100 * (i + 1 - self.apply_metadata["n_contracted_operations"]) / (i + 1)
                )
            self.p_bar_set_n(i + 1)
            self.p_bar_set_postfix_str(f"Compression: {self.apply_metadata.get('percentage_contracted', 0):.2f}%")
            if kwargs.get("gc_op", False):
                del op

        self.p_bar_set_total(self.apply_metadata["n_operations"])
        last_op = self.contraction_strategy.get_reminding()
        if last_op is not None:
            self.apply_op(last_op)
            self.apply_metadata["n_contracted_operations"] = self.apply_metadata.get("n_contracted_operations", 0) + 1
        self.apply_metadata["percentage_contracted"] = (
            100
            * (self.apply_metadata["n_operations"] - self.apply_metadata.get("n_contracted_operations", 0))
            / max(self.apply_metadata["n_operations"], 1)
        )

        self.p_bar_set_n(self.apply_metadata["n_operations"])
        self.p_bar_set_postfix_str("Computing transition matrix")
        if kwargs.get("cache_global_sptm", False):
            self.apply_metadata["global_sptm"] = torch_utils.to_numpy(self.global_sptm.matrix())
        self.p_bar_set_postfix_str(
            f"Global Sptm matrix computed. Compression: {self.apply_metadata.get('percentage_contracted', 0):.2f}%"
        )
        self.close_p_bar()
        return self

    def apply_op(self, op: qml.operation.Operation) -> SingleParticleTransitionMatrixOperation:
        """
        Applies a given quantum operation to compute and update the single-particle
        transition matrix representation.

        This method processes the input operation, converts it into its corresponding
        single-particle transition matrix operation, and updates the global single-
        particle transition matrix attribute. It also determines if batching is
        enabled by checking the dimensions of the resulting matrix.

        :param op: The quantum operation to be applied, represented as an instance
            of ``qml.operation.Operation``.
        :return: The updated single-particle transition matrix, represented as an
            instance of ``SingleParticleTransitionMatrixOperation``.
        """
        op_sptm = SingleParticleTransitionMatrixOperation.from_operation(op).pad(self.wires).matrix()
        self._batched = self._batched or (qml.math.ndim(op_sptm) > 2)
        self.global_sptm = self.update_single_particle_transition_matrix(self._global_sptm, op_sptm)
        return self.global_sptm

    def compute_star_state(self, **kwargs):
        """
        Compute the star state of the device. The star state is the state that has the highest probability.

        :param kwargs:  Additional keyword arguments
        :return: The star state and its probability
        """
        if self._star_state is None or self._star_probability is None:
            self._star_state, self._star_probability = self.star_state_finding_strategy(
                self,
                partial(self.get_states_probability, show_progress=False),
                show_progress=self.show_progress,
                samples=self._samples,
            )
        return self.star_state, self.star_probability

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

        if kwargs.get("index", 0) > 0 and isinstance(operation, (StatePrepBase, qml.BasisState)):
            raise qml.DeviceError(
                f"Operation {operation.name} cannot be used after other Operations have already been applied "
                f"on a {self.short_name} device."
            )

        is_applied = True
        if isinstance(operation, qml.BasisState):
            self._apply_basis_state(operation.parameters[0], operation.wires)
            self._state_prep_op = operation
        elif isinstance(operation, qml.StatePrep):
            self._state_prep_op = operation
        elif isinstance(operation, StatePrepBase):
            self._state_prep_op = operation
        elif isinstance(operation, qml.Snapshot):
            raise NotImplementedError("The Snapshot operation is not implemented yet.")
        elif isinstance(operation, qml.pulse.ParametrizedEvolution):
            raise NotImplementedError("The Parametrized Evolution operation is not implemented yet.")
        else:
            is_applied = False
        return is_applied

    def get_state_probability(self, target_binary_state: TensorLike, wires: Optional[Wires] = None):
        """
        Calculates the probability of the system being in a specific binary state for the given wires.

        This method determines how likely the system's quantum state corresponds to the provided
        target binary state. The functionality supports different formats for the input state:
        integer, list, string, or a tensor-like object. It also processes the specified wires to
        pinpoint the relevant parts of the quantum state and computes the probability using
        internally defined strategies.

        :param target_binary_state: The desired binary state of the system to calculate the probability for. \
        Can be provided as an integer, list, string, or tensor-like object.
        :type target_binary_state: TensorLike
        :param wires: Optional argument specifying which wires to consider when computing the probability. \
        If not provided, defaults to all wires in the system.
        :type wires: Optional[Wires]
        :return: Probability associated with the target binary state on the given wires. Returns `None` if \
        the state has not been initialized.
        :rtype: float or None
        """
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
            f"The target binary state must have {num_wires} elements. " f"Got {len(target_binary_state)} instead."
        )
        return self.prob_strategy(
            system_state=self.binary_state,
            target_binary_state=target_binary_state,
            wires=wires,
            all_wires=self.wires,
            lookup_table=self.lookup_table,
            transition_matrix=self.transition_matrix,
            global_sptm=self.global_sptm,
            pfaffian_method=self.pfaffian_method,
            majorana_getter=self.majorana_getter,
            show_progress=self.show_progress,
        )

    def get_states_probability(
        self,
        target_binary_states: TensorLike,
        batch_wires: Optional[Wires] = None,
        **kwargs,
    ):
        """
        Calculates the probability of the provided binary states with respect to the
        current system state. This function supports various input formats for the
        target binary states, including integers, strings, lists, and NumPy arrays,
        and performs necessary conversions to ensure compatibility. It handles batch
        processing of wires in conjunction with target states to compute the resulting
        probabilities. If the system state is uninitialized, the function returns None.

        :param target_binary_states: Tensor-like object representing the target binary
            states. May be an integer, string, list, or NumPy array. If provided as an
            integer or string, it is converted into a vectorized binary format.
        :param batch_wires: Optional; defines the wires to process in batch. Defaults
            to the current set of system wires. Expected to match the shape of
            `target_binary_states`.
        :param kwargs: Additional keyword arguments for fine-tuning the probability
            computation process. Includes parameters like `show_progress`, which
            controls the display of progress feedback, and others related to the
            internal state representation and computation strategies.
        :return: If successful, returns probabilities of the provided binary states
            as computed by the configured probability strategy. In case the system
            state is not initialized, returns None.
        """
        if not self.is_state_initialized:
            return None

        if isinstance(target_binary_states, int):
            target_binary_states = utils.binary_string_to_vector(
                utils.state_to_binary_string(target_binary_states, self.num_wires)
            )
        elif isinstance(target_binary_states, list):
            target_binary_states = np.array(target_binary_states)
        elif isinstance(target_binary_states, str):
            target_binary_states = utils.binary_string_to_vector(target_binary_states)
        else:
            target_binary_states = np.asarray(target_binary_states)

        if batch_wires is None:
            batch_wires = self.wires
        batch_wires = np.asarray(batch_wires)
        batch_wires = np.broadcast_to(batch_wires, target_binary_states.shape)

        assert target_binary_states.shape == batch_wires.shape, (
            f"The target binary states must have the shape {batch_wires.shape}. "
            f"Got {target_binary_states.shape} instead."
        )
        return self.prob_strategy.batch_call(
            system_state=self.binary_state,
            target_binary_states=target_binary_states,
            batch_wires=batch_wires,
            all_wires=self.wires,
            lookup_table=self.lookup_table,
            transition_matrix=self.transition_matrix,
            global_sptm=self.global_sptm.matrix(),
            state_prep_op=self._state_prep_op,
            pfaffian_method=self.pfaffian_method,
            majorana_getter=self.majorana_getter,
            show_progress=kwargs.pop("show_progress", self.show_progress),
            **kwargs,
        )

    def analytic_probability(self, wires=None):
        r"""Return the (marginal) probability of each computational basis
        state from the last run of the device.

        PennyLane uses the convention
        :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where :math:`q_0` is the most
        significant bit.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.


        .. note::

            :meth:`~.marginal_prob` may be used as a utility method
            to calculate the marginal probability distribution.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        if not self.is_state_initialized:
            return None
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires, _override=True)
        wires_array = wires.toarray()
        wires_shape = wires_array.shape
        wires_binary_states = self.states_to_binary(np.arange(2 ** wires_shape[-1]), wires_shape[-1])

        if len(wires_shape) == 2:
            wires_batched = np.stack([wires_array for _ in range(wires_binary_states.shape[-2])], axis=-2)
            wires_binary_states = np.stack([wires_binary_states for _ in range(wires_shape[0])])
            probs = self.get_states_probability(
                wires_binary_states.reshape(-1, wires_shape[-1]),
                wires_batched.reshape(-1, wires_shape[-1]),
            )
            probs = qml.math.transpose(probs.reshape(*wires_binary_states.shape[:-1], -1), (-1, 0, 1))
            probs = probs / qml.math.sum(probs, -1)[..., np.newaxis]
            return probs
        elif len(wires_shape) > 2:
            raise ValueError(f"The wires must be a 1D or 2D array. Got a {len(wires_shape)}D array instead.")

        probs = self.get_states_probability(wires_binary_states, wires)
        probs = qml.math.transpose(probs)
        probs = probs / qml.math.sum(probs)
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
        self._samples = self.sampling_strategy.batch_generate_samples(
            self,
            partial(self.get_states_probability, show_progress=False),
            show_progress=self.show_progress,
        )
        return self.samples

    def exact_expval(self, observable):
        """
        Computes the expectation value of a given observable.

        This method evaluates the expectation value of the provided observable by using
        appropriate execution strategies or by splitting its terms. If the observable
        cannot be computed using any of the defined strategies, an error is raised.

        :param observable: The observable for which the expectation value is to be
            computed. It must be of a compatible type such as BasisStateProjector
            or BatchProjector, or match the supported execution strategies.
        :type observable: object
        :return: The computed expectation value of the given observable.
        :rtype: float
        :raises qml.DeviceError: If the expectation value of the observable cannot
            be computed on the current device due to compatibility issues.
        """
        if isinstance(observable, BasisStateProjector):
            return self.get_state_probability(observable.parameters[0], observable.wires)
        if isinstance(observable, BatchProjector):
            return self.get_states_probability(observable.get_states(), observable.get_batch_wires())
        if self.clifford_expval_strategy.can_execute(self.state_prep_op, observable):
            return self.clifford_expval_strategy(self.state_prep_op, observable, global_sptm=self.global_sptm.matrix())
        if self.expval_from_probabilities_strategy.can_execute(self.state_prep_op, observable):
            return self.expval_from_probabilities_strategy(self.state_prep_op, observable, prob_func=self.probability)
        terms_splitter = TermsSplitter([self.clifford_expval_strategy, self.expval_from_probabilities_strategy])
        if terms_splitter.can_execute(self.state_prep_op, observable):
            return terms_splitter(
                self.state_prep_op, observable, global_sptm=self.global_sptm.matrix(), prob_func=self.probability
            )

        raise qml.DeviceError(
            f"The expectation value of the observable {observable} "
            f"cannot be computed on a {self.short_name} device."
            "Please check the device's capabilities."
        )

    def expval(self, observable, shot_range=None, bin_size=None):
        """
        Calculates the expectation value of a given observable.

        This method computes the expectation value of the provided observable. If
        the `shots` attribute is set to None, it uses an exact method to calculate
        the expectation value. Otherwise, it leverages the parent class method
        with the specified shot range and bin size.

        :param observable: Observable for which the expectation value is being calculated.
        :param shot_range: Tuple specifying the range of shots to consider.
        :param bin_size: Integer specifying the number of shots per bin.
        :return: The computed expectation value.
        """
        if self.shots is None:
            output = self.exact_expval(observable)
        else:
            output = super().expval(observable, shot_range, bin_size)
        return output

    def batch_transform(self, circuit: QuantumTape):
        if len(circuit.observables) == 1:
            if isinstance(circuit.observables[0], BatchHamiltonian):
                circuits = [circuit]

                def hamiltonian_fn(res):
                    return res[0]

                if self.capabilities().get("supports_broadcasting"):
                    return circuits, hamiltonian_fn
                if circuit.batch_size is None:
                    return circuits, hamiltonian_fn
        return self._patched_super_batch_transform(circuit)

    def default_expand_fn(self, circuit: QuantumTape, max_expansion=10):
        ops_not_supported = not all(self.stopping_condition(op) for op in circuit.operations)
        if ops_not_supported:
            circuit = _local_tape_expand(circuit, depth=max_expansion, stop_at=self.stopping_condition)
        return circuit

    def reset(self):
        """Reset the device"""
        super().reset()
        self._binary_state = None
        self._basis_state_index = 0
        self._global_sptm = None
        self._batched = False
        self._transition_matrix = None
        self._lookup_table = None
        self._star_state = None
        self._star_probability = None
        self._samples = None
        self.apply_metadata = defaultdict()
        self.contraction_strategy.reset()

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

    def initialize_p_bar(self, *args, **kwargs) -> Optional[tqdm.tqdm]:
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

    def _asarray(self, x, dtype=None):
        r"""
        Convert the input to an array of type ``dtype``.

        :Note: If the input is on cuda, it will be copied to cpu.

        :param x: input to be converted
        :param dtype: type of the output array
        :return: array of type ``dtype``
        """
        is_complex = qml.math.any(qml.math.iscomplex(x))
        if dtype is None and is_complex:
            dtype = self.C_DTYPE
        elif dtype is None:
            dtype = self.R_DTYPE
        if not is_complex:
            x = qml.math.real(x)
        return qml.math.cast(torch_utils.to_cpu(x, dtype=dtype), dtype=dtype)

    def _dot(self, a, b):
        r"""
        Compute the dot product of two arrays.

        :param a: input array
        :param b: input array
        :return: dot product of the input arrays
        """
        return qml.math.einsum("...i,...i->...", self._asarray(a), self._asarray(b))

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

        self.binary_state = state

    def _patched_super_batch_transform(self, circuit: QuantumScript):
        """Apply a differentiable batch transform for preprocessing a circuit
        prior to execution. This method is called directly by the QNode, and
        should be overwritten if the device requires a transform that
        generates multiple circuits prior to execution.

        By default, this method contains logic for generating multiple
        circuits, one per term, of a circuit that terminates in ``expval(H)``,
        if the underlying device does not support Hamiltonian expectation values,
        or if the device requires finite shots.

        .. warning::

            This method will be tracked by autodifferentiation libraries,
            such as Autograd, JAX, TensorFlow, and Torch. Please make sure
            to use ``qml.math`` for autodiff-agnostic tensor processing
            if required.

        Args:
            circuit (.QuantumTape): the circuit to preprocess

        Returns:
            tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
            the sequence of circuits to be executed, and a post-processing function
            to be applied to the list of evaluated circuit results.
        """

        def null_postprocess(results):
            return results[0]

        finite_shots = self.shots is not None
        has_shadow = any(isinstance(m, ShadowExpvalMP) for m in circuit.measurements)
        is_analytic_or_shadow = not finite_shots or has_shadow
        all_obs_usable = self._all_multi_term_obs_supported(circuit)
        exists_multi_term_obs = any(isinstance(m.obs, (Hamiltonian, Sum, Prod, SProd)) for m in circuit.measurements)
        has_overlapping_wires = len(circuit.obs_sharing_wires) > 0
        single_hamiltonian = len(circuit.measurements) == 1 and isinstance(
            circuit.measurements[0].obs, (Hamiltonian, Sum)
        )
        single_hamiltonian_with_grouping_known = (
            single_hamiltonian and circuit.measurements[0].obs.grouping_indices is not None
        )

        if not getattr(self, "use_grouping", True) and single_hamiltonian and all_obs_usable:
            # Special logic for the braket plugin
            circuits = [circuit]
            processing_fn = null_postprocess

        elif not exists_multi_term_obs and not has_overlapping_wires:
            circuits = [circuit]
            processing_fn = null_postprocess

        elif is_analytic_or_shadow and all_obs_usable and not has_overlapping_wires:
            circuits = [circuit]
            processing_fn = null_postprocess

        elif single_hamiltonian_with_grouping_known:

            # Use qwc grouping if the circuit contains a single measurement of a
            # Hamiltonian/Sum with grouping indices already calculated.
            circuits, processing_fn = qml.transforms.split_non_commuting(circuit, "qwc")

        elif any(isinstance(m.obs, (Hamiltonian, LinearCombination)) for m in circuit.measurements):

            # Otherwise, use wire-based grouping if the circuit contains a Hamiltonian
            # that is potentially very large.
            circuits, processing_fn = qml.transforms.split_non_commuting(circuit, "wires")

        else:
            circuits, processing_fn = qml.transforms.split_non_commuting(circuit)

        ##############################################################################################################
        # ORIGINAL CODE
        # Check whether the circuit was broadcasted and whether broadcasting is supported
        # if circuit.batch_size is None or self.capabilities().get("supports_broadcasting"):
        #     # If the circuit wasn't broadcasted or broadcasting is supported, no action required
        #     return circuits, processing_fn

        # COMMENTS ON THE BUG FROM THE ORIGINAL CODE
        # If the device supports broadcasting, we can return the circuits as is, otherwise
        # The property 'circuit.batch_size' will raise a error if there are multiple batch sizes in the circuits.
        # As an example, if the circuit has a batch size of 1 and 4, the property will raise an error which is
        # not supposed to happen because the device supports broadcasting.
        if self.capabilities().get("supports_broadcasting"):
            return circuits, processing_fn
        if circuit.batch_size is None:
            return circuits, processing_fn
        ##############################################################################################################

        # Expand each of the broadcasted Hamiltonian-expanded circuits
        expanded_tapes, expanded_fn = qml.transforms.broadcast_expand(circuits)

        # Chain the postprocessing functions of the broadcasted-tape expansions and the Hamiltonian
        # expansion. Note that the application order is reversed compared to the expansion order,
        # i.e. while we first applied `split_non_commuting` to the tape, we need to process the
        # results from the broadcast expansion first.
        def total_processing(results):
            return processing_fn(expanded_fn(results))

        return expanded_tapes, total_processing

    @property
    def binary_state(self) -> np.ndarray:
        if self._binary_state is not None:
            return self._binary_state
        elif self._basis_state_index is not None:
            return binary_string_to_vector(np.binary_repr(self._basis_state_index, width=self.num_wires))
        raise ValueError("The state doesn't seem to be initialized.")

    @binary_state.setter
    def binary_state(self, value):
        self._basis_state_index = None
        if isinstance(value, str):
            value = binary_string_to_vector(value)
        else:
            value = np.asarray(value)
        if value.size != self.num_wires:
            raise ValueError(
                f"Got state of length {value.size} while setting binary state of the "
                f"{self.__class__.__name__} device. Expected length of {self.num_wires}."
            )
        self._binary_state = value

    @property
    def state(self) -> np.ndarray:
        """
        NIF device does not support dense state representation.

        :return: state vector of the device
        :rtype: array[complex]

        :Note: This function comes from the ``default.qubit`` device.
        """
        raise NotImplementedError(
            "The NIF device does not support dense state representation. It would require an exponential amount of memory."
        )

    @property
    def is_state_initialized(self) -> bool:
        state_attrs = [
            self._basis_state_index,
            self._binary_state,
        ]
        return any([s is not None for s in state_attrs])

    @property
    def transition_matrix(self):
        if self._transition_matrix is None and self.global_sptm is not None:
            self._transition_matrix = utils.make_transition_matrix_from_action_matrix(self.global_sptm.matrix())
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value):
        if self._transition_matrix is not None and value is not None:
            value = convert_and_cast_like(value, self._transition_matrix)
        self._transition_matrix = value
        self._lookup_table = None

    @property
    def global_sptm(self) -> Optional[SingleParticleTransitionMatrixOperation]:
        if self._global_sptm is None:
            matrix = np.eye(2 * self.num_wires)[None, ...]
            if not self._batched:
                matrix = matrix[0]
            return SingleParticleTransitionMatrixOperation(matrix, wires=self.wires)
        return self._global_sptm

    @global_sptm.setter
    def global_sptm(self, value):
        if isinstance(value, Operation):
            value = SingleParticleTransitionMatrixOperation.from_operation(value)
        if not isinstance(value, SingleParticleTransitionMatrixOperation):
            value = SingleParticleTransitionMatrixOperation(value, wires=self.wires)
        self._global_sptm = value
        self.transition_matrix = None

    @property
    def lookup_table(self) -> NonInteractingFermionicLookupTable:
        if self._lookup_table is None:
            self._lookup_table = NonInteractingFermionicLookupTable(self.transition_matrix)
        return self._lookup_table

    @property
    def star_state(self) -> Optional[TensorLike]:
        return self._star_state

    @property
    def star_probability(self) -> Optional[TensorLike]:
        return self._star_probability

    @property
    def samples(self) -> Optional[TensorLike]:
        return self._samples

    @property
    def state_prep_op(self) -> StatePrepBase:
        return self._state_prep_op
