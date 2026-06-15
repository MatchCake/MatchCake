from collections import defaultdict
from dataclasses import replace as _dataclass_replace
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Union

import numpy as np
import pennylane as qml
import torch
import tqdm
from pennylane import BasisState
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.preprocess import decompose
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP, ProbabilityMP, SampleMP, Shots
from pennylane.operation import Operation, StatePrepBase
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.tape import QuantumScript
from pennylane.transforms.core import CompilePipeline
from pennylane.wires import Wires

from .. import utils
from ..base.lookup_table import NonInteractingFermionicLookupTable
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..observables.batch_projector import BatchProjector
from ..operations.matchgate_operation import MatchgateOperation
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..operations.state_preparation import ProductState, StatePrepFromGates
from ..typing import TensorLike
from ..utils import (
    torch_utils,
)
from ..utils.math import (
    complex_dtype_name_like,
    convert_like_and_cast_to,
    fermionic_operator_matmul,
)
from .contraction_strategies import ContractionStrategy, get_contraction_strategy
from .expval_strategies.clifford_expval.clifford_expval_strategy import (
    CliffordExpvalStrategy,
)
from .expval_strategies.expval_from_probabilities import ExpvalFromProbabilitiesStrategy
from .expval_strategies.m_pfaffian import MPfaffianExpvalStrategy
from .expval_strategies.m_pfaffian._extended_covariance import (
    displacement_vector as _displacement_vector,
)
from .expval_strategies.m_pfaffian._extended_covariance import (
    extended_covariance_matrix as _extended_covariance_matrix,
)
from .expval_strategies.m_pfaffian._extended_covariance import (
    sptm_lift as _sptm_lift,
)
from .expval_strategies.terms_splitter import TermsSplitter
from .probability_strategies import (
    ProbabilityFuncDispatcher,
    ProductStateProbabilityStrategy,
    get_probability_strategy,
)
from .sampling_strategies import SamplingStrategy, get_sampling_strategy
from .star_state_finding_strategies import (
    StarStateFindingStrategy,
    get_star_state_finding_strategy,
)

_UNSET = object()  # sentinel distinguishing "shots not provided" from an explicit ``shots=None``


@qml.transform
def _nif_split_non_commuting(tape: QuantumScript):
    """Pass BatchHamiltonian measurements through unchanged; otherwise delegate to split_non_commuting."""
    if (
        len(tape.measurements) == 1
        and tape.measurements[0].obs is not None
        and isinstance(tape.measurements[0].obs, BatchHamiltonian)
    ):
        return [tape], lambda results: results[0]
    return qml.transforms.split_non_commuting(tape, grouping_strategy="wires")


class NonInteractingFermionicDevice(qml.devices.Device):
    r"""
    The Non-Interacting Fermionic Simulator device. This device simulates non-interacting fermions using
    matchgate operations and single particle transition matrices.

    The initial state of the device is the Z-zero state :math:`|0\rangle_{Z}^{\otimes N}` unless a state preparation
    operation is applied at the beginning of the circuit.

    :param wires: The number of wires of the device
    :type wires: Union[int, Wires, List[int]]

    :kwargs: Additional keyword arguments

    :keyword prob_strategy: The strategy to compute the probabilities. Can be either "lookup_table" or "explicit_sum".
        Defaults to "lookup_table".
    :type prob_strategy: str
    :keyword majorana_getter: The Majorana getter to use. Defaults to a new instance of MajoranaGetter.
    :type majorana_getter: MajoranaGetter
    :keyword contraction_method: The contraction method to use. Can be either None or "neighbours".
        Defaults to None.
    :type contraction_method: Optional[str]
    :keyword n_workers: The number of workers to use for multiprocessing. Defaults to 0.
    :type n_workers: int
    :keyword star_state_finding_strategy: The strategy to find the star state.
    :type star_state_finding_strategy: Union[str, StarStateFindingStrategy]
    :keyword r_dtype: The real floating-point dtype used for real-valued tensors. Defaults to ``torch.float64``.
    :type r_dtype: torch.dtype
    :keyword c_dtype: The complex dtype used for complex-valued tensors throughout the pipeline (single-particle
        transition matrices, lookup table, observables). Defaults to ``torch.complex128`` to preserve maximum
        precision. Set to e.g. ``torch.complex64`` to reduce memory usage and computation cost.
    :type c_dtype: torch.dtype

    :Note: This device is a simulator for non-interacting fermions. It is based on the ``default.qubit`` device.
    :Note: This device supports batch execution.
    :Note: This device is in development, and its API is subject to change.
    """

    name = "nif.qubit"
    _wires: Optional[Wires]

    R_DTYPE = torch.float64
    C_DTYPE = torch.complex128

    _supported_ops = {
        MatchgateOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(MatchgateOperation)],
        SingleParticleTransitionMatrixOperation.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(SingleParticleTransitionMatrixOperation)],
        StatePrepFromGates.__name__,
        *[c.__name__ for c in utils.get_all_subclasses(StatePrepFromGates)],
        "BasisEmbedding",
        "StatePrep",
        "BasisState",
        ProductState.__name__,
    }

    DEFAULT_PROB_STRATEGY = "LookupTable"
    DEFAULT_CONTRACTION_METHOD = "neighbours"
    DEFAULT_SAMPLING_STRATEGY = "2QubitBy2QubitSampling"
    DEFAULT_STAR_STATE_FINDING_STRATEGY = "FromSampling"

    casting_priorities = [
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ]  # greater index means higher priority

    @staticmethod
    def states_to_binary(samples: np.ndarray, num_wires: int, dtype: type = np.int64) -> np.ndarray:
        """Convert an array of integer state indices to binary representation.

        :param samples: Array of integer state indices, shape (...).
        :type samples: np.ndarray
        :param num_wires: Number of wires (bits per state).
        :type num_wires: int
        :param dtype: Output integer dtype.
        :type dtype: type
        :return: Binary state array, shape (..., num_wires).
        :rtype: np.ndarray
        """
        powers_of_two = 1 << np.arange(num_wires, dtype=dtype)
        states_sampled_base_ten = samples[..., None] & powers_of_two
        return (states_sampled_base_ten > 0).astype(dtype)[..., ::-1]

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
        wires: Optional[Union[int, Wires, List[int]]] = None,
        *,
        shots: Optional[int] = None,
        **kwargs,
    ):
        if wires is not None:
            if isinstance(wires, int):
                assert wires > 1, "At least two wires are required for this device."
            else:
                assert len(list(wires)) > 1, "At least two wires are required for this device."
        super().__init__(wires=wires)
        self._shots = Shots(shots)
        self._debugger = kwargs.get("debugger", None)
        self._init_kwargs = kwargs

        self.R_DTYPE = torch_utils.get_torch_dtype(kwargs.get("r_dtype"), type(self).R_DTYPE)
        self.C_DTYPE = torch_utils.get_torch_dtype(kwargs.get("c_dtype"), type(self).C_DTYPE)
        self._c_dtype_name = str(self.C_DTYPE).rsplit(".", 1)[-1]
        self._r_dtype_name = str(self.R_DTYPE).rsplit(".", 1)[-1]

        self._star_state = None
        self._star_probability = None

        self._transition_matrix: Optional[TensorLike] = None
        self._lookup_table: Optional[NonInteractingFermionicLookupTable] = None
        self._global_sptm: Optional[SingleParticleTransitionMatrixOperation] = None
        self._batched = False
        self._samples: Optional[np.ndarray] = None
        self._current_shots: Optional[int] = None

        self.sampling_strategy: SamplingStrategy = get_sampling_strategy(
            kwargs.get("sampling_strategy", self.DEFAULT_SAMPLING_STRATEGY)
        )
        self.prob_dispatcher: ProbabilityFuncDispatcher = ProbabilityFuncDispatcher(
            [
                get_probability_strategy(kwargs.get("prob_strategy", self.DEFAULT_PROB_STRATEGY)),
                ProductStateProbabilityStrategy(),
            ]
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
        self.apply_metadata: defaultdict = defaultdict()
        self.clifford_expval_strategy = CliffordExpvalStrategy()
        self.expval_from_probabilities_strategy = ExpvalFromProbabilitiesStrategy()
        self.m_pfaffian_expval_strategy = MPfaffianExpvalStrategy()

        self.majorana_getter: Optional[utils.MajoranaGetter] = None
        self._state_prep_op: Optional[StatePrepBase] = None
        if self._wires is not None:
            self._setup_wire_dependent_state()

    def apply(
        self,
        operations: Iterable[qml.operation.Operation],
        rotations: Optional[List[qml.operation.Operation]] = None,
        **kwargs,
    ) -> None:
        """
        Apply a list of operations to the device. Updates the ``_transition_matrix`` attribute.

        :param operations: The operations to apply.
        :type operations: Iterable[qml.operation.Operation]
        :param rotations: The rotations to apply.
        :type rotations: Optional[List[qml.operation.Operation]]
        :param kwargs: Additional keyword arguments.
        :return: None
        """
        if not isinstance(operations, Iterable):
            operations = [operations]
        ops_list = list(operations)
        kwargs.setdefault("n_ops", len(ops_list))
        self.apply_generator(iter(ops_list), rotations=rotations, **kwargs)

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
        if self._wires is None:
            ops = list(op_iterator)
            all_wires = Wires.all_wires([op.wires for op in ops if len(op.wires) > 0])
            assert len(all_wires) > 1, "At least two wires are required for this device."
            self._wires = all_wires
            self._setup_wire_dependent_state()
            op_iterator = iter(ops)
        n_ops = kwargs.get("n_ops", getattr(op_iterator, "__len__", lambda: None)())
        total = n_ops or 0
        self.initialize_p_bar(total=total, desc="Applying operations", unit="op")
        self.apply_metadata["n_operations"] = self.apply_metadata.get("n_operations", 0)
        for i, op in enumerate(op_iterator):
            self.apply_metadata["n_operations"] = i + 1
            self.p_bar_set_total(max(i + 1, total))
            if isinstance(op, qml.Identity):
                continue  # pragma: no cover
            is_prep = self.apply_state_prep(op, index=i)
            if is_prep:
                continue
            op = self.convert_op_to_supported(op)
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
                del op  # pragma: no cover

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

    def convert_op_to_supported(self, op: qml.operation.Operation) -> qml.operation.Operation:
        """
        Ensure an operation is supported by the device, converting native matchgates when possible.

        If ``op`` is already a :class:`MatchgateOperation` or a
        :class:`SingleParticleTransitionMatrixOperation` (or a subclass of either), or if it
        already knows how to produce a single-particle transition matrix (it exposes a
        ``to_sptm_operation`` method or a ``single_particle_transition_matrix`` attribute), it is
        returned unchanged and handled downstream by :meth:`apply_op`. Otherwise, the operation's
        matrix is fed to the :class:`MatchgateOperation` constructor, which validates that it is a
        matchgate. If the matrix is a matchgate, the resulting :class:`MatchgateOperation` is
        returned; if not, a :class:`DeviceError` is raised.

        :param op: The operation to check and convert.
        :type op: qml.operation.Operation
        :return: A supported operation equivalent to ``op``.
        :rtype: qml.operation.Operation
        :raises DeviceError: If ``op`` is neither a supported operation nor a matchgate.
        """
        if isinstance(op, (MatchgateOperation, SingleParticleTransitionMatrixOperation)):
            return op
        if hasattr(op, "to_sptm_operation") or hasattr(op, "single_particle_transition_matrix"):
            return op
        try:
            return MatchgateOperation(op.matrix(), wires=op.wires)
        except Exception as error:
            raise DeviceError(
                f"The {self.name} device can only handle operations that are an instance of "
                f"MatchgateOperation or SingleParticleTransitionMatrixOperation. "
                f"The operation {op} is neither and could not be converted to a MatchgateOperation."
            ) from error

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
        op_sptm = qml.math.cast(op_sptm, self._r_dtype_name)
        self._batched = self._batched or (qml.math.ndim(op_sptm) > 2)
        self.global_sptm = self.update_single_particle_transition_matrix(self._global_sptm, op_sptm)
        return self.global_sptm

    def apply_state_prep(self, operation: qml.operation.Operation, **kwargs) -> bool:
        """
        Apply a state preparation operation to the device. Will set the internal state of the device
        only if the operation is a state preparation operation and then return True. Otherwise, it will
        return False.

        :param operation: The operation to apply
        :type operation: qml.operation.Operation
        :param kwargs: Additional keyword arguments
        :return: True if the operation was applied, False otherwise
        :rtype: bool
        """

        if kwargs.get("index", 0) > 0 and isinstance(operation, StatePrepBase):
            raise DeviceError(
                f"Operation {operation.name} cannot be used after other Operations have already been applied "
                f"on a {self.name} device."
            )

        is_applied = True
        if isinstance(operation, BasisState):
            self._state_prep_op = ProductState.from_basis_state(operation)
        elif isinstance(operation, StatePrepBase):
            self._state_prep_op = operation
        else:
            is_applied = False
        return is_applied

    def analytic_probability(self, wires: Optional[Union[Wires, List[int]]] = None) -> TensorLike:
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
        if wires is None:
            wires = self.wires
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires, _override=True)
        wires_array = wires.toarray()  # type: ignore[union-attr]
        wires_shape = wires_array.shape
        wires_binary_states = self.states_to_binary(np.arange(2 ** wires_shape[-1]), wires_shape[-1])

        if len(wires_shape) == 2:
            wires_batched = np.stack([wires_array for _ in range(wires_binary_states.shape[-2])], axis=-2)
            wires_binary_states = np.stack([wires_binary_states for _ in range(wires_shape[0])])
            probs = self.get_states_probability(
                wires_binary_states.reshape(-1, wires_shape[-1]),
                wires_batched.reshape(-1, wires_shape[-1]),
            )
            probs = qml.math.transpose(probs.reshape(*wires_binary_states.shape[:-1], -1), (-1, 0, 1))  # type: ignore[union-attr]
            probs = probs / qml.math.sum(probs, -1)[..., np.newaxis]
            return probs
        elif len(wires_shape) > 2:
            raise ValueError(f"The wires must be a 1D or 2D array. Got a {len(wires_shape)}D array instead.")

        probs = self.get_states_probability(wires_binary_states, wires)
        probs = qml.math.transpose(probs)
        probs = probs / qml.math.sum(probs, axis=-1, keepdims=True)
        return probs

    def close_p_bar(self) -> None:
        """Close the progress bar if one is active.

        :return: None
        """
        if self.p_bar is not None:
            self.p_bar.close()

    def compute_star_state(self, **kwargs) -> tuple:
        """
        Compute the star state of the device. The star state is the state that has the highest probability.

        :param kwargs: Additional keyword arguments
        :return: The star state and its probability
        :rtype: tuple
        """
        if self._star_state is None or self._star_probability is None:
            self._star_state, self._star_probability = self.star_state_finding_strategy(
                self,
                partial(self.get_states_probability, show_progress=False),
                show_progress=self.show_progress,
                samples=self._samples,
            )
        return self.star_state, self.star_probability

    def exact_expval(self, observable: qml.operation.Operator) -> TensorLike:
        """
        Computes the expectation value of a given observable.

        This method evaluates the expectation value of the provided observable by using
        appropriate execution strategies or by splitting its terms. If the observable
        cannot be computed using any of the defined strategies, an error is raised.

        :param observable: The observable for which the expectation value is to be
            computed. It must be of a compatible type such as BasisStateProjector
            or BatchProjector, or match the supported execution strategies.
        :type observable: qml.operation.Operator
        :return: The computed expectation value of the given observable.
        :rtype: TensorLike
        :raises DeviceError: If the expectation value of the observable cannot
            be computed on the current device due to compatibility issues.
        """
        if isinstance(observable, BatchHamiltonian):
            return qml.math.stack(
                [qml.math.real(c * self.exact_expval(op)) for c, op in zip(observable.coeffs, observable.ops)]
            )
        if isinstance(observable, BasisStateProjector):
            return self.get_state_probability(observable.parameters[0], observable.wires)
        if isinstance(observable, BatchProjector):
            return self.get_states_probability(observable.get_states(), observable.get_batch_wires())
        if self.m_pfaffian_expval_strategy.can_execute(self.state_prep_op, observable):
            return self.m_pfaffian_expval_strategy(
                self.state_prep_op,
                observable,
                extended_covariance_matrix=self.extended_covariance_matrix,
            )
        if self.clifford_expval_strategy.can_execute(self.state_prep_op, observable):  # pragma: no cover
            return self.clifford_expval_strategy(
                self.state_prep_op, observable, global_sptm=self.global_sptm.matrix()
            )  # pragma: no cover
        if self.expval_from_probabilities_strategy.can_execute(self.state_prep_op, observable):
            return self.expval_from_probabilities_strategy(self.state_prep_op, observable, prob_func=self.probability)
        terms_splitter = TermsSplitter(
            [self.m_pfaffian_expval_strategy, self.clifford_expval_strategy, self.expval_from_probabilities_strategy]
        )
        if terms_splitter.can_execute(self.state_prep_op, observable):  # pragma: no cover
            return terms_splitter(  # pragma: no cover
                self.state_prep_op,
                observable,
                extended_covariance_matrix=self.extended_covariance_matrix,
                global_sptm=self.global_sptm.matrix(),
                prob_func=self.probability,
            )

        raise DeviceError(
            f"The expectation value of the observable {observable} "
            f"with the initial state {self.state_prep_op} "
            f"cannot be computed on a {self.name} device."
            "Please check the device's capabilities."
        )

    def execute(
        self,
        circuits: Union[QuantumScript, Iterable[QuantumScript]],
        execution_config: Optional[ExecutionConfig] = None,
    ) -> Union[Any, tuple]:
        """Execute a batch of circuits and return their results.

        :param circuits: A single circuit or an iterable of circuits to execute.
        :type circuits: Union[QuantumScript, Iterable[QuantumScript]]
        :param execution_config: Execution configuration options.
        :type execution_config: Optional[ExecutionConfig]
        :return: A single result for a single circuit, or a tuple of results for a batch.
        :rtype: Union[Any, tuple]
        """
        if isinstance(circuits, QuantumScript):
            return self._execute_circuit(circuits)
        return tuple(self._execute_circuit(c) for c in circuits)

    def execute_generator(
        self,
        op_iterator: Iterable[qml.operation.Operation],
        observable: Optional[Any] = None,
        output_type: Optional[Literal["samples", "expval", "probs", "star_state", "*state"]] = None,
        *,
        shots: Any = _UNSET,
        **kwargs,
    ) -> Optional[Any]:
        """
        Execute a generator of operations on the device and return the result in the specified output type.

        :param op_iterator: A generator of operations to execute
        :type op_iterator: Iterable[qml.operation.Operation]
        :param observable: The observable to measure
        :type observable: Optional
        :param output_type: The type of output to return. Supported types are "samples", "expval", and "probs"
        :type output_type: Optional[Literal["samples", "expval", "probs"]]
        :param shots: Number of shots for sample-based outputs, forwarded to :meth:`execute_output`.
            When provided, it overrides the device-level default for this call.
        :type shots: Optional[int]
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
        return self.execute_output(observable=observable, output_type=output_type, shots=shots, **kwargs)

    def execute_output(
        self,
        observable: Optional[Any] = None,
        output_type: Optional[Literal["samples", "expval", "probs", "star_state", "*state"]] = None,
        *,
        shots: Any = _UNSET,
        **kwargs,
    ) -> Optional[Any]:
        """
        Return the result of the execution in the specified output type.

        :param observable: The observable to measure
        :type observable: Optional
        :param output_type: The type of output to return. Supported types are "samples", "expval", and "probs"
        :type output_type: Optional[Literal["samples", "expval", "probs"]]
        :param shots: Number of shots for sample-based outputs. When provided, it overrides the
            device-level default for this call. When omitted, the device-level default (set at
            construction, ``None`` by default for analytic execution) is used. This is the preferred
            way to specify shots for direct ``execute_output``/``execute_generator`` use; with a
            QNode, use the ``qml.set_shots`` transform instead.
        :type shots: Optional[int]
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
            return None
        if shots is not _UNSET:
            if shots != self._current_shots:
                self._samples = None
            self._current_shots = shots
        if self._active_shots is not None and self._samples is None:
            self._samples = self.generate_samples()
        if output_type == "samples":
            if self._active_shots is None:
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

    def expval(
        self,
        observable: qml.operation.Operator,
        shot_range: Optional[tuple] = None,
        bin_size: Optional[int] = None,
    ) -> TensorLike:
        """
        Calculates the expectation value of a given observable.

        Uses the exact analytic method when no shots are active, and the PennyLane
        measurement process API for shot-based estimation.

        :param observable: Observable for which the expectation value is being calculated.
        :type observable: qml.operation.Operator
        :param shot_range: Tuple specifying the range of shots to consider.
        :type shot_range: Optional[tuple]
        :param bin_size: Integer specifying the number of shots per bin.
        :type bin_size: Optional[int]
        :return: The computed expectation value.
        :rtype: TensorLike
        """
        if self._active_shots is None:
            return self.exact_expval(observable)
        mp = ExpectationMP(obs=observable)
        return mp.process_samples(self._samples, wire_order=self.wires, shot_range=shot_range, bin_size=bin_size)

    def generate_samples(self) -> np.ndarray:
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
        self._samples = self.sampling_strategy.batch_generate_samples(
            self,
            partial(self.get_states_probability, show_progress=False),
            show_progress=self.show_progress,
        )
        assert self._samples is not None
        return self._samples

    def get_state_probability(self, target_binary_state: TensorLike, wires: Optional[Wires] = None) -> TensorLike:
        """Thin wrapper around :meth:`get_states_probability` for a single outcome.

        Accepts the same input formats (int, str, list, array) and returns a scalar.

        :param target_binary_state: The desired binary state. Accepts int, str, list, or array.
        :type target_binary_state: TensorLike
        :param wires: Wires to measure. Defaults to all device wires.
        :type wires: Optional[Wires]
        :return: Scalar probability.
        :rtype: TensorLike
        """
        return self.get_states_probability(target_binary_state, wires)

    def get_states_probability(
        self,
        target_binary_states: TensorLike,
        wires: Optional[Wires] = None,
        **kwargs,
    ) -> TensorLike:
        """Compute probabilities for one or a batch of basis outcomes.

        Dispatches through :attr:`prob_dispatcher` which routes to the first compatible
        strategy. Routing is determined by each strategy's ``can_execute`` method — no
        manual ``is_basis_state`` check is performed here.

        Detects input dimensionality:

        * 1-D input ``(k,)`` → returns a scalar.
        * 2-D input ``(B, k)`` → returns a ``(B,)`` vector.

        :param target_binary_states: Binary outcomes. Accepts int, str, list, or array
            of shape ``(k,)`` (single) or ``(B, k)`` (batch).
        :type target_binary_states: TensorLike
        :param wires: Measured wires. For a single outcome this is a flat ``Wires``; for
            a batch it is broadcast to ``(B, k)``. Defaults to all device wires.
        :type wires: Optional[Wires]
        :return: Scalar for single input, ``(B,)`` vector for batch input.
        :rtype: TensorLike
        """
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

        is_single = target_binary_states.ndim == 1

        if wires is None:
            wires = self.wires

        if is_single:
            wires = Wires(wires)
            assert len(target_binary_states) == len(wires), (
                f"The target binary state must have {len(wires)} elements. Got {len(target_binary_states)} instead."
            )
        else:
            wires = np.asarray(wires)
            wires = np.broadcast_to(wires, target_binary_states.shape)

        strategy_kwargs = dict(
            all_wires=self.wires,
            lookup_table=self.lookup_table,
            transition_matrix=self.transition_matrix,
            global_sptm=self.global_sptm.matrix(),
            majorana_getter=self.majorana_getter,
            show_progress=kwargs.pop("show_progress", self.show_progress),
        )
        if isinstance(self.state_prep_op, ProductState):
            strategy_kwargs["covariance_matrix"] = self.covariance_matrix

        return self.prob_dispatcher(
            state_prep_op=self.state_prep_op,
            target_binary_states=target_binary_states,
            wires=wires,
            **strategy_kwargs,
            **kwargs,
        )

    def initialize_p_bar(self, *args, **kwargs) -> Optional[tqdm.tqdm]:
        """Initialize the progress bar.

        :return: The progress bar instance, or None if progress display is disabled.
        :rtype: Optional[tqdm.tqdm]
        """
        kwargs.setdefault("disable", not self.show_progress)
        if self.p_bar is None and not self.show_progress:
            return None
        self.p_bar = tqdm.tqdm(*args, **kwargs)
        return self.p_bar

    def p_bar_set_n(self, n: int) -> None:
        """Set the current count on the progress bar.

        :param n: The current iteration count.
        :type n: int
        :return: None
        """
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def p_bar_set_postfix(self, *args, **kwargs) -> None:
        """Set a postfix dict on the progress bar.

        :return: None
        """
        if self.p_bar is not None:
            self.p_bar.set_postfix(*args, **kwargs)
            self.p_bar.refresh()

    def p_bar_set_postfix_str(self, *args, **kwargs) -> None:
        """Set a postfix string on the progress bar.

        :return: None
        """
        if self.p_bar is not None:
            self.p_bar.set_postfix_str(*args, **kwargs)
            self.p_bar.refresh()

    def p_bar_set_total(self, total: int) -> None:
        """Set the total count on the progress bar.

        :param total: The total number of iterations.
        :type total: int
        :return: None
        """
        if self.p_bar is not None:
            self.p_bar.total = total
            self.p_bar.refresh()

    def preprocess_transforms(self, execution_config: Optional[ExecutionConfig] = None) -> CompilePipeline:
        """Return the compile pipeline for preprocessing circuits before execution.

        The pipeline decomposes unsupported operations, splits circuits with non-commuting
        observables (preserving BatchHamiltonian measurements), and expands broadcast dimensions.

        :param execution_config: Execution configuration options.
        :type execution_config: Optional[ExecutionConfig]
        :return: The preprocessing compile pipeline.
        :rtype: CompilePipeline
        """
        program = CompilePipeline()
        program.add_transform(
            decompose,
            stopping_condition=self._stopping_condition,
            name=self.name,
            strict=False,
        )
        program.add_transform(_nif_split_non_commuting)
        return program

    def probability(
        self,
        wires: Optional[Union[Wires, List[int]]] = None,
        shot_range: Optional[tuple] = None,
        bin_size: Optional[int] = None,
    ) -> TensorLike:
        """
        Compute the marginal probability of each computational basis state.

        Uses the analytic method when no shots are active and sample-based estimation otherwise.

        :param wires: Wires to return marginal probabilities for. Defaults to all device wires.
        :type wires: Optional[Union[Wires, List[int]]]
        :param shot_range: Tuple specifying the range of shots to consider.
        :type shot_range: Optional[tuple]
        :param bin_size: Integer specifying the number of shots per bin.
        :type bin_size: Optional[int]
        :return: The computed probabilities.
        :rtype: TensorLike
        """
        wires = wires or self.wires
        if self._active_shots is None:
            return self.analytic_probability(wires=wires)
        mp = ProbabilityMP(wires=Wires(wires))
        return mp.process_samples(self._samples, wire_order=self.wires, shot_range=shot_range, bin_size=bin_size)

    def reset(self) -> None:
        """Reset the device state to its initial configuration.

        :return: None
        """
        self._global_sptm = None
        self._batched = False
        self._transition_matrix = None
        self._lookup_table = None
        self._star_state = None
        self._star_probability = None
        self._samples = None
        self.apply_metadata = defaultdict()
        self.contraction_strategy.reset()
        if self._wires is not None:
            self._state_prep_op = ProductState.from_basis_state(np.zeros(self.num_wires, dtype=int), wires=self.wires)

    def sample_basis_states(self, number_of_states: int, state_probability: np.ndarray) -> np.ndarray:
        """Sample basis state indices from a probability distribution.

        :param number_of_states: The number of basis states to sample from.
        :type number_of_states: int
        :param state_probability: The probability vector over basis states. Shape (number_of_states,)
            or (batch, number_of_states) for batched sampling.
        :type state_probability: np.ndarray
        :return: Sampled basis state indices, shape (shots,) or (batch, shots).
        :rtype: np.ndarray
        :raises ValueError: If no shots are set on the device.
        """
        if self._active_shots is None:
            raise ValueError("The number of shots must be specified to generate samples.")
        shots = self._active_shots
        basis_states = np.arange(number_of_states)
        state_probs = qml.math.unwrap(state_probability)
        if np.ndim(state_probs) == 2:
            return np.array([np.random.choice(basis_states, shots, p=prob) for prob in state_probs])
        return np.random.choice(basis_states, shots, p=state_probs)

    def setup_execution_config(
        self,
        config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumScript] = None,
    ) -> ExecutionConfig:
        """Configure execution settings, defaulting to torch backpropagation.

        :param config: Initial execution configuration.
        :type config: Optional[ExecutionConfig]
        :param circuit: The circuit to configure execution for.
        :type circuit: Optional[QuantumScript]
        :return: The updated execution configuration.
        :rtype: ExecutionConfig
        """
        if config is None:
            config = ExecutionConfig()
        if config.gradient_method in {"best", None}:
            config = _dataclass_replace(config, gradient_method="backprop")
        return config

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumScript] = None,
    ) -> bool:
        """Declare derivative support. Backpropagation is supported for analytic circuits.

        :param execution_config: The requested execution configuration.
        :type execution_config: Optional[ExecutionConfig]
        :param circuit: The circuit to check derivative support for.
        :type circuit: Optional[QuantumScript]
        :return: True when derivatives can be computed.
        :rtype: bool
        """
        if execution_config is None:
            return True
        if execution_config.gradient_method in {"backprop", "best"}:
            if circuit is None:
                return True
            return not circuit.shots
        return False

    def update_p_bar(self, *args, **kwargs) -> None:
        """Update the progress bar by one step.

        :return: None
        """
        if self.p_bar is None:
            return
        self.p_bar.update(*args, **kwargs)
        self.p_bar.refresh()

    def _setup_wire_dependent_state(self) -> None:
        """Initialize wire-dependent attributes from the current ``_wires``.

        Called at construction when wires are provided, or lazily on the first
        circuit execution / operation application when ``wires=None`` was used.

        :return: None
        """
        assert self.num_wires is not None
        kwargs = self._init_kwargs
        self.majorana_getter = kwargs.get("majorana_getter", utils.MajoranaGetter(self.num_wires, maxsize=256))
        assert isinstance(self.majorana_getter, utils.MajoranaGetter), (
            f"The majorana_getter must be an instance of {utils.MajoranaGetter}. "
            f"Got {type(self.majorana_getter)} instead."
        )
        assert self.majorana_getter.n == self.num_wires, (
            f"The majorana_getter must be initialized with {self.num_wires} wires. "
            f"Got {self.majorana_getter.n} instead."
        )
        self._state_prep_op = ProductState.from_basis_state(np.zeros(self.num_wires, dtype=int), wires=self.wires)

    def _asarray(self, x: TensorLike, dtype: Any = None) -> TensorLike:
        r"""
        Convert the input to an array of type ``dtype``.

        :Note: If the input is on cuda, it will be copied to cpu.

        :param x: input to be converted
        :type x: TensorLike
        :param dtype: type of the output array
        :type dtype: Optional[type]
        :return: array of type ``dtype``
        :rtype: TensorLike
        """
        is_complex = qml.math.any(qml.math.iscomplex(x))
        if dtype is None and is_complex:
            dtype = self.C_DTYPE
        elif dtype is None:
            dtype = self.R_DTYPE
        if not is_complex:
            x = qml.math.real(x)
        return qml.math.cast(torch_utils.to_cpu(x, dtype=dtype), dtype=dtype)

    def _dot(self, a: TensorLike, b: TensorLike) -> TensorLike:
        r"""
        Compute the dot product of two arrays.

        :param a: input array
        :type a: TensorLike
        :param b: input array
        :type b: TensorLike
        :return: dot product of the input arrays
        :rtype: TensorLike
        """
        return qml.math.einsum("...i,...i->...", self._asarray(a), self._asarray(b))

    def _execute_circuit(self, circuit: QuantumScript) -> Any:
        """Execute a single preprocessed circuit and return its result.

        :param circuit: The circuit to execute.
        :type circuit: QuantumScript
        :return: The measurement result(s) for the circuit.
        :rtype: Any
        """
        self._current_shots = circuit.shots.total_shots
        if self._wires is None:
            assert len(circuit.wires) > 1, "At least two wires are required for this device."
            self._wires = circuit.wires
            self._setup_wire_dependent_state()
        self.reset()
        self.apply_generator(iter(circuit.operations))
        if self._current_shots is not None:
            self._samples = self.generate_samples()
        results = [self._process_measurement(m) for m in circuit.measurements]
        return results[0] if len(results) == 1 else tuple(results)

    def _process_measurement(self, measurement: qml.measurements.MeasurementProcess) -> Any:
        """Dispatch a single measurement to the appropriate handler.

        :param measurement: The measurement process to evaluate.
        :type measurement: qml.measurements.MeasurementProcess
        :return: The measurement result.
        :rtype: Any
        :raises NotImplementedError: If the measurement type is not supported.
        """
        if isinstance(measurement, ExpectationMP):
            return self.expval(measurement.obs)
        if isinstance(measurement, ProbabilityMP):
            return self.probability(wires=measurement.wires or self.wires)
        if isinstance(measurement, SampleMP):
            if measurement.obs is None:
                return self._samples
            return measurement.process_samples(self._samples, wire_order=self.wires)
        raise NotImplementedError(f"Measurement {type(measurement).__name__} is not supported by {self.name}.")

    def _stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Return True when an operator is natively supported by this device.

        An operator is natively supported if its name is in ``_supported_ops`` or if it can be
        converted to a :class:`MatchgateOperation` (i.e. it is a native matchgate such as
        ``IsingXX``). Stopping the decomposition on native matchgates lets them reach
        :meth:`apply_op`, where they are converted by :meth:`convert_op_to_supported`.

        :param op: The operator to check.
        :type op: qml.operation.Operator
        :return: True if the operator is natively supported.
        :rtype: bool
        """
        if op.name in self._supported_ops:
            return True
        try:
            self.convert_op_to_supported(op)
            return True
        except DeviceError:
            return False

    @property
    def covariance_matrix(self) -> TensorLike:
        """Majorana covariance matrix evolved under the current global SPTM.

        :return: The covariance matrix, shape (..., 2n, 2n).
        :rtype: TensorLike
        """
        if not isinstance(self.state_prep_op, ProductState):
            raise ValueError(
                f"Covariance matrix can only be computed for product states. "
                f"Got {type(self.state_prep_op)} instead of ProductState."
            )
        cov0 = self.state_prep_op.covariance_matrix
        sptm = self.global_sptm.matrix(self.wires)
        return qml.math.einsum("...ij,...ik,...kl->...jl", sptm, cov0, sptm)

    @property
    def extended_covariance_matrix(self) -> TensorLike:
        """Extended (2n+1)x(2n+1) covariance matrix tilde_Lambda.

        Lifts the standard covariance matrix by appending a displacement
        column/row at position 2n (the parity index). Evolves the full
        extended state under the matchgate SPTM as tilde_Q^T tilde_Lambda_0 tilde_Q.

        BasisState inputs are promoted to ProductState first (displacement is zero).

        See expval_from_mpf.md for the math.
        """
        state_prep = self.state_prep_op
        if isinstance(state_prep, BasisState):
            state_prep = ProductState.from_basis_state(state_prep)
        if not isinstance(state_prep, ProductState):
            raise ValueError(
                f"Extended covariance matrix requires a ProductState or BasisState. Got {type(state_prep)}."
            )
        # Build tilde_Lambda_0 from the initial state. Cast to the device's configured
        # precision so r_dtype/c_dtype control the whole m-Pfaffian pipeline.
        amplitudes = qml.math.cast(state_prep.data[0], self._c_dtype_name)  # (n, 2) complex
        cov0 = qml.math.cast(state_prep.covariance_matrix, self._r_dtype_name)  # (2n, 2n)
        d0 = _displacement_vector(amplitudes, self.wires)  # (2n,)
        ext_cov0 = _extended_covariance_matrix(cov0, d0)  # (2n+1, 2n+1)

        # Lift SPTM: \tilde{R} = R ⊕ 1
        sptm = self.global_sptm.matrix(self.wires)  # (2n, 2n)
        lifted_sptm = _sptm_lift(sptm)  # (2n+1, 2n+1)

        # Evolve: tilde_Lambda(t) = tilde_Q^T tilde_Lambda_0 tilde_Q
        return qml.math.einsum("...ij,...ik,...kl->...jl", lifted_sptm, ext_cov0, lifted_sptm)

    @property
    def global_sptm(self) -> SingleParticleTransitionMatrixOperation:
        """The current global single-particle transition matrix.

        :return: The global SPTM wrapped as a SingleParticleTransitionMatrixOperation.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        if self._global_sptm is None:
            assert self.num_wires is not None
            matrix = qml.math.cast(np.eye(2 * self.num_wires)[None, ...], self._r_dtype_name)
            if not self._batched:
                matrix = matrix[0]
            return SingleParticleTransitionMatrixOperation(matrix, wires=self.wires)
        return self._global_sptm

    @global_sptm.setter
    def global_sptm(self, value: Union[Operation, TensorLike]) -> None:
        if isinstance(value, Operation):
            value = SingleParticleTransitionMatrixOperation.from_operation(value)
        if not isinstance(value, SingleParticleTransitionMatrixOperation):
            value = SingleParticleTransitionMatrixOperation(value, wires=self.wires)
        matrix = qml.math.cast(value.matrix(), self._r_dtype_name)
        self._global_sptm = SingleParticleTransitionMatrixOperation(matrix, wires=value.wires, **value._hyperparameters)
        self.transition_matrix = None

    @property
    def lookup_table(self) -> NonInteractingFermionicLookupTable:
        """Lazily-constructed lookup table for the current transition matrix.

        :return: The lookup table.
        :rtype: NonInteractingFermionicLookupTable
        """
        if self._lookup_table is None:
            self._lookup_table = NonInteractingFermionicLookupTable(self.transition_matrix)
        return self._lookup_table

    @property
    def num_wires(self) -> Optional[int]:
        """The number of wires on this device, or ``None`` when wires are not yet determined.

        :return: Number of wires, or None if the device has not been assigned wires yet.
        :rtype: Optional[int]
        """
        if self._wires is None:
            return None
        return len(self._wires)

    @property
    def samples(self) -> Optional[TensorLike]:
        """The computational basis samples from the last execution.

        :return: Sample array of shape (shots, num_wires), or None.
        :rtype: Optional[TensorLike]
        """
        return self._samples

    @property
    def shots(self) -> Optional[int]:
        """Return shots as int or None for the current execution context.

        During a circuit execution the count comes from ``_current_shots`` (set per-circuit).
        At all other times the device-level default is used.

        :return: Number of shots, or None for analytic execution.
        :rtype: Optional[int]
        """
        return self._active_shots

    @property
    def star_probability(self) -> Optional[TensorLike]:
        """The probability of the star state from the last computation.

        :return: Star state probability, or None if not yet computed.
        :rtype: Optional[TensorLike]
        """
        return self._star_probability

    @property
    def star_state(self) -> Optional[TensorLike]:
        """The star state (highest-probability basis state) from the last computation.

        :return: Star state array, or None if not yet computed.
        :rtype: Optional[TensorLike]
        """
        return self._star_state

    @property
    def state(self) -> np.ndarray:
        """
        NIF device does not support dense state representation.

        :return: state vector of the device
        :rtype: array[complex]

        :Note: This function comes from the ``default.qubit`` device.
        """
        raise NotImplementedError(
            "The NIF device does not support dense state representation. "
            "It would require an exponential amount of memory."
        )

    @property
    def state_prep_op(self) -> StatePrepBase:
        """The current state preparation operator.

        :return: The state preparation operator.
        :rtype: StatePrepBase
        """
        return self._state_prep_op

    @property
    def transition_matrix(self) -> Optional[TensorLike]:
        """The transition matrix derived from the global SPTM.

        :return: The transition matrix, or None if no operations have been applied.
        :rtype: Optional[TensorLike]
        """
        if self._transition_matrix is None and self.global_sptm is not None:
            self._transition_matrix = utils.make_transition_matrix_from_action_matrix(self.global_sptm.matrix())
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value: Optional[TensorLike]) -> None:
        if self._global_sptm is not None and value is not None:
            global_sptm_matrix = self._global_sptm.matrix()
            value = convert_like_and_cast_to(
                value, global_sptm_matrix, dtype=complex_dtype_name_like(global_sptm_matrix)
            )
        self._transition_matrix = value
        self._lookup_table = None

    @property
    def _active_shots(self) -> Optional[int]:
        """Return shots as int (or None) for the current execution context.

        :return: Active shot count for the current circuit, or the device default.
        :rtype: Optional[int]
        """
        if self._current_shots is not None:
            return self._current_shots
        return self._shots.total_shots
