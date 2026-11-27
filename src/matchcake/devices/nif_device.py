import warnings
from collections import defaultdict
from dataclasses import replace as _dataclass_replace
from functools import partial
from itertools import product
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
import torch
import tqdm
from pennylane import BasisState
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.preprocess import decompose
from pennylane.exceptions import DeviceError
from pennylane.measurements import (
    DensityMatrixMP,
    ExpectationMP,
    ProbabilityMP,
    PurityMP,
    SampleMP,
    Shots,
    VnEntropyMP,
)
from pennylane.operation import Operation, StatePrepBase
from pennylane.ops.op_math import Adjoint, Pow
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.tape import QuantumScript
from pennylane.transforms.core import CompilePipeline
from pennylane.wires import Wires

from .. import utils
from ..base.lookup_table import NonInteractingFermionicLookupTable
from ..measurements.covariance_matrix import CovarianceMatrixMP
from ..measurements.reduced_state import ReducedStateMP
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..observables.batch_projector import BatchProjector
from ..operations.matchgate_operation import MatchgateOperation
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..operations.state_preparation import (
    GaussianStatePrep,
    GaussianStatePrepBase,
    ProductState,
    StatePrepFromGates,
)
from ..typing import TensorLike
from ..utils import (
    torch_utils,
)
from ..utils.covariance_matrix_toolbox import CovarianceMatrixToolbox
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
    extended_covariance_matrix as _extended_covariance_matrix,
)
from .expval_strategies.m_pfaffian._extended_covariance import (
    sptm_lift as _sptm_lift,
)
from .expval_strategies.terms_splitter import TermsSplitter
from .probability_strategies import (
    CliffordSumStrategy,
    ExplicitSumStrategy,
    LookupTableStrategy,
    ProbabilityFuncDispatcher,
    ProductStateProbabilityStrategy,
)
from .sampling_strategies import ProductBasisSampling, SamplingStrategy, get_sampling_strategy
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

    :param wires: The number of wires of the device, or its wire labels. Labels must be the consecutive
        integers ``0, 1, ..., N - 1`` (see the note on wire labels below).
    :type wires: Union[int, Wires, List[int]]
    :param shots: The number of shots used by the device. Defaults to ``None`` (analytic mode).
    :type shots: Optional[int]
    :param r_dtype: The real floating-point dtype used for real-valued tensors. Accepts a ``torch.dtype``, a
        numpy scalar type (e.g. ``np.float32``) or a Python builtin (e.g. ``float``). Defaults to the class
        attribute :attr:`R_DTYPE` (``torch.float64``).
    :type r_dtype: Optional[Union[torch.dtype, type]]
    :param c_dtype: The complex dtype used for complex-valued tensors throughout the pipeline (single-particle
        transition matrices, lookup table, observables). Accepts a ``torch.dtype``, a numpy scalar type
        (e.g. ``np.complex64``) or a Python builtin (e.g. ``complex``). Defaults to the class attribute
        :attr:`C_DTYPE` (``torch.complex128``) to preserve maximum precision. Set to e.g. ``torch.complex64``
        to reduce memory usage and computation cost.
    :type c_dtype: Optional[Union[torch.dtype, type]]
    :param pfaffian_chunk_size: Max number of matrices reduced per batched Pfaffian call, forwarded to
        :func:`matchcake.utils.pfaffian` to bound memory. Defaults to ``None`` (no chunking).
    :type pfaffian_chunk_size: Optional[int]
    :param show_progress: Whether to display progress bars while applying operations and computing
        probabilities. Defaults to ``None``, meaning progress is shown only when an external progress bar is
        supplied through the ``p_bar`` keyword argument.
    :type show_progress: Optional[bool]

    :kwargs: Additional keyword arguments

    :keyword majorana_getter: The Majorana getter to use. Defaults to a new instance of MajoranaGetter.
    :type majorana_getter: MajoranaGetter
    :keyword contraction_strategy: The contraction strategy used to merge consecutive operations. Accepts a
        strategy name or instance, or ``None`` to disable contraction. Defaults to
        :attr:`DEFAULT_CONTRACTION_METHOD`.
    :type contraction_strategy: Optional[Union[str, ContractionStrategy]]
    :keyword sampling_strategy: The strategy used to draw samples in finite-shot mode, as a name or an
        instance. Defaults to :attr:`DEFAULT_SAMPLING_STRATEGY`. A
        :class:`~matchcake.devices.sampling_strategies.ProductBasisSampling` instance is also used for the
        samples in other bases, so that e.g. ``ProductBasisSampling(device="cuda")`` samples every basis on
        the GPU. The default
        :class:`~matchcake.devices.sampling_strategies.ProductBasisSampling` needs a product state (or basis
        state) input, and :attr:`FALLBACK_SAMPLING_STRATEGY` is used instead for other state preparations.
    :type sampling_strategy: Union[str, SamplingStrategy]
    :keyword star_state_finding_strategy: The strategy to find the star state. Defaults to
        :attr:`DEFAULT_STAR_STATE_FINDING_STRATEGY`.
    :type star_state_finding_strategy: Union[str, StarStateFindingStrategy]
    :keyword p_bar: An external progress bar reused by the device instead of creating its own. Defaults to
        ``None``. Supplying one turns ``show_progress`` on unless ``show_progress`` is given a boolean value.
    :type p_bar: Optional[tqdm.tqdm]

    :Note: This device is a simulator for non-interacting fermions. It is based on the ``default.qubit`` device.
    :Note: This device supports batch execution.
    :Note: Probabilities are routed by :attr:`prob_dispatcher`, a fixed ordered chain of strategies resolved by
        each strategy's ``can_execute``. There is no keyword to select a strategy; to pin one, replace
        :attr:`prob_dispatcher` on the constructed device.
    :Note: With shots, observables that are not diagonal in the computational basis can be measured when
        they are products of Pauli operators that use a single Pauli basis per wire, such as ``qml.X(0)``,
        ``qml.X(0) @ qml.Y(1)`` or a Hamiltonian whose terms commute qubit-wise. The samples are then drawn
        in that product basis by :class:`~matchcake.devices.sampling_strategies.ProductBasisSampling`,
        which requires a product state (or basis state) input. See :meth:`generate_samples`.
    :Note: Only the wire labels ``0, 1, ..., N - 1`` are supported, in this order. Custom labels such as
        ``[10, 20, 30]`` or ``["a", "b", "c"]`` are not supported and raise an error at execution time.
        Wires used in circuits and measurements (e.g. ``qml.sample(wires=[2, 0])``) must refer to these labels.
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
        GaussianStatePrep.__name__,
    }

    DEFAULT_CONTRACTION_METHOD = "neighbours"
    DEFAULT_SAMPLING_STRATEGY = "ProductBasisSampling"
    FALLBACK_SAMPLING_STRATEGY = "2QubitBy2QubitSampling"
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

    @staticmethod
    def _has_sptm_interface(op: qml.operation.Operator) -> bool:
        """Return True when an operator is an SPTM or knows how to produce one.

        :param op: The operator to check.
        :type op: qml.operation.Operator
        :return: True if ``op`` is a :class:`MatchgateOperation`, a
            :class:`SingleParticleTransitionMatrixOperation`, or exposes a ``to_sptm_operation``
            method or a ``single_particle_transition_matrix`` attribute.
        :rtype: bool
        """
        if isinstance(op, (MatchgateOperation, SingleParticleTransitionMatrixOperation)):
            return True
        return hasattr(op, "to_sptm_operation") or hasattr(op, "single_particle_transition_matrix")

    @staticmethod
    def _can_process_samples_directly(observable: qml.operation.Operator) -> bool:
        """Return True when computational basis samples can be processed with the observable itself.

        PennyLane maps samples to the eigenvalues of the observable assuming its diagonalizing
        gates were applied, which this device never does. This is only correct when the
        observable is diagonal and its eigenvalues are given in computational basis order:
        Pauli words made only of ``Z`` and identity factors, or observables that need no
        diagonalizing gates.

        :param observable: The observable to check.
        :type observable: qml.operation.Operator
        :return: True if the samples can be processed with the observable itself.
        :rtype: bool
        """
        pauli_rep = observable.pauli_rep
        if pauli_rep is not None:
            return all(set(pauli_word.values()) <= {"Z"} for pauli_word in pauli_rep)
        if not observable.has_diagonalizing_gates:
            return False
        return len(observable.diagonalizing_gates()) == 0

    @staticmethod
    def _diagonal_matrix_eigvals(observable: qml.operation.Operator) -> Optional[np.ndarray]:
        """Return the eigenvalues of an observable with a diagonal matrix, in computational basis order.

        This covers diagonal observables such as a diagonal ``qml.Hermitian``, whose
        ``eigvals`` are sorted and therefore not aligned with the computational basis states.

        :param observable: The observable to inspect.
        :type observable: qml.operation.Operator
        :return: The diagonal of the matrix of the observable, shape (2^k,) for k wires, or None
            if the observable has no matrix, is batched, or is not diagonal.
        :rtype: Optional[np.ndarray]
        """
        if not observable.has_matrix or observable.batch_size is not None:
            return None
        matrix = qml.math.to_numpy(observable.matrix())
        diagonal = np.diagonal(matrix)
        if not np.allclose(matrix, np.diag(diagonal)):
            return None
        return np.real(diagonal)

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
            new_sptm = SingleParticleTransitionMatrixOperation.from_operation(new_sptm).sptm_matrix()
        if old_sptm is None:
            return new_sptm
        if isinstance(old_sptm, Operation):
            old_sptm = SingleParticleTransitionMatrixOperation.from_operation(old_sptm).sptm_matrix()
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
        r_dtype: Optional[Union[torch.dtype, type]] = None,
        c_dtype: Optional[Union[torch.dtype, type]] = None,
        pfaffian_chunk_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
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

        self.R_DTYPE = torch_utils.get_torch_dtype(r_dtype, type(self).R_DTYPE)
        self.C_DTYPE = torch_utils.get_torch_dtype(c_dtype, type(self).C_DTYPE)
        self._c_dtype_name = str(self.C_DTYPE).rsplit(".", 1)[-1]
        self._r_dtype_name = str(self.R_DTYPE).rsplit(".", 1)[-1]

        self._star_state = None
        self._star_probability = None

        self._transition_matrix: Optional[TensorLike] = None
        self._lookup_table: Optional[NonInteractingFermionicLookupTable] = None
        self._global_sptm: Optional[SingleParticleTransitionMatrixOperation] = None
        # True once the stored matrix was handed out, so that the next gate writes into a copy of it.
        self._global_sptm_is_shared = False
        self._batched = False
        self._samples: Optional[np.ndarray] = None
        self._samples_bases: Dict[Any, np.ndarray] = {}
        self._current_shots: Optional[int] = None

        self.sampling_strategy: SamplingStrategy = get_sampling_strategy(
            kwargs.get("sampling_strategy", self.DEFAULT_SAMPLING_STRATEGY)
        )
        self.fallback_sampling_strategy: SamplingStrategy = get_sampling_strategy(self.FALLBACK_SAMPLING_STRATEGY)
        self.product_basis_sampling_strategy: ProductBasisSampling = (
            self.sampling_strategy
            if isinstance(self.sampling_strategy, ProductBasisSampling)
            else ProductBasisSampling()
        )
        self.prob_dispatcher: ProbabilityFuncDispatcher = ProbabilityFuncDispatcher(
            [
                LookupTableStrategy(),
                ProductStateProbabilityStrategy(),
                CliffordSumStrategy(),
                ExplicitSumStrategy(),
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
        self.pfaffian_chunk_size: Optional[int] = pfaffian_chunk_size
        self.p_bar: Optional[tqdm.tqdm] = kwargs.get("p_bar", None)
        self.show_progress: bool = (self.p_bar is not None) if show_progress is None else show_progress
        self.apply_metadata: defaultdict = defaultdict()
        self.clifford_expval_strategy = CliffordExpvalStrategy()
        self.expval_from_probabilities_strategy = ExpvalFromProbabilitiesStrategy()
        self.m_pfaffian_expval_strategy = MPfaffianExpvalStrategy()

        self.majorana_getter: Optional[utils.MajoranaGetter] = None
        self._state_prep_op: Optional[StatePrepBase] = None
        if self._wires is not None:
            self._setup_wire_dependent_state()

    def __copy__(self) -> "NonInteractingFermionicDevice":
        """Return a shallow copy whose gates do not write into the matrix of this device.

        :return: The copy.
        :rtype: NonInteractingFermionicDevice
        """
        new_device = type(self).__new__(type(self))
        new_device.__dict__.update(self.__dict__)
        # both devices hold the same matrix, so each one copies it before its next gate
        self._global_sptm_is_shared = True
        new_device._global_sptm_is_shared = True
        return new_device

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
        # A subclass that overrides apply_op is still called for every operation. The base one is
        # skipped because it marks the matrix it returns as shared, which costs a copy per gate.
        is_overridden = type(self).apply_op is not NonInteractingFermionicDevice.apply_op
        apply_op = self.apply_op if is_overridden else self._apply_op
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
            self._check_operation_against_barriers(op)
            op = self.convert_op_to_supported(op)
            op_list = self.contraction_strategy.get_next_operations(op)
            for j, op_j in enumerate(op_list):
                if op_j is None:
                    continue
                apply_op(op_j)
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
            apply_op(last_op)
            self.apply_metadata["n_contracted_operations"] = self.apply_metadata.get("n_contracted_operations", 0) + 1
        self.apply_metadata["percentage_contracted"] = (
            100
            * (self.apply_metadata["n_operations"] - self.apply_metadata.get("n_contracted_operations", 0))
            / max(self.apply_metadata["n_operations"], 1)
        )

        self.p_bar_set_n(self.apply_metadata["n_operations"])
        self.p_bar_set_postfix_str("Computing transition matrix")
        if kwargs.get("cache_global_sptm", False):
            self.apply_metadata["global_sptm"] = torch_utils.to_numpy(self.global_sptm.sptm_matrix())
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
        returned unchanged and handled downstream by :meth:`apply_op`. Nested lazy
        :class:`~pennylane.ops.op_math.Adjoint` and integer :class:`~pennylane.ops.op_math.Pow`
        wrappers around one of these are converted directly to the corresponding SPTM, without
        decomposing. Otherwise, the operation's
        matrix is fed to the :class:`MatchgateOperation` constructor, which validates that it is a
        matchgate. If the matrix is a matchgate, the resulting :class:`MatchgateOperation` is
        returned; if not, a :class:`DeviceError` is raised.

        :param op: The operation to check and convert.
        :type op: qml.operation.Operation
        :return: A supported operation equivalent to ``op``.
        :rtype: qml.operation.Operation
        :raises DeviceError: If ``op`` is neither a supported operation nor a matchgate.
        """
        if self._has_sptm_interface(op):
            return op
        base = op
        while isinstance(base, Adjoint) or (
            isinstance(base, Pow) and SingleParticleTransitionMatrixOperation._is_integer_exponent(base.z)
        ):
            base = base.base
        if base is not op and self._has_sptm_interface(base):
            return SingleParticleTransitionMatrixOperation.from_operation(op)
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
        self._apply_op(op)
        return self.global_sptm

    def _apply_op(self, op: qml.operation.Operation) -> None:
        """Multiply the global single-particle transition matrix by the one of an operation.

        Only the columns the operation acts on are updated, in place. The matrix is copied first when
        it was handed out since the last update, so that a matrix read from the device never changes.

        :param op: The operation to apply.
        :type op: qml.operation.Operation
        """
        sptm_op = SingleParticleTransitionMatrixOperation.from_operation(op)
        self._batched = self._batched or (sptm_op.batch_size is not None)
        wires = SingleParticleTransitionMatrixOperation.make_wires_continuous(Wires(self.wires))
        blocks = [(start, qml.math.cast(block, self._r_dtype_name)) for start, block in sptm_op.column_blocks(wires)]
        if self._global_sptm is None:
            matrix = qml.math.cast(np.eye(2 * len(wires)), self._r_dtype_name)
        else:
            matrix = self._global_sptm.single_particle_transition_matrix
            if self._global_sptm_is_shared:
                matrix = matrix.clone() if isinstance(matrix, torch.Tensor) else np.array(matrix)
        matrix = SingleParticleTransitionMatrixOperation.multiply_columns(matrix, blocks)
        if matrix is None:
            op_sptm = qml.math.cast(sptm_op.pad(self.wires).sptm_matrix(), self._r_dtype_name)
            matrix = self.update_single_particle_transition_matrix(self._global_sptm, op_sptm)
        self._global_sptm = SingleParticleTransitionMatrixOperation(matrix, wires=self.wires)
        self._global_sptm_is_shared = False
        self.transition_matrix = None

    def apply_state_prep(self, operation: qml.operation.Operation, **kwargs) -> bool:
        r"""
        Apply a state preparation operation to the device. Will set the internal state of the device
        only if the operation is a state preparation operation and then return True. Otherwise, it will
        return False.

        A product state preparation (:class:`~pennylane.BasisState` or
        :class:`~matchcake.operations.state_preparation.ProductState`) is extended to the device wires in
        the device wire order, so it can act on part of the wires or list them in any order. The other
        wires start in :math:`|0\rangle`. A
        :class:`~matchcake.operations.state_preparation.GaussianStatePrep` can also act on part of the
        wires, which it must list in the device wire order.

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
            operation = ProductState.from_basis_state(operation)
        if isinstance(operation, GaussianStatePrepBase) and self._wires is not None:
            self._state_prep_op = operation.to_wires(self.wires)
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
            computed. It must be of a compatible type such as BasisStateProjector,
            BatchProjector or Hermitian (decomposed into Pauli words), or match the supported
            execution strategies.
        :type observable: qml.operation.Operator
        :return: The computed expectation value of the given observable.
        :rtype: TensorLike
        :raises DeviceError: If the expectation value of the observable cannot
            be computed on the current device due to compatibility issues, or if the observable
            breaks the rule of a gate barrier of the initial state.
        """
        self._check_observable_against_barriers(observable)
        if isinstance(observable, BatchHamiltonian):
            return qml.math.stack(
                [qml.math.real(c * self.exact_expval(op)) for c, op in zip(observable.coeffs, observable.ops)]
            )
        if isinstance(observable, BasisStateProjector):
            return self.get_state_probability(observable.parameters[0], observable.wires)
        if isinstance(observable, BatchProjector):
            return self.get_states_probability(observable.get_states(), observable.get_batch_wires())
        if isinstance(observable, qml.Hermitian):
            return self._hermitian_exact_expval(observable)
        if self.m_pfaffian_expval_strategy.can_execute(self.state_prep_op, observable):
            return self.m_pfaffian_expval_strategy(
                self.state_prep_op,
                observable,
                extended_covariance_matrix=self.extended_covariance_matrix,
                pfaffian_chunk_size=self.pfaffian_chunk_size,
            )
        if self.clifford_expval_strategy.can_execute(self.state_prep_op, observable):  # pragma: no cover
            return self.clifford_expval_strategy(
                self.state_prep_op, observable, global_sptm=self.global_sptm.sptm_matrix()
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
                global_sptm=self.global_sptm.sptm_matrix(),
                prob_func=self.probability,
                pfaffian_chunk_size=self.pfaffian_chunk_size,
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
        mp = self._build_sampled_measurement(ExpectationMP, observable)
        return mp.process_samples(
            self._samples_in_pennylane_layout(), wire_order=self.wires, shot_range=shot_range, bin_size=bin_size
        )

    def generate_samples(self, bases: Optional[Mapping[Any, Union[str, Sequence[float]]]] = None) -> np.ndarray:
        r"""Generate samples for all wires, in the computational basis or in a product basis.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        Without ``bases``, or when every basis is ``"Z"``, the samples are drawn in the computational
        basis by :attr:`sampling_strategy`, or by :attr:`fallback_sampling_strategy` when
        :attr:`sampling_strategy` is a :class:`~matchcake.devices.sampling_strategies.ProductBasisSampling`
        and the state preparation is not a Gaussian state. Otherwise, each wire listed in ``bases`` is measured in its
        own basis and the samples are drawn by :attr:`product_basis_sampling_strategy`, which requires a
        product state, basis state or Gaussian state input. Bit ``0`` then stands for the eigenvalue :math:`+1` of the
        measured Pauli operator, or of :math:`\hat{n} \cdot \vec{\sigma}` for a Bloch vector
        :math:`\hat{n}`, and bit ``1`` for :math:`-1`.

        :param bases: The measurement basis of some wires, keyed by wire label. A basis is a Pauli
            letter (``"X"``, ``"Y"`` or ``"Z"``) or a Bloch vector of length 3. Wires not listed are
            measured in the computational basis. Defaults to None.
        :type bases: Optional[Mapping[Any, Union[str, Sequence[float]]]]
        :return: Samples of shape (shots, ..., num_wires), also stored in :attr:`samples`.
        :rtype: np.ndarray
        :raises DeviceError: If a non computational basis is requested and the state preparation is
            not a product state, a basis state or a Gaussian state, or if such a basis is requested
            on a wire that is on the right of a gate barrier of the initial state.
        """
        bloch_vectors = self._bases_to_bloch_vectors(bases or {})
        rotated_bases = {
            wire: bloch_vector
            for wire, bloch_vector in zip(self.wires, bloch_vectors)
            if not np.allclose(bloch_vector, ProductBasisSampling.PAULI_BLOCH_VECTORS["Z"])
        }
        self._check_sampling_bases_against_barriers(rotated_bases)
        if not rotated_bases:
            sampling_strategy = self.sampling_strategy
            if isinstance(sampling_strategy, ProductBasisSampling) and not isinstance(
                self.state_prep_op, GaussianStatePrepBase
            ):
                sampling_strategy = self.fallback_sampling_strategy
            self._samples = sampling_strategy.batch_generate_samples(
                self,
                partial(self.get_states_probability, show_progress=False),
                show_progress=self.show_progress,
            )
        else:
            if not isinstance(self.state_prep_op, GaussianStatePrepBase):
                raise DeviceError(
                    f"Sampling in a basis other than the computational basis on {self.name} requires a product "
                    f"state, basis state or Gaussian state preparation. Got {self.state_prep_op} instead."
                )
            self._samples = self.product_basis_sampling_strategy.batch_generate_samples(
                self, bloch_vectors=bloch_vectors, show_progress=self.show_progress
            )
        self._samples_bases = rotated_bases
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
            global_sptm=self.global_sptm.sptm_matrix(),
            majorana_getter=self.majorana_getter,
            show_progress=kwargs.pop("show_progress", self.show_progress),
            pfaffian_chunk_size=kwargs.pop("pfaffian_chunk_size", self.pfaffian_chunk_size),
        )
        if isinstance(self.state_prep_op, GaussianStatePrepBase):
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
        return mp.process_samples(
            self._samples_in_pennylane_layout(), wire_order=self.wires, shot_range=shot_range, bin_size=bin_size
        )

    def partial_trace(
        self,
        wires: Union[Wires, List[int]],
        *,
        reduced_wires: Optional[Union[Wires, List[int]]] = None,
        gate_barrier: bool = True,
    ) -> GaussianStatePrep:
        r"""
        Trace out the given wires of the current state and return the state of the remaining ones.

        See :meth:`reduced_state` for the semantics of the result. This method takes the wires to remove
        instead of the wires to keep.

        :param wires: Wires to trace out.
        :type wires: Union[Wires, List[int]]
        :param reduced_wires: Wire labels of the returned state preparation. Defaults to ``range(k)`` for
            ``k`` remaining wires.
        :type reduced_wires: Optional[Union[Wires, List[int]]]
        :param gate_barrier: Whether the returned state preparation carries the gate barriers left by the
            removed wires, see :meth:`reduced_state`.
        :type gate_barrier: bool
        :return: The state of the remaining wires as a state preparation.
        :rtype: GaussianStatePrep
        :raises ValueError: If a wire is not a wire of the device.
        """
        traced_wires = Wires(wires)
        unknown_wires = [wire for wire in traced_wires if wire not in self.wires]
        if unknown_wires:
            raise ValueError(f"Wires {unknown_wires} are not wires of the device ({self.wires.tolist()}).")
        kept_wires = [wire for wire in self.wires if wire not in traced_wires]
        return self.reduced_state(kept_wires, reduced_wires=reduced_wires, gate_barrier=gate_barrier)

    def reduced_covariance_matrix(self, wires: Union[Wires, List[int]], *, gate_barrier: bool = True) -> TensorLike:
        r"""
        Majorana covariance matrix of the current state on the given wires, the others being traced out.

        The current state is the initial state evolved by the operations applied so far. Its reduced state
        on a subset of wires is the fermionic Gaussian state, generally mixed, whose covariance matrix is
        the principal submatrix of :attr:`covariance_matrix` on the Majorana indices of the kept wires.

        The Majorana operators of the reduced state carry Jordan-Wigner strings that run through the kept
        wires only. When the kept wires are a contiguous block listed in the device wire order, this is
        the reduced state of the qubits. Otherwise, a Pauli word on the reduced state stands in the full
        system for the same word with a :math:`Z` on every removed wire located between two of its
        :math:`X` or :math:`Y` factors: the result is the reduced state of the fermionic modes, which is
        not the reduced state of the qubits. A matrix cannot carry the gate barriers that make this safe,
        so that case is refused here. :meth:`reduced_state` returns the reduced state with its barriers,
        and ``gate_barrier=False`` returns the matrix of the fermionic reduced state with a warning.

        This also holds for an initial state with a nonzero displacement vector, such as a product state
        with superposed qubits: the products of two Majorana operators of a contiguous block are supported
        on that block, so their expectation values do not involve the displacement. The covariance matrix
        alone does not determine the reduced state then, see :meth:`reduced_state`.

        :param wires: Wires to keep.
        :type wires: Union[Wires, List[int]]
        :param gate_barrier: Whether to refuse the kept wires that are not listed in the device wire order
            or that have a removed wire or a gate barrier of the initial state between them.
        :type gate_barrier: bool
        :return: The reduced covariance matrix, shape (..., 2k, 2k) for k kept wires.
        :rtype: TensorLike
        :raises ValueError: If a wire is not a wire of the device, if no wire is kept, or if the kept wires
            are not listed in the device wire order or have a removed wire or a gate barrier between them
            while ``gate_barrier`` is ``True``.
        """
        wire_indices = self._reduced_wire_indices(wires, any_order=True)
        if self._has_barrier_inside(wire_indices):
            if gate_barrier:
                raise ValueError(
                    f"The kept wires (device positions {wire_indices}) are not a block of neighbouring wires "
                    f"listed in the device wire order without gate barrier on {self.name}: a removed wire or "
                    f"a gate barrier lies between them. The Jordan-Wigner strings of their Majorana operators then "
                    f"run through removed wires, so the covariance matrix restricted to them describes the "
                    f"reduced state of the fermionic modes, which differs from the reduced state of the qubits "
                    f"and from what a qubit simulator returns. A matrix cannot carry the gate barriers that "
                    f"make this safe: use reduced_state, partial_trace or matchcake.reduced_state to get the "
                    f"reduced state with its barriers, or pass gate_barrier=False to get the covariance "
                    f"matrix of the fermionic reduced state."
                )
            warnings.warn(CovarianceMatrixToolbox.FERMIONIC_REDUCTION_WARNING, UserWarning, stacklevel=2)
        return CovarianceMatrixToolbox.reduce(self.covariance_matrix, wire_indices)

    def reduced_state(
        self,
        wires: Union[Wires, List[int]],
        *,
        reduced_wires: Optional[Union[Wires, List[int]]] = None,
        gate_barrier: bool = True,
    ) -> GaussianStatePrep:
        r"""
        Keep the given wires of the current state, trace out the others, and return a state preparation.

        The returned :class:`~matchcake.operations.state_preparation.GaussianStatePrep` holds the principal
        submatrix of :attr:`covariance_matrix` on the kept wires and can be queued as the first operation
        of another circuit, typically on a smaller device. Inside a QNode, the measurement
        :func:`matchcake.reduced_state` returns the same state preparation.

        Any set of wires can be kept. The Majorana operators of the kept wires still hold the
        Jordan-Wigner strings of the removed wires, and the returned state remembers where with its
        :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.barriers`:

        - a barrier before a kept wire when a removed wire lies between it and the previous kept wire;
        - a barrier before the first kept wire when wires are removed before it and the state has a
          nonzero displacement vector, such as a product state with superposed qubits;
        - the barriers of the initial state of this device, moved to the kept wires.

        The expectation value of a Pauli word in the returned state is the one of the reduced state of
        the qubits when, for every barrier, the word has an even number of :math:`X` and :math:`Y` on the
        wires on the right of the barrier. A matchgate on one side of a barrier keeps this parity. A
        device started from the returned state therefore gives exact results for the gates on one side of
        a barrier, for the probabilities and the samples in the computational basis and for the
        observables that follow the rule, and it raises a :class:`~pennylane.exceptions.DeviceError` for
        the other gates and observables.

        When the state has a nonzero displacement vector, the returned state preparation holds the part
        of :attr:`displacement_vector` on the kept wires. With a barrier before the first kept wire, the
        observables with an odd number of :math:`X` and :math:`Y` are refused, and the covariance matrix
        is returned alone.

        With ``gate_barrier=False``, the returned state has no barrier, and a warning is raised when it
        would have had one. It is then the reduced state of the fermionic modes: a Pauli word on it stands
        for the same word with a :math:`Z` on every removed wire that its Jordan-Wigner strings cross, and
        nothing checks the gates and the observables any more.

        :param wires: Wires to keep, in the device wire order.
        :type wires: Union[Wires, List[int]]
        :param reduced_wires: Wire labels of the returned state preparation. Defaults to ``range(k)`` for
            ``k`` kept wires.
        :type reduced_wires: Optional[Union[Wires, List[int]]]
        :param gate_barrier: Whether the returned state preparation carries its gate barriers. With
            ``False``, the wires can be listed in any order.
        :type gate_barrier: bool
        :return: The reduced state as a state preparation.
        :rtype: GaussianStatePrep
        :raises ValueError: If a wire is not a wire of the device, if no wire is kept, or if the kept wires
            are not listed in the device wire order while ``gate_barrier`` is ``True``.
        """
        wire_indices = self._reduced_wire_indices(wires, any_order=not gate_barrier)
        reduced_covariance = CovarianceMatrixToolbox.reduce(self.covariance_matrix, wire_indices)
        barriers: Tuple[int, ...] = ()
        if gate_barrier:
            barriers = self._reduced_barriers(wire_indices)
        elif wire_indices != sorted(wire_indices) or self._reduced_barriers(wire_indices):
            warnings.warn(CovarianceMatrixToolbox.FERMIONIC_REDUCTION_WARNING, UserWarning, stacklevel=2)
        reduced_displacement = None
        # With a barrier before the first kept wire, the odd observables are refused and the
        # displacement vector of the kept wires, which holds the strings of the removed wires, is dropped.
        is_displacement_dropped = self._has_displacement() and 0 in barriers
        if self._has_displacement() and not is_displacement_dropped:
            indices = CovarianceMatrixToolbox.majorana_indices(wire_indices)
            reduced_displacement = self.displacement_vector[..., indices]  # type: ignore[index]
        # The mark stays as long as a barrier does: the wires that remain without any barrier are on the
        # left of every barrier, where nothing was dropped.
        odd_part_dropped = bool(barriers) and (is_displacement_dropped or self._is_odd_part_dropped())
        if reduced_wires is None:
            reduced_wires = list(range(len(wire_indices)))
        return GaussianStatePrep(
            reduced_covariance,
            wires=reduced_wires,
            validate=False,
            displacement_vector=reduced_displacement,
            barriers=barriers,
            odd_part_dropped=odd_part_dropped,
        )

    def reset(self) -> None:
        """Reset the device state to its initial configuration.

        :return: None
        """
        self._global_sptm = None
        self._global_sptm_is_shared = False
        self._batched = False
        self._transition_matrix = None
        self._lookup_table = None
        self._star_state = None
        self._star_probability = None
        self._samples = None
        self._samples_bases = {}
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

    def _bases_to_bloch_vectors(self, bases: Mapping[Any, Union[str, Sequence[float]]]) -> np.ndarray:
        """Convert measurement bases keyed by wire into one Bloch vector per device wire.

        :param bases: The measurement basis of some wires, as a Pauli letter or a Bloch vector.
        :type bases: Mapping[Any, Union[str, Sequence[float]]]
        :return: The Bloch vectors in device wire order, shape (num_wires, 3). Wires not listed
            in ``bases`` get the computational basis.
        :rtype: np.ndarray
        :raises ValueError: If a wire is not on the device or a basis is not a Pauli letter or a
            non-zero Bloch vector of length 3.
        """
        assert self.num_wires is not None
        bloch_vectors = ProductBasisSampling.pauli_bloch_vectors(["Z"] * self.num_wires)
        for wire, basis in bases.items():
            if wire not in self.wires:
                raise ValueError(f"The wire {wire} is not on the device, whose wires are {self.wires.tolist()}.")
            if isinstance(basis, str):
                bloch_vector = ProductBasisSampling.pauli_bloch_vectors([basis.upper()])[0]
            else:
                bloch_vector = np.asarray(qml.math.unwrap(basis), dtype=float)
                norm = np.linalg.norm(bloch_vector) if bloch_vector.shape == (3,) else 0.0
                if norm < 1e-12:
                    raise ValueError(f"The basis of wire {wire} must be a non-zero Bloch vector of length 3.")
                bloch_vector = bloch_vector / norm
            bloch_vectors[self.wires.index(wire)] = bloch_vector
        return bloch_vectors

    def _build_sampled_measurement(
        self,
        measurement_type: type[Union[ExpectationMP, SampleMP]],
        observable: Optional[qml.operation.Operator],
    ) -> Union[ExpectationMP, SampleMP]:
        """Build a measurement process that correctly processes the current samples.

        The device does not rotate the samples into the eigenbasis of the observable. When the
        samples were drawn in a product basis (see :meth:`generate_samples`), Pauli observables
        that match that basis are first rewritten with ``Z`` on the rotated wires, by
        :meth:`_observable_in_sampled_frame`. The observable must then be diagonal in the
        computational basis. Diagonal observables whose eigenvalues are not in computational
        basis order (e.g. a diagonal ``qml.Hermitian``) are measured through the diagonal of
        their matrix.

        :param measurement_type: The measurement process class to build.
        :type measurement_type: type[Union[ExpectationMP, SampleMP]]
        :param observable: The observable to estimate from samples, or None when there is none.
        :type observable: Optional[qml.operation.Operator]
        :return: The measurement process to call ``process_samples`` on.
        :rtype: Union[ExpectationMP, SampleMP]
        :raises DeviceError: If the observable is not diagonal in the basis of the samples.
        """
        if observable is not None:
            observable = self._observable_in_sampled_frame(observable)
        if observable is None or self._can_process_samples_directly(observable):
            return measurement_type(obs=observable)
        eigvals = self._diagonal_matrix_eigvals(observable)
        if eigvals is not None:
            return measurement_type(eigvals=eigvals, wires=observable.wires)
        raise DeviceError(
            f"The observable {observable} is not diagonal in the computational basis and cannot be "
            f"estimated from shots on {self.name}. Use analytic mode (shots=None) or an observable "
            f"diagonal in the computational basis."
        )

    def _hermitian_exact_expval(self, observable: qml.Hermitian) -> TensorLike:
        """Compute the exact expectation value of a Hermitian observable.

        A Hermitian matrix ``H`` on k wires decomposes as ``H = sum_P tr(P H) / 2^k P`` over the
        4^k Pauli words ``P``, which the Pauli word strategies evaluate exactly. When the matrix
        is not trainable, the words with a non-zero coefficient are evaluated together as a
        single Pauli sum. The strategies do not propagate gradients through the coefficients of
        a sum, so when the matrix is trainable each word is evaluated separately and weighted
        here instead.

        :param observable: The Hermitian observable.
        :type observable: qml.Hermitian
        :return: The expectation value of the observable, shape (...) for batched circuits.
        :rtype: TensorLike
        """
        matrix = observable.matrix()
        wires = observable.wires
        dimension = 2 ** len(wires)
        is_trainable = qml.math.requires_grad(matrix)
        coeffs, pauli_words = [], []
        for letters in product("IXYZ", repeat=len(wires)):
            pauli_word = qml.pauli.PauliWord({wire: letter for wire, letter in zip(wires, letters) if letter != "I"})
            word_matrix = qml.math.convert_like(pauli_word.to_mat(wire_order=wires), matrix)
            coeff = qml.math.real(qml.math.trace(qml.math.dot(word_matrix, matrix))) / dimension
            # The identity word is always kept so that the decomposition is never empty.
            is_identity = len(pauli_word) == 0
            if not is_identity and not is_trainable and qml.math.allclose(coeff, 0.0):
                continue
            coeffs.append(coeff)
            pauli_words.append(pauli_word.operation(wire_order=wires))
        if not is_trainable:
            return qml.math.real(self.exact_expval(qml.Hamiltonian(coeffs, pauli_words)))
        weighted_terms = []
        for coeff, pauli_word in zip(coeffs, pauli_words):
            word_expval = qml.math.real(self.exact_expval(pauli_word))  # (...)
            weighted_terms.append(utils.math.convert_and_cast_like(coeff, word_expval) * word_expval)
        # The sum broadcasts the unbatched identity term over the batch dimensions.
        return sum(weighted_terms[1:], weighted_terms[0])

    def _check_observable_against_barriers(self, observable: qml.operation.Operator) -> None:
        r"""
        Check that the expectation value of an observable is the one of the reduced state of the qubits.

        With gate barriers on the initial state, a Pauli word is exact when, for every barrier, it has an
        even number of :math:`X` and :math:`Y` on the wires on the right of the barrier. The observables
        that are evaluated term by term (:class:`~matchcake.observables.BatchHamiltonian`,
        :class:`pennylane.Hermitian`) are checked on their terms, and the projectors on computational-basis
        states only hold :math:`Z` operators.

        :param observable: The observable to check.
        :type observable: qml.operation.Operator
        :raises DeviceError: If a Pauli word of the observable breaks the rule of a barrier, or if the
            observable has no Pauli representation to check.
        """
        barriers = self.barriers
        if not barriers:
            return
        if isinstance(observable, (BatchHamiltonian, qml.Hermitian, BasisStateProjector, BatchProjector)):
            return
        pauli_rep = observable.pauli_rep
        if pauli_rep is None:
            raise DeviceError(
                f"The observable {observable} has no Pauli representation, so {self.name} cannot check it "
                f"against the gate barriers of the initial state (before the device positions "
                f"{list(barriers)}). Use an observable made of Pauli words."
            )
        for pauli_word in pauli_rep:
            transverse_positions = [
                self.wires.index(wire) for wire, letter in pauli_word.items() if letter in ("X", "Y")
            ]
            for barrier in barriers:
                if sum(position >= barrier for position in transverse_positions) % 2 == 1:
                    raise DeviceError(
                        f"The Pauli word {pauli_word} of the observable {observable} has an odd number of X "
                        f"and Y on the right of the gate barrier before the device position {barrier} on "
                        f"{self.name}. The initial state is a reduced state: wires were traced out at this "
                        f"barrier, and the expectation value of this word in it is not the one of the "
                        f"reduced state of the qubits."
                    )

    def _check_operation_against_barriers(self, operation: qml.operation.Operator) -> None:
        r"""
        Check that an operation does not act on both sides of a gate barrier of the initial state.

        A matchgate on one side of a barrier keeps the parity of the number of Majorana operators on the
        right of the barrier, on which the exactness of the results relies. A gate with wires on both
        sides does not. A barrier before the first wire of the device has no wire on its left.

        :param operation: The operation to check.
        :type operation: qml.operation.Operator
        :raises DeviceError: If the wires of the operation are on both sides of a barrier.
        """
        barriers = self.barriers
        if not barriers or len(operation.wires) == 0:
            return
        positions = self.wires.indices(operation.wires)
        first_position, last_position = min(positions), max(positions)
        for barrier in barriers:
            if first_position < barrier <= last_position:
                raise DeviceError(
                    f"The operation {operation} acts on both sides of the gate barrier before the device "
                    f"position {barrier} on {self.name}. The initial state is a reduced state: wires were "
                    f"traced out at this barrier, and a gate across it would not be the gate on the qubits "
                    f"but the same gate with a Z on the removed wires."
                )

    def _check_sampling_bases_against_barriers(self, rotated_bases: Mapping[Any, Any]) -> None:
        r"""
        Check that the wires sampled in another basis than the computational one are on the left of
        every gate barrier of the initial state.

        The distribution of samples in the X or Y basis of a wire involves Pauli words with a single
        :math:`X` or :math:`Y` on that wire, which are not exact on the right of a barrier.

        :param rotated_bases: The wires that are not sampled in the computational basis, as keys.
        :type rotated_bases: Mapping[Any, Any]
        :raises DeviceError: If such a wire is on the right of a barrier.
        """
        barriers = self.barriers
        if not barriers or not rotated_bases:
            return
        first_barrier = min(barriers)
        blocked_wires = [wire for wire in rotated_bases if self.wires.index(wire) >= first_barrier]
        if blocked_wires:
            raise DeviceError(
                f"The wires {blocked_wires} cannot be sampled in another basis than the computational one "
                f"on {self.name}: they are on the right of the gate barrier before the device position "
                f"{first_barrier}, left by wires that were traced out of the initial state."
            )

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
            self._samples = self.generate_samples(self._measurement_bases(circuit.measurements))
        results = [self._process_measurement(m) for m in circuit.measurements]
        return results[0] if len(results) == 1 else tuple(results)

    def _measurement_bases(self, measurements: Sequence[qml.measurements.MeasurementProcess]) -> Dict[Any, str]:
        """Find the Pauli basis in which each wire must be sampled for a set of measurements.

        Pauli observables give the Pauli letter of each of their wires. Measurements without an
        observable, and observables without a Pauli representation (such as a diagonal
        ``qml.Hermitian``), need the computational basis on their wires.

        :param measurements: The measurements of a circuit.
        :type measurements: Sequence[qml.measurements.MeasurementProcess]
        :return: The Pauli letter of the wires not sampled in the computational basis.
        :rtype: Dict[Any, str]
        :raises DeviceError: If a wire needs two different bases, i.e. the measurements do not
            commute qubit-wise.
        """
        letters_by_wire: Dict[Any, set] = defaultdict(set)
        for measurement in measurements:
            observable = measurement.obs
            pauli_rep = None if observable is None else observable.pauli_rep
            if pauli_rep is None:
                wires = measurement.wires if observable is None else observable.wires
                for wire in wires or self.wires:
                    letters_by_wire[wire].add("Z")
                continue
            for pauli_word in pauli_rep:
                for wire, letter in pauli_word.items():
                    letters_by_wire[wire].add(letter)
        conflicts = {wire: sorted(letters) for wire, letters in letters_by_wire.items() if len(letters) > 1}
        if conflicts:
            raise DeviceError(
                f"The measurements need several bases on the same wires ({conflicts}) and cannot be estimated "
                f"from the same shots on {self.name}. Measure observables that commute qubit-wise."
            )
        return {wire: letters.pop() for wire, letters in letters_by_wire.items() if letters != {"Z"}}

    def _has_displacement(self) -> bool:
        r"""
        Tell whether the current state has a nonzero displacement vector.

        A matchgate circuit rotates the displacement vector by an orthogonal matrix, so the current state
        has a displacement exactly when the initial state has one.

        :return: ``True`` for an initial state with a nonzero displacement vector.
        :rtype: bool
        """
        state_prep = self.state_prep_op
        return isinstance(state_prep, GaussianStatePrepBase) and state_prep.is_displaced()

    def _is_odd_part_dropped(self) -> bool:
        r"""
        Tell whether the current state stands for a state whose displacement vector was dropped.

        :return: The mark of the initial state, see
            :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.odd_part_dropped`.
        :rtype: bool
        """
        state_prep = self.state_prep_op
        return isinstance(state_prep, GaussianStatePrepBase) and state_prep.odd_part_dropped

    def _has_barrier_inside(self, wire_indices: Sequence[int]) -> bool:
        r"""
        Tell whether a removed wire or a gate barrier of the current state lies between the given wires.

        :param wire_indices: Positions of the kept wires in the device wire order.
        :type wire_indices: Sequence[int]
        :return: ``True`` when the reduced state on these wires needs a barrier between two of its wires,
            or when the wires are not listed in increasing order.
        :rtype: bool
        """
        positions = [int(position) for position in wire_indices]
        has_gap = not CovarianceMatrixToolbox.are_wires_contiguous(positions)
        return has_gap or any(min(positions) < barrier <= max(positions) for barrier in self.barriers)

    def _observable_in_sampled_frame(self, observable: qml.operation.Operator) -> qml.operation.Operator:
        """Rewrite an observable for samples drawn in a product basis.

        On the wires sampled in a rotated basis, the samples are the bits of the measured
        Pauli operator. A Pauli observable that uses that Pauli on those wires is therefore
        processed as the same observable with ``Z`` on them.

        :param observable: The observable to rewrite.
        :type observable: qml.operation.Operator
        :return: The observable to process the samples with. It is unchanged when it does not
            act on a rotated wire.
        :rtype: qml.operation.Operator
        :raises DeviceError: If the observable acts on a rotated wire with another operator than
            the Pauli measured there.
        """
        if not any(wire in self._samples_bases for wire in observable.wires):
            return observable
        pauli_rep = observable.pauli_rep
        error = DeviceError(
            f"The observable {observable} does not match the bases in which the samples were drawn on {self.name}."
        )
        if pauli_rep is None:
            raise error
        sentence = qml.pauli.PauliSentence()
        for pauli_word, coeff in pauli_rep.items():
            diagonal_word = {}
            for wire, letter in pauli_word.items():
                letter_bloch_vector = ProductBasisSampling.PAULI_BLOCH_VECTORS[letter]
                sampled_bloch_vector = self._samples_bases.get(wire, ProductBasisSampling.PAULI_BLOCH_VECTORS["Z"])
                if not np.allclose(letter_bloch_vector, sampled_bloch_vector):
                    raise error
                diagonal_word[wire] = "Z"
            sentence[qml.pauli.PauliWord(diagonal_word)] = coeff
        return sentence.operation(wire_order=observable.wires)

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
            samples = self._samples_in_pennylane_layout()  # (..., shots, num_wires)
            if measurement.obs is None:
                if not measurement.wires:
                    return samples
                # samples: (..., shots, num_wires) -> (..., shots, len(measurement.wires))
                return samples[..., self.wires.indices(measurement.wires)]
            mp = self._build_sampled_measurement(SampleMP, measurement.obs)
            return mp.process_samples(samples, wire_order=self.wires)
        if isinstance(measurement, (CovarianceMatrixMP, ReducedStateMP, VnEntropyMP, PurityMP, DensityMatrixMP)):
            return self._process_reduced_state_measurement(measurement)
        raise NotImplementedError(f"Measurement {type(measurement).__name__} is not supported by {self.name}.")

    def _process_reduced_state_measurement(self, measurement: qml.measurements.MeasurementProcess) -> Any:
        """Evaluate a measurement of the reduced state on the wires of the measurement.

        The covariance matrix is computed by :meth:`reduced_covariance_matrix` and the state preparation
        by :meth:`reduced_state`. The PennyLane measurements are quantities of the reduced state of the
        qubits, which the covariance matrix gives on a contiguous block of wires without gate barrier.

        :param measurement: A covariance matrix, reduced state, Von Neumann entropy, purity or density
            matrix measurement.
        :type measurement: qml.measurements.MeasurementProcess
        :return: The measurement result.
        :rtype: Any
        :raises DeviceError: If the circuit is executed with shots, since these measurements are analytic,
            or if an entropy, a purity or a density matrix is asked on wires with a gate barrier of the
            initial state between two of them, or on wires that are not all on the left of every barrier
            when a displacement vector was dropped.
        :raises NotImplementedError: If an entropy, a purity or a density matrix is asked while the initial
            state has a nonzero displacement vector, since the covariance matrix does not determine the
            state.
        """
        if self._active_shots is not None:
            raise DeviceError(
                f"Measurement {type(measurement).__name__} is analytic and cannot be estimated from shots "
                f"on {self.name}. Execute the circuit without shots."
            )
        wires = measurement.wires or self.wires
        if isinstance(measurement, ReducedStateMP):
            return self.reduced_state(
                wires, reduced_wires=measurement.reduced_wires, gate_barrier=measurement.gate_barrier
            )
        if isinstance(measurement, CovarianceMatrixMP):
            return self.reduced_covariance_matrix(wires, gate_barrier=measurement.gate_barrier)
        wire_indices = self._reduced_wire_indices(wires, any_order=True)
        # When a displacement vector was dropped, the wires on the right of a barrier may have lost their
        # odd part, so every barrier that is not on the right of the measured wires is refused.
        first_refused_position = 0 if self._is_odd_part_dropped() else min(wire_indices) + 1
        crossed_barriers = [
            barrier
            for barrier in self.barriers
            if barrier == 0 or first_refused_position <= barrier <= max(wire_indices)
        ]
        if crossed_barriers:
            raise DeviceError(
                f"Measurement {type(measurement).__name__} on the device positions {wire_indices} cannot be "
                f"evaluated on {self.name}: the initial state {self.state_prep_op} has gate barriers before "
                f"the positions {crossed_barriers}, left by wires that were traced out. The reduced state "
                f"of the qubits is then not determined by the covariance matrix of these wires."
            )
        reduced_covariance = self.reduced_covariance_matrix(wires)
        if self._has_displacement():
            raise NotImplementedError(
                f"Measurement {type(measurement).__name__} is only implemented for initial states without "
                f"displacement: computational-basis states and GaussianStatePrep without a displacement "
                f"vector. A state with a nonzero displacement vector, such as a product state with "
                f"superposed qubits, is not determined by its covariance matrix. Got {self.state_prep_op} "
                f"on {self.name}."
            )
        if isinstance(measurement, VnEntropyMP):
            return CovarianceMatrixToolbox.von_neumann_entropy(reduced_covariance, base=measurement.log_base)
        if isinstance(measurement, PurityMP):
            return CovarianceMatrixToolbox.purity(reduced_covariance)
        density_matrix = CovarianceMatrixToolbox.to_density_matrix(reduced_covariance)
        return qml.math.cast(density_matrix, self._c_dtype_name)

    def _reduced_barriers(self, wire_indices: Sequence[int]) -> Tuple[int, ...]:
        r"""
        Gate barriers of the reduced state on the given wire positions, in the order of the kept wires.

        With the kept positions :math:`k_0 < k_1 < \dots`, the reduced state has:

        - a barrier before the kept wire :math:`i \geq 1` when a removed wire lies between
          :math:`k_{i-1}` and :math:`k_i`, since the Majorana operators of the wires :math:`k_j` with
          :math:`j \geq i` hold the Jordan-Wigner string of that removed wire;
        - every barrier :math:`p` of the current state, moved before the first kept wire at a position
          :math:`\geq p`. It is dropped when no kept wire is on its right. When no kept wire is on its
          left, it becomes a barrier before the first kept wire, which is only needed when it was one
          already, when the state has a nonzero displacement vector, or when a displacement vector was
          dropped earlier (:attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.odd_part_dropped`):
          otherwise the Pauli words with an odd number of :math:`X` and :math:`Y` vanish in the reduced
          state of the qubits as well;
        - a barrier before the first kept wire when wires are removed before it and the state has a
          nonzero displacement vector.

        :param wire_indices: Positions of the kept wires in the device wire order, in increasing order.
        :type wire_indices: Sequence[int]
        :return: The sorted positions of the barriers of the reduced state.
        :rtype: Tuple[int, ...]
        """
        positions = sorted(int(position) for position in wire_indices)
        is_displaced = self._has_displacement()
        has_odd_part = is_displaced or self._is_odd_part_dropped()
        barriers = {index for index in range(1, len(positions)) if positions[index] - positions[index - 1] > 1}
        for barrier in self.barriers:
            n_kept_before = sum(position < barrier for position in positions)
            if n_kept_before == len(positions):
                continue
            if n_kept_before > 0 or barrier == 0 or has_odd_part:
                barriers.add(n_kept_before)
        if positions[0] > 0 and is_displaced:
            barriers.add(0)
        return tuple(sorted(barriers))

    def _reduced_wire_indices(self, wires: Union[Wires, List[int]], any_order: bool = False) -> List[int]:
        r"""
        Positions of the wires to keep in the device wire order, checked for a reduced state.

        :param wires: Wires to keep.
        :type wires: Union[Wires, List[int]]
        :param any_order: Whether the kept wires can be listed in another order than the one of the device.
        :type any_order: bool
        :return: The positions of the kept wires, in the order they are given.
        :rtype: List[int]
        :raises ValueError: If a wire is not a wire of the device, if no wire is kept, if a wire is
            repeated, or if the kept wires are not listed in the device wire order while ``any_order`` is
            ``False``.
        """
        kept_wires = Wires(wires)
        unknown_wires = [wire for wire in kept_wires if wire not in self.wires]
        if unknown_wires:
            raise ValueError(f"Wires {unknown_wires} are not wires of the device ({self.wires.tolist()}).")
        wire_indices = [int(index) for index in self.wires.indices(kept_wires)]
        CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices, allow_gaps=True)
        if not any_order and wire_indices != sorted(wire_indices):
            raise ValueError(
                f"The kept wires {kept_wires.tolist()} must be listed in the device wire order "
                f"({self.wires.tolist()}). Reordering fermionic modes is not a relabeling of qubits, and a "
                f"reduced state with gate barriers is only defined in the device wire order. Pass "
                f"gate_barrier=False to get the reduced state of the fermionic modes in this order."
            )
        return wire_indices

    def _samples_in_pennylane_layout(self) -> np.ndarray:
        """Return the current samples with the axes in the order PennyLane expects.

        The sampling strategies return samples of shape (shots, ..., num_wires), with the batch
        axes of a batched circuit after the shots. PennyLane processes samples of shape
        (..., shots, num_wires), with the batch axes first.

        :return: The samples, shape (..., shots, num_wires).
        :rtype: np.ndarray
        :raises ValueError: If no samples have been generated.
        """
        if self._samples is None:
            raise ValueError("No samples have been generated. The number of shots must be specified.")
        return np.moveaxis(np.asarray(self._samples), 0, -2)

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
    def barriers(self) -> Tuple[int, ...]:
        r"""
        Gate barriers of the current state, as positions in the device wire order.

        They are the barriers of the initial state, see
        :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.barriers`. A barrier at
        position :math:`p` sits just before the wire at position :math:`p`.

        :return: The sorted positions of the barriers, empty for a state without barrier.
        :rtype: Tuple[int, ...]
        """
        state_prep = self.state_prep_op
        return state_prep.barriers if isinstance(state_prep, GaussianStatePrepBase) else ()

    @property
    def covariance_matrix(self) -> TensorLike:
        """Majorana covariance matrix evolved under the current global SPTM.

        :return: The covariance matrix, shape (..., 2n, 2n).
        :rtype: TensorLike
        """
        if not isinstance(self.state_prep_op, GaussianStatePrepBase):
            raise ValueError(
                f"Covariance matrix can only be computed for product states and Gaussian states. "
                f"Got {type(self.state_prep_op)} instead of a GaussianStatePrepBase."
            )
        cov0 = self.state_prep_op.covariance_matrix
        sptm = self.global_sptm.sptm_matrix(self.wires)
        return qml.math.einsum("...ij,...ik,...kl->...jl", sptm, cov0, sptm)

    @property
    def displacement_vector(self) -> TensorLike:
        r"""Majorana displacement vector :math:`d_\mu = \langle c_\mu \rangle` evolved under the current global SPTM.

        It is :math:`R^\top d_0`, with :math:`d_0` the displacement vector of the initial state and
        :math:`R` the global SPTM. It vanishes for a state with a definite fermion parity.

        :return: The displacement vector, shape (..., 2n).
        :rtype: TensorLike
        """
        if not isinstance(self.state_prep_op, GaussianStatePrepBase):
            raise ValueError(
                f"Displacement vector can only be computed for product states and Gaussian states. "
                f"Got {type(self.state_prep_op)} instead of a GaussianStatePrepBase."
            )
        sptm = self.global_sptm.sptm_matrix(self.wires)
        # The cast keeps the backend of the displacement vector, so gradients flow through it.
        d0 = qml.math.cast(self.state_prep_op.displacement_vector, self._r_dtype_name)
        return qml.math.einsum("...ij,...i->...j", sptm, d0)

    @property
    def extended_covariance_matrix(self) -> TensorLike:
        """Extended (2n+1)x(2n+1) covariance matrix tilde_Lambda.

        Lifts the standard covariance matrix by appending a displacement
        column/row at position 2n (the parity index). Evolves the full
        extended state under the matchgate SPTM as tilde_Q^T tilde_Lambda_0 tilde_Q.

        BasisState inputs are promoted to ProductState first (displacement is zero). A
        :class:`~matchcake.operations.state_preparation.GaussianStatePrep` has no displacement unless it
        was given a displacement vector.

        See docs/expectation_values_theory.md for the math.
        """
        state_prep = self.state_prep_op
        if isinstance(state_prep, BasisState):
            state_prep = ProductState.from_basis_state(state_prep)
        if not isinstance(state_prep, GaussianStatePrepBase):
            raise ValueError(
                f"Extended covariance matrix requires a ProductState, BasisState or GaussianStatePrep. "
                f"Got {type(state_prep)}."
            )
        # Build tilde_Lambda_0 from the initial state. Cast to the device's configured
        # precision so r_dtype/c_dtype control the whole m-Pfaffian pipeline.
        cov0 = qml.math.cast(state_prep.covariance_matrix, self._r_dtype_name)  # (2n, 2n)
        d0 = state_prep.displacement_vector  # (2n,), cast to the precision of cov0 below
        ext_cov0 = _extended_covariance_matrix(cov0, d0)  # (2n+1, 2n+1)

        # Lift SPTM: \tilde{R} = R ⊕ 1
        sptm = self.global_sptm.sptm_matrix(self.wires)  # (2n, 2n)
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
        self._global_sptm_is_shared = True
        return self._global_sptm

    @global_sptm.setter
    def global_sptm(self, value: Union[Operation, TensorLike]) -> None:
        if isinstance(value, Operation):
            value = SingleParticleTransitionMatrixOperation.from_operation(value)
        if not isinstance(value, SingleParticleTransitionMatrixOperation):
            value = SingleParticleTransitionMatrixOperation(value, wires=self.wires)
        matrix = qml.math.cast(value.sptm_matrix(), self._r_dtype_name)
        self._global_sptm = SingleParticleTransitionMatrixOperation(matrix, wires=value.wires, **value._hyperparameters)
        self._global_sptm_is_shared = True
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
        """The samples from the last execution, in the bases given to :meth:`generate_samples`.

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
            self._transition_matrix = utils.make_transition_matrix_from_action_matrix(self.global_sptm.sptm_matrix())
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value: Optional[TensorLike]) -> None:
        if self._global_sptm is not None and value is not None:
            global_sptm_matrix = self._global_sptm.sptm_matrix()
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
