from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.measurements import ExpectationMP
from pennylane.operation import Operator
from pennylane.ops.op_math import Prod, SProd, Sum
from pennylane.ops.qubit import Projector
from pennylane.tape import QuantumScript
from pennylane.typing import PostprocessingFn, ResultBatch

from .. import utils
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..typing import TensorLike

_SINGLE_PRECISION = (torch.float32, torch.complex64)


def live_observable(observable: Operator, r_dtype: Optional[torch.dtype] = None) -> Operator:
    """Rebuild an observable that PennyLane would read stale, as a fresh Hamiltonian read live.

    ``split_non_commuting`` reads a composite observable through ``Sum.terms()``, which serves the
    ``pauli_rep`` PennyLane cached when the observable was built. That cache goes stale after an
    in-place coefficient update, ``H + H`` adds into its shared coefficient tensors in place, and
    the simplification fallback of ``Sum.terms()`` accumulates into the caller's own coefficient
    tensor on every call. Such an observable (a sum, scalar product, product, adjoint or power at
    the top level) is rebuilt from :func:`~matchcake.utils.pauli_decomposition.live_pauli_terms`,
    whose coefficients are computed from the caller's, so gradients with respect to them still flow.
    A part without a Pauli representation, such as a projector, is reduced there as ``Sum.terms()``
    reduces it, so that ``P @ I`` is measured as ``P``. The coefficients are given the precision of
    a single-precision device, see :func:`_in_working_precision`.

    A Hamiltonian is read through ``LinearCombination.terms()``, which already returns its own
    coefficients as they are, and each of its terms is measured as written and read live by the
    device's expval strategies, so it keeps its path. With grouping indices, though, the splitter
    accumulates repeated words into the coefficient tensor in place, so a grouped Hamiltonian gets a
    fresh copy of its coefficients, of the same type and precision, with the same terms and the
    same grouping. A Hamiltonian term that carries a trainable scalar of its own, such as
    ``w * Z(0)``, is the exception: the device's expval strategies read that scalar as a constant,
    so it is folded into the Hamiltonian coefficient here, see :func:`_has_trainable_term_scalar`.

    A grouped sum is rebuilt by :func:`_live_grouped_sum`, which keeps its grouping.

    Also returned unchanged, so that it takes exactly the path it took before: a plain Pauli word; a
    scalar multiple of a Hamiltonian or of a word, which ``SProd.terms()`` reads through the base's
    own ``terms()``; one with no Pauli part; and one reducing to a multiple of the identity or to
    nothing, which ``split_non_commuting`` treats as an offset.

    :param observable: The observable of a measurement.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The observable to measure instead.
    :rtype: Operator
    """
    if isinstance(observable, qml.ops.LinearCombination):
        if len(observable.ops) == 0:
            return observable
        if observable.grouping_indices is not None:
            return _live_grouped_hamiltonian(observable, r_dtype)
    elif getattr(observable, "grouping_indices", None) is not None:
        return _live_grouped_sum(observable, r_dtype)
    if _is_read_live(observable) and not _has_trainable_term_scalar(observable):
        return _with_array_coefficients(observable, r_dtype)
    coefficients, operators = utils.live_pauli_terms(
        observable, keep_non_pauli=True, keep_trainable=_is_measurable_term
    )
    if not operators or any(operator is observable for operator in operators) or _is_identity_only(operators):
        return observable
    return qml.Hamiltonian(_in_working_precision(coefficients, observable, r_dtype), operators)


def _with_array_coefficients(observable: Operator, r_dtype: Optional[torch.dtype] = None) -> Operator:
    """Rebuild an observable read live whose coefficients are all python numbers, with a numpy array of them.

    PennyLane multiplies each coefficient into the torch value measured for its term, and torch
    reads a python float in its default precision, single unless changed: ``Hamiltonian([0.1], ...)``
    and ``0.1 * Z(0)`` would be off by a relative ``1e-8`` on a double-precision device. A numpy
    array keeps its precision. The terms are kept as written.

    :param observable: A plain Pauli word, a Hamiltonian, or a scalar multiple of either.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The Hamiltonian to measure instead, or the observable itself when it carries no
        coefficient, a coefficient that is not a python number, or only identity terms, or is a
        ``BatchHamiltonian``, which the device evaluates as one observable.
    :rtype: Operator
    """
    if not isinstance(observable, (qml.ops.LinearCombination, SProd)) or isinstance(observable, BatchHamiltonian):
        return observable
    coefficients, operators = observable.terms()
    if not operators or _is_identity_only(operators) or not _are_python_numbers(coefficients):
        return observable
    return qml.Hamiltonian(_in_working_precision(np.array(coefficients), observable, r_dtype), operators)


def _are_python_numbers(coefficients: TensorLike) -> bool:
    """Whether coefficients are a list or tuple of plain python numbers.

    :param coefficients: The coefficients of an observable.
    :return: ``True`` for a list or tuple holding only ``int``, ``float`` or ``complex`` values.
    :rtype: bool
    """
    return isinstance(coefficients, (list, tuple)) and all(
        type(value) in (int, float, complex) for value in coefficients
    )


def _live_grouped_hamiltonian(observable: Operator, r_dtype: Optional[torch.dtype] = None) -> Operator:
    """Rebuild a grouped Hamiltonian with fresh coefficients, the same number of terms and the same grouping.

    The coefficients are copied as they are, python numbers into a numpy array, see
    :func:`_with_array_coefficients`. When a term carries a trainable scalar of its own, every term
    that is a multiple of one Pauli word has its factor folded into the coefficients instead, see
    :func:`_single_word`, one term staying one term so that the grouping indices still refer to
    the right ones.

    :param observable: A Hamiltonian with grouping indices and at least one term.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The Hamiltonian to measure instead.
    :rtype: Operator
    """
    operators = list(observable.ops)
    if not any(_holds_trainable_scalar(operator) for operator in operators):
        coefficients = _copied_coefficients(observable.coeffs)
        if _are_python_numbers(coefficients):
            coefficients = _in_working_precision(np.array(coefficients), observable, r_dtype)
    else:
        values = []
        for index, coefficient in enumerate(observable.coeffs):
            factor, operators[index] = _single_word(operators[index])
            values.append(utils.read_coefficient(coefficient) * factor)
        coefficients = _in_working_precision(utils.stacked_coefficients(values), observable, r_dtype)
    rebuilt = qml.Hamiltonian(coefficients, operators)
    rebuilt.grouping_indices = observable.grouping_indices
    return rebuilt


def _single_word(operator: Operator) -> Tuple[Any, Operator]:
    """``(factor, word)`` of a Hamiltonian term that is a multiple of one Pauli word, read live.

    :param operator: The term.
    :return: Its factor and its word. A term that expands into several words, or has no Pauli
        representation, is returned whole with a factor of one, after the scalar products at its
        top are taken off.
    :rtype: Tuple[Any, Operator]
    """
    try:
        coefficients, operators = utils.live_pauli_terms(operator)
    except TypeError:
        operators = []
    if len(operators) == 1:
        return utils.read_coefficient(coefficients[0]), operators[0]
    factor: Any = 1.0
    while isinstance(operator, SProd):
        factor = factor * utils.read_coefficient(operator.scalar)
        operator = operator.base
    return factor, operator


def _has_trainable_term_scalar(observable: Operator) -> bool:
    """Whether a Hamiltonian, or a scalar multiple of one, has a term carrying a trainable scalar.

    ``split_non_commuting`` measures each term of a Hamiltonian as one observable, and the device's
    expval strategies read a scalar inside it as a constant, so ``Hamiltonian([0.5], [w * Z(0)])``
    would give ``w`` no gradient. Such a Hamiltonian is rebuilt with the scalar folded into its
    coefficients.

    :param observable: The observable.
    :return: ``True`` if a term of the underlying Hamiltonian holds a trainable torch scalar.
    :rtype: bool
    """
    base = observable
    while isinstance(base, SProd):
        base = base.base
    if not isinstance(base, qml.ops.LinearCombination):
        return False
    return any(_holds_trainable_scalar(operator) for operator in base.ops)


def _holds_trainable_scalar(operator: Operator) -> bool:
    """Whether a trainable torch scalar appears anywhere in an operator's expression tree.

    :param operator: The operator.
    :return: ``True`` if a scalar-product scalar or a Hamiltonian coefficient in it requires a gradient.
    :rtype: bool
    """
    visited: Set[int] = set()
    pending = [operator]
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, qml.ops.LinearCombination):
            values = node.coeffs if isinstance(node.coeffs, (list, tuple)) else [node.coeffs]
            if any(_is_trainable(value) for value in values):
                return True
            pending.extend(node.ops)
        elif isinstance(node, SProd):
            if _is_trainable(node.scalar):
                return True
            pending.append(node.base)
        elif hasattr(node, "operands"):
            pending.extend(node.operands)
        elif hasattr(node, "base"):
            pending.append(node.base)
    return False


def _is_trainable(value: TensorLike) -> bool:
    """Whether a coefficient is a torch tensor that requires a gradient.

    :param value: The coefficient.
    :return: ``True`` for a torch tensor with ``requires_grad``.
    :rtype: bool
    """
    return isinstance(value, torch.Tensor) and value.requires_grad


def _live_grouped_sum(observable: Operator, r_dtype: Optional[torch.dtype] = None) -> Operator:
    """Rebuild a grouped sum as a Hamiltonian with the same grouping and coefficients read live.

    The grouping indices of a sum refer to the terms of its cached ``pauli_rep``, in that order, so
    the Hamiltonian lists the same words in the same order and takes the grouping as it is. Only
    the words of the cache are read, never its coefficients, which are the stale part. A word the
    live expansion holds but the cache does not, because its coefficients cancelled when the sum was
    built, is appended in a group of its own, so that it is measured if an update made it nonzero.

    :param observable: A sum with grouping indices.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The Hamiltonian to measure instead, or the sum itself when it has no Pauli terms.
    :rtype: Operator
    """
    cached_words = list(observable.pauli_rep.keys()) if observable.pauli_rep else []
    if not cached_words:
        return observable
    sentence = utils.live_pauli_sentence(observable)
    known_words = set(cached_words)
    extra_words = [word for word in sentence if word not in known_words]
    words = cached_words + extra_words
    with qml.QueuingManager.stop_recording():
        operators = [word.operation() for word in words]
    coefficients = utils.stacked_coefficients([sentence.get(word, 0.0) for word in words])
    rebuilt = qml.Hamiltonian(_in_working_precision(coefficients, observable, r_dtype), operators)
    groups = [tuple(group) for group in observable.grouping_indices]
    groups.extend((index,) for index in range(len(cached_words), len(words)))
    rebuilt.grouping_indices = tuple(groups)
    return rebuilt


def _is_measurable_term(operator: Operator) -> bool:
    """Whether the device measures a single term: a plain Pauli word, a projector or a Hermitian matrix.

    A term whose trainable coefficient is zero is kept only in that case, so that the coefficient
    gets its gradient. Any other term of zero weight, such as a Hadamard, is dropped as PennyLane
    drops it, since measuring it would raise.

    :param operator: The operator of the term.
    :return: ``True`` for a plain Pauli word, a projector or a Hermitian matrix.
    :rtype: bool
    """
    return isinstance(operator, (Projector, qml.Hermitian)) or utils.has_live_terms(operator)


def _is_read_live(observable: Operator) -> bool:
    """Whether ``split_non_commuting`` already reads an observable live.

    A plain Pauli word carries no coefficient, and ``SProd.terms()`` multiplies its scalar, read
    live, into the base's own ``terms()``, which for a Hamiltonian returns its coefficients and
    terms as they are. Flattening such a Hamiltonian into words would also put terms that share a
    wire on separate tapes, where the device measures a composite term as one.

    :param observable: The observable.
    :return: ``True`` for a plain Pauli word, or a scalar multiple of a Hamiltonian or of a word.
    :rtype: bool
    """
    base = observable
    while isinstance(base, SProd):
        base = base.base
    return isinstance(base, qml.ops.LinearCombination) or utils.has_live_terms(base)


def _in_working_precision(coefficients: TensorLike, observable: Operator, r_dtype: Optional[torch.dtype]) -> TensorLike:
    """Rebuilt coefficients in the precision of a single-precision device.

    The expansion computes in double precision, and PennyLane multiplies each coefficient into the
    value measured for its term, so a double-precision coefficient would make a single-precision
    device return double precision. On such a device, numpy coefficients are cast to single
    precision, and so are torch ones when every torch coefficient of the observable is single
    precision, as PennyLane would have computed them. A double-precision device keeps them all in
    double precision: PennyLane would have kept each constant next to a single-precision weight in
    its own precision, which one stacked array cannot.

    :param coefficients: The stacked coefficients of the rebuilt observable.
    :param observable: The observable they were read from.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The coefficients to build the Hamiltonian with.
    :rtype: TensorLike
    """
    if r_dtype != torch.float32:
        return coefficients
    if isinstance(coefficients, torch.Tensor):
        dtypes = _torch_coefficient_dtypes(observable)
        if dtypes and all(dtype in _SINGLE_PRECISION for dtype in dtypes):
            return coefficients.to(torch.complex64 if coefficients.is_complex() else torch.float32)
        return coefficients
    if type(coefficients) is np.ndarray:
        return coefficients.astype(np.complex64 if np.iscomplexobj(coefficients) else np.float32)
    return coefficients


def _torch_coefficient_dtypes(observable: Operator) -> Set[torch.dtype]:
    """Dtypes of the torch coefficients appearing anywhere in an observable's expression tree.

    :param observable: The observable.
    :return: The dtypes of its Hamiltonian coefficients and scalar-product scalars held as torch tensors.
    :rtype: Set[torch.dtype]
    """
    dtypes: Set[torch.dtype] = set()
    visited: Set[int] = set()
    pending = [observable]
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, qml.ops.LinearCombination):
            values = node.coeffs if isinstance(node.coeffs, (list, tuple)) else [node.coeffs]
            dtypes.update(value.dtype for value in values if isinstance(value, torch.Tensor))
            pending.extend(node.ops)
        elif isinstance(node, SProd):
            if isinstance(node.scalar, torch.Tensor):
                dtypes.add(node.scalar.dtype)
            pending.append(node.base)
        elif hasattr(node, "operands"):
            pending.extend(node.operands)
        elif hasattr(node, "base"):
            pending.append(node.base)
    return dtypes


def _copied_coefficients(coefficients: TensorLike) -> TensorLike:
    """A copy of a Hamiltonian's coefficients, of the same type and precision, sharing no array with them.

    :param coefficients: A torch tensor, a numpy or pennylane.numpy array, or a list of scalars.
    :return: The copy. A python scalar, and an array that cannot change in place, are kept as they are.
    :rtype: TensorLike
    """
    if isinstance(coefficients, torch.Tensor):
        return coefficients.clone()
    if isinstance(coefficients, np.ndarray):
        return coefficients.copy()
    if isinstance(coefficients, (list, tuple)):
        return [_copied_coefficients(value) for value in coefficients]
    return coefficients


def _is_identity_only(operators: List[Operator]) -> bool:
    """Whether every operator of a term list is an identity.

    :param operators: The operators.
    :return: ``True`` if there is at least one and all are identities.
    :rtype: bool
    """
    return bool(operators) and all(isinstance(operator, qml.Identity) for operator in operators)


def _unit_coefficient_candidates(tape: QuantumScript) -> Dict[int, Tuple[Any, Any]]:
    """Trainable coefficients that the postprocessing of ``split_non_commuting`` may leave out of a result.

    PennyLane returns the raw result of a measurement whose observable has one term with a
    coefficient equal to one and an identity offset equal to zero, so neither enters the autograd
    graph: a weight initialised at one, or an offset initialised at zero, would get no gradient.
    Whether it does depends on how the splitter merges repeated terms, which differs with and
    without grouping, so the result itself tells, see :func:`_is_raw_result`. This has to be read
    before the splitting, which accumulates repeated terms into the coefficients in place.

    :param tape: The circuit about to be handed to ``split_non_commuting``.
    :return: ``(coefficient, offset)`` by measurement index: the summed coefficient of the single
        non-identity operator and the summed identity coefficient, for each expectation value of a
        Hamiltonian, or of a scalar multiple of one or of a word, where either is trainable.
    :rtype: Dict[int, Tuple[Any, Any]]
    """
    candidates = {}
    for index, measurement in enumerate(tape.measurements):
        observable = measurement.obs
        if not isinstance(measurement, ExpectationMP) or not isinstance(observable, (qml.ops.LinearCombination, SProd)):
            continue
        if not isinstance(observable, qml.ops.LinearCombination) and not _is_read_live(observable):
            continue
        offset: Any = 0
        total: Any = 0
        hashes = set()
        for coefficient, operator in zip(*observable.terms()):
            if isinstance(operator, qml.Identity):
                offset = offset + coefficient
            else:
                total = total + coefficient
                hashes.add(operator.hash)
        if len(hashes) == 1 and (_is_trainable(total) or _is_trainable(offset)):
            candidates[index] = (total, offset)
    return candidates


def _is_raw_result(output: TensorLike, results: ResultBatch) -> bool:
    """Whether a postprocessed output is one of the executed results itself, coefficients left out.

    :param output: The output of PennyLane's postprocessing for one measurement.
    :param results: The results of the executed circuits.
    :return: ``True`` if the output is the very object of one of the results.
    :rtype: bool
    """
    pending = [results]
    while pending:
        node = pending.pop()
        if node is output:
            return True
        if isinstance(node, (list, tuple)):
            pending.extend(node)
    return False


def _identity_offset(observable: Operator) -> Optional[torch.Tensor]:
    """The torch coefficient of an observable that is a multiple of the identity alone.

    ``split_non_commuting`` measures nothing for such an observable and returns its coefficient
    times a numpy array of ones, which numpy cannot multiply by a torch tensor. The coefficient is
    read by the live expansion: ``terms()`` of a product simplifies it, which changes how PennyLane
    splits it afterwards.

    :param observable: The observable of an expectation value.
    :return: The summed coefficient of its identity terms, real when its imaginary part is at
        round-off level, when every term is an identity and the sum is a one-element torch tensor;
        ``None`` otherwise.
    :rtype: Optional[torch.Tensor]
    """
    if not isinstance(observable, (Sum, Prod, SProd)):
        return None
    try:
        sentence = utils.live_pauli_sentence(observable)
    except TypeError:
        return None
    if not sentence or any(len(word) > 0 for word in sentence):
        return None
    offset: Any = 0
    for coefficient in sentence.values():
        offset = offset + coefficient
    if not isinstance(offset, torch.Tensor) or offset.numel() != 1:
        return None
    return utils.stacked_coefficients([offset]).reshape(())


def _like_precision(dtype: torch.dtype, r_dtype: torch.dtype) -> torch.dtype:
    """The real or complex dtype of the precision of a device.

    :param dtype: The dtype of a result, real or complex.
    :param r_dtype: The real dtype of the device.
    :return: ``r_dtype`` for a real result, the complex dtype of the same precision for a complex one.
    :rtype: torch.dtype
    """
    if not dtype.is_complex:
        return r_dtype
    return torch.complex64 if r_dtype == torch.float32 else torch.complex128


def _restored(result: TensorLike, coefficient: TensorLike, offset: TensorLike) -> TensorLike:
    """``coefficient * result + offset`` in the precision of the result, for a unit coefficient and a zero offset.

    The value is unchanged; the coefficient and the offset enter the autograd graph.

    :param result: The raw result PennyLane returned for the measurement.
    :param coefficient: The coefficient of its single term, equal to one.
    :param offset: Its identity offset, equal to zero.
    :return: The same value, differentiable with respect to both.
    :rtype: TensorLike
    """
    result = qml.math.asarray(result, like="torch")
    values = []
    for value in (coefficient, offset):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.reshape(())
            if not value.is_complex():
                value = value.to(dtype=result.real.dtype, device=result.device)
        values.append(value)
    return result * values[0] + values[1]


@qml.transform
def nif_split_non_commuting(
    tape: QuantumScript, r_dtype: Optional[torch.dtype] = None
) -> Tuple[List[QuantumScript], PostprocessingFn]:
    """Pass BatchHamiltonian measurements through unchanged; otherwise delegate to split_non_commuting.

    Composite expectation-value observables are first rebuilt by :func:`live_observable`, so that
    the splitting reads their coefficients live and leaves the caller's tensors untouched. The
    trainable coefficients that PennyLane's postprocessing leaves out of the result are multiplied
    back in, see :func:`_unit_coefficient_candidates`. A torch multiple of the identity is
    handed to PennyLane as a constant, which its postprocessing can multiply, and gets its
    gradient back the same way, see :func:`_identity_offset`.

    :param tape: The circuit to split.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The circuits to execute and the function combining their results.
    :rtype: Tuple[List[QuantumScript], PostprocessingFn]
    """
    if (
        len(tape.measurements) == 1
        and tape.measurements[0].obs is not None
        and isinstance(tape.measurements[0].obs, BatchHamiltonian)
    ):
        return [tape], lambda results: results[0]
    measurements = []
    identity_offsets: Dict[int, torch.Tensor] = {}
    for index, measurement in enumerate(tape.measurements):
        if isinstance(measurement, ExpectationMP) and measurement.obs is not None:
            observable = live_observable(measurement.obs, r_dtype)
            if observable is not measurement.obs:
                measurement = qml.expval(observable)
            offset = None if tape.shots.has_partitioned_shots else _identity_offset(measurement.obs)
            if offset is not None:
                constant = offset.detach()
                measurement = qml.expval(qml.s_prod(constant.item(), qml.Identity(measurement.obs.wires)))
                identity_offsets[index] = offset - constant
        measurements.append(measurement)
    if any(new is not old for new, old in zip(measurements, tape.measurements)):
        # the operations are handed over as they are: without them in the update, every one is copied
        tape = tape.copy(measurements=measurements, operations=tape.operations)
    candidates = {} if tape.shots.has_partitioned_shots else _unit_coefficient_candidates(tape)
    tapes, postprocessing = qml.transforms.split_non_commuting(tape, grouping_strategy="wires")
    if not candidates and not identity_offsets:
        return tapes, postprocessing
    n_measurements = len(tape.measurements)

    def differentiable_postprocessing(results: ResultBatch):
        output = postprocessing(results)
        outputs = [output] if n_measurements == 1 else list(output)
        for index, (coefficient, offset) in candidates.items():
            if _is_raw_result(outputs[index], results):
                outputs[index] = _restored(outputs[index], coefficient, offset)
        for index, difference in identity_offsets.items():
            outputs[index] = _restored(outputs[index], 1.0, difference)
            if r_dtype is not None:
                outputs[index] = outputs[index].to(_like_precision(outputs[index].dtype, r_dtype))
        return outputs[0] if n_measurements == 1 else tuple(outputs)

    return tapes, differentiable_postprocessing
