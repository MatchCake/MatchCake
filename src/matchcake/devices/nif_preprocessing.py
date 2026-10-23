from typing import List, Optional, Set, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.measurements import ExpectationMP
from pennylane.operation import Operator
from pennylane.ops.op_math import SProd
from pennylane.tape import QuantumScript
from pennylane.typing import PostprocessingFn

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
    same grouping.

    Also returned unchanged, so that it takes exactly the path it took before: a plain Pauli word; a
    scalar multiple of a Hamiltonian or of a word, which ``SProd.terms()`` reads through the base's
    own ``terms()``; a grouped sum, whose grouping indices refer to its merged terms rather than to
    its operands; one with no Pauli part; and one reducing to a multiple of the identity or to
    nothing, which ``split_non_commuting`` treats as an offset.

    :param observable: The observable of a measurement.
    :param r_dtype: The real dtype of the device, double precision when not given.
    :return: The observable to measure instead.
    :rtype: Operator
    """
    if isinstance(observable, qml.ops.LinearCombination):
        if observable.grouping_indices is None or len(observable.ops) == 0:
            return observable
        rebuilt = qml.Hamiltonian(_copied_coefficients(observable.coeffs), list(observable.ops))
        rebuilt.grouping_indices = observable.grouping_indices
        return rebuilt
    if getattr(observable, "grouping_indices", None) is not None or _is_read_live(observable):
        return observable
    coefficients, operators = utils.live_pauli_terms(observable, keep_non_pauli=True)
    if not operators or any(operator is observable for operator in operators) or _is_identity_only(operators):
        return observable
    return qml.Hamiltonian(_in_working_precision(coefficients, observable, r_dtype), operators)


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
    dtypes = set()
    visited = set()
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


@qml.transform
def nif_split_non_commuting(
    tape: QuantumScript, r_dtype: Optional[torch.dtype] = None
) -> Tuple[List[QuantumScript], PostprocessingFn]:
    """Pass BatchHamiltonian measurements through unchanged; otherwise delegate to split_non_commuting.

    Composite expectation-value observables are first rebuilt by :func:`live_observable`, so that
    the splitting reads their coefficients live and leaves the caller's tensors untouched.

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
    for measurement in tape.measurements:
        if isinstance(measurement, ExpectationMP) and measurement.obs is not None:
            observable = live_observable(measurement.obs, r_dtype)
            if observable is not measurement.obs:
                measurement = qml.expval(observable)
        measurements.append(measurement)
    if any(new is not old for new, old in zip(measurements, tape.measurements)):
        tape = tape.copy(measurements=measurements)
    return qml.transforms.split_non_commuting(tape, grouping_strategy="wires")
