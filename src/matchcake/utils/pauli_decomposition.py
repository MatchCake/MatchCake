import itertools
import numbers
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator
from pennylane.ops import Identity, LinearCombination, PauliX, PauliY, PauliZ
from pennylane.ops.functions import bind_new_parameters
from pennylane.ops.op_math import Adjoint, ChangeOpBasis, CompositeOp, Pow, Prod, SProd, Sum, SymbolicOp
from pennylane.pauli import PauliWord

from ..typing import TensorLike

_PAULI_LETTERS = {PauliX: "X", PauliY: "Y", PauliZ: "Z"}

_ADDITIVE = (LinearCombination, Sum, SProd)
_MULTIPLICATIVE = (Prod, ChangeOpBasis, Adjoint, Pow)

# Multiple of the double-precision epsilon, times the largest coefficient, under which an imaginary
# part is taken as round-off: products of a few dozen terms through i phases stay well inside it.
_ROUND_OFF_FACTOR = 64

# PennyLane's default cutoff in Sum.simplify: when Sum.terms() falls back to simplification, it drops
# every summand whose coefficient is at or under it in magnitude.
_SIMPLIFY_CUTOFF = 1.0e-12

# Product of two single-qubit Pauli letters as (letter, phase), with "I" the identity:
# X Y = i Z, Y Z = i X, Z X = i Y, and the reversed products carry -i.
_LETTER_PRODUCTS = {
    ("X", "Y"): ("Z", 1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "X"): ("Y", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Z", "Y"): ("X", -1j),
    ("X", "Z"): ("Y", -1j),
}


class _WholeTerm:
    """Key of a term kept whole in a lenient expansion, compared by the identity of its operator.

    :param operator: The operator kept whole.
    """

    def __init__(self, operator: Operator):
        self.operator = operator

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _WholeTerm) and other.operator is self.operator

    def __hash__(self) -> int:
        return id(self.operator)


def pauli_word_product(left: PauliWord, right: PauliWord) -> Tuple[PauliWord, complex]:
    """Product ``left @ right`` of two Pauli words, as a word and a phase in ``{1, -1, i, -i}``.

    :param left: Left factor.
    :param right: Right factor.
    :return: The product word and its phase.
    :rtype: Tuple[PauliWord, complex]
    """
    letters = dict(left)
    phase = 1 + 0j
    for wire, letter in right.items():
        if wire not in letters:
            letters[wire] = letter
        elif letters[wire] == letter:
            del letters[wire]
        else:
            letters[wire], factor = _LETTER_PRODUCTS[(letters[wire], letter)]
            phase = phase * factor
    return PauliWord(letters), phase


def has_live_terms(observable: Operator) -> bool:
    """Whether an observable can be read safely as it is: a plain Pauli word, or a Hamiltonian of them.

    A plain Pauli word carries no coefficient, and ``LinearCombination.terms()`` returns its own
    coefficient array and operators as they are, so neither can be stale or mutated by the read.
    Anything else (a sum, a scalar product, a Hamiltonian whose terms carry their own factors) has to
    be walked. The answer depends on the structure alone, so it never changes for one observable
    object, which a decomposition cached per object relies on.

    :param observable: The observable.
    :return: ``True`` if the observable needs no live rebuild.
    :rtype: bool
    """
    if isinstance(observable, LinearCombination):
        return all(_is_plain_pauli_word(operator) for operator in observable.ops)
    return _is_plain_pauli_word(observable)


def live_pauli_sentence(observable: Operator) -> Dict[PauliWord, TensorLike]:
    """Expand an observable into ``{Pauli word: coefficient}`` by walking its expression tree.

    Every coefficient is read live from the node that carries it. A torch coefficient is kept as a
    torch tensor, so that gradients with respect to it flow through, and raised to double precision;
    a numpy or pennylane.numpy number becomes a python number, whose integers cannot overflow; a
    one-element array becomes a scalar. Arithmetic between the interfaces is then always well defined: torch
    cannot multiply a trainable tensor by a numpy array, and drops the imaginary part of a numpy
    single-precision complex scalar. PennyLane's
    own summaries are avoided on purpose: ``pauli_rep`` is cached when an operator is built, and
    ``Sum.terms()`` and ``Prod.terms()`` read that cache, so both go stale after an in-place
    coefficient update and after ``H + H``, whose ``PauliSentence.__add__`` adds into the shared
    coefficient tensors in place. ``Sum.terms()`` can even accumulate into the caller's own
    coefficient tensor when it falls back to simplification. This function performs no in-place
    operation on any coefficient, and every coefficient it returns is a new object.

    The set of words and their order depend only on the structure of the tree, never on the values
    of the coefficients, so a decomposition can be cached per observable and refreshed with the
    live coefficients. A word whose coefficient cancels to zero is kept, and a power must have a
    plain number as its exponent.

    The walk is iterative and linear in the size of the tree: sums, Hamiltonians and scalar
    products carry a scale down to their operands instead of building an expansion at every level,
    so a ``qml.sum(H, term)`` loop over a long Hamiltonian, nested past the interpreter's recursion
    limit, is expanded in one pass. An operand shared by several nodes is expanded once.

    :param observable: A Pauli-decomposable observable: Pauli words and identities combined with
        sums, Hamiltonians, scalar products, products, basis changes, integer powers and adjoints,
        plus any gate with a fixed Pauli representation, such as ``SWAP``.
    :return: The expansion, as an insertion-ordered dict from Pauli word to coefficient.
    :rtype: Dict[PauliWord, TensorLike]
    :raises TypeError: if part of the observable has no Pauli representation, or is a power whose
        exponent is not a real integer given as a plain number, or a negative power of anything but
        a single Pauli word.
    """
    return _expand(observable, keep_non_pauli=False)


def live_pauli_terms(observable: Operator, keep_non_pauli: bool = False) -> Tuple[TensorLike, List[Operator]]:
    """``(coefficients, operators)`` of an observable, read live, with one plain Pauli word per term.

    The drop-in replacement for ``observable.terms()`` where a stale or mutating read matters; see
    :func:`live_pauli_sentence` for why ``terms()`` is not used in general. An observable for which
    :func:`has_live_terms` holds is returned through its own ``terms()``, terms not merged, since
    that read is already live and is much cheaper.

    Otherwise the coefficients are returned by :func:`stacked_coefficients`, as one array of a
    single interface in double precision. An identity term acts on the wires of
    ``observable``, so that a strategy gathering probabilities over its wires still finds them.

    With ``keep_non_pauli``, an observable without a Pauli representation is reduced as
    ``Sum.terms()`` reduces it when it falls back to simplification, so that the terms are those
    PennyLane would have given, read live: each part kept whole is simplified (``P @ I`` to ``P``,
    ``H @ H`` to the identity) and taken apart into its summands, terms acting as the same operator
    are merged, and terms whose coefficient is at or under PennyLane's cutoff are dropped. Which
    terms are dropped then depends on the coefficients, as it does for ``Sum.terms()``.

    :param observable: A Pauli-decomposable observable.
    :param keep_non_pauli: When ``True``, a part of the observable without a Pauli representation
        is kept as a term of its own, reduced as described above, instead of raising.
    :return: Index-aligned coefficients and operators.
    :rtype: Tuple[TensorLike, List[Operator]]
    :raises TypeError: if part of the observable has no Pauli representation and ``keep_non_pauli``
        is ``False``, or for any other reason listed by :func:`live_pauli_sentence`.
    """
    if has_live_terms(observable):
        if isinstance(observable, LinearCombination):
            return observable.terms()
        return [1.0], [observable]
    with qml.QueuingManager.stop_recording():
        sentence = _expand(observable, keep_non_pauli=keep_non_pauli)
        terms = [
            (value, key.operator if isinstance(key, _WholeTerm) else _word_operator(key, observable.wires))
            for key, value in sentence.items()
        ]
        if keep_non_pauli and not observable.pauli_rep:
            terms = _simplified_terms(terms, observable.wires)
    if not terms:
        return [], []
    return stacked_coefficients([value for value, _ in terms]), [operator for _, operator in terms]


def stacked_coefficients(values: Sequence[TensorLike]) -> TensorLike:
    """Stack Hamiltonian coefficients into one array of a single interface, in double precision.

    A list mixing python scalars with tensors, or shapes ``()`` and ``(1,)``, is not accepted by
    PennyLane's ``Hamiltonian``, and a python scalar is rounded to single precision when PennyLane
    multiplies it into a torch result. The result is real when every imaginary part is at round-off
    level relative to the largest coefficient: a Hermitian observable built through ``i`` phases
    leaves such residues, and a complex expectation would refuse ``backward``.

    :param values: Python, numpy, pennylane.numpy or torch scalars, or one-element arrays.
    :return: The stacked coefficients, a new object.
    :rtype: TensorLike
    """
    coefficients = [read_coefficient(value) for value in values]
    interface = qml.math.get_interface(*coefficients)
    if interface == "torch":
        is_complex = any(
            value.is_complex() if isinstance(value, torch.Tensor) else isinstance(value, complex)
            for value in coefficients
        )
        dtype = torch.complex128 if is_complex else torch.float64
        device = next(value.device for value in coefficients if isinstance(value, torch.Tensor))
        stacked = torch.stack(
            torch.broadcast_tensors(*[torch.as_tensor(value, dtype=dtype, device=device) for value in coefficients])
        )
        if stacked.is_complex():
            residue = torch.finfo(torch.float64).eps * _ROUND_OFF_FACTOR * torch.max(torch.abs(stacked))
            if bool(torch.all(torch.abs(stacked.imag) <= residue)):
                stacked = stacked.real
        return stacked
    shape = np.broadcast_shapes(*[qml.math.shape(value) for value in coefficients])
    ones = np.ones(shape)
    stacked = qml.math.stack([value * ones for value in coefficients], like=interface)
    if np.iscomplexobj(qml.math.unwrap(stacked)):
        magnitudes = qml.math.unwrap(qml.math.abs(stacked))
        residue = np.finfo(np.float64).eps * _ROUND_OFF_FACTOR * np.max(magnitudes)
        if np.all(np.abs(qml.math.unwrap(qml.math.imag(stacked))) <= residue):
            stacked = qml.math.real(stacked)
    return stacked


def read_coefficient(value: TensorLike) -> TensorLike:
    """A coefficient as read from the tree, in a form every interface can multiply with every other.

    :param value: A python, numpy, pennylane.numpy or torch scalar, or a one-element array, or an
        autograd or jax tracer.
    :return: A double-precision 0-d torch tensor for torch input (differentiable), a python number
        for numpy and python input, and a tracer unchanged.
    :rtype: TensorLike
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.reshape(())
        return value.to(torch.complex128 if value.is_complex() else torch.float64)
    if isinstance(value, (np.ndarray, np.generic)) and np.size(value) == 1:
        return value.item()
    if not isinstance(value, numbers.Number):
        shape = qml.math.shape(value)
        if len(shape) > 0 and int(np.prod(shape)) == 1:
            return qml.math.reshape(value, ())
    return value


def _simplified_terms(terms: List[Tuple[TensorLike, Operator]], wires: Sequence) -> List[Tuple[TensorLike, Operator]]:
    """Terms reduced as ``Sum.terms()`` reduces them when it falls back to simplification.

    :param terms: ``(coefficient, operator)`` terms of a live expansion: Pauli words, and parts kept
        whole.
    :param wires: Wires of the observable, which an identity word acts on.
    :return: The terms with each part kept whole simplified and taken apart by :func:`_simplified_parts`,
        terms acting as the same operator merged in order of first appearance, and the terms whose
        coefficient is negligible dropped.
    :rtype: List[Tuple[TensorLike, Operator]]
    """
    merged: Dict[int, List] = {}
    for coefficient, operator in terms:
        parts = [(1.0, operator)] if _is_plain_pauli_word(operator) else _simplified_parts(operator, wires)
        for factor, part in parts:
            value = coefficient * factor
            if part.hash in merged:
                merged[part.hash][0] = merged[part.hash][0] + value
            else:
                merged[part.hash] = [value, part]
    return [(value, part) for value, part in merged.values() if not _is_negligible(value)]


def _simplified_parts(operator: Operator, wires: Sequence) -> List[Tuple[TensorLike, Operator]]:
    """``(coefficient, operator)`` parts of a term kept whole, simplified as ``Sum.terms()`` simplifies it.

    A summand with a Pauli representation, such as ``SX``, ``Adjoint(S)`` or ``Y0 @ Y0``, is replaced
    by its Pauli words, built as the words of the rest of the expansion, so that the two merge: the
    identity acts on the observable's wires, not on those ``simplify`` gave it. Kept as it is, a gate
    would be measured on its own: a strategy refuses it, or takes the real part of a non-Hermitian
    term whose imaginary part cancels against the words of the same gate read elsewhere in the tree.

    Simplifying adds repeated terms into a copy of their coefficient in place, which torch refuses for
    a trainable coefficient, copied as a leaf, and torch and numpy refuse for a complex value added
    into a real one. ``Prod.terms()`` is no way around it: it writes into a copy that shares storage
    with coefficients other products still read, which gives wrong values. The term is then
    multiplied out by :func:`_distributed` instead, and only products of its leaves, which hold no
    coefficient to add into, are simplified.

    :param operator: The term kept whole.
    :param wires: Wires of the observable, which an identity word acts on.
    :return: Its parts, in order.
    :rtype: List[Tuple[TensorLike, Operator]]
    :raises RuntimeError: if simplifying fails, and simplifying a product of the term's leaves fails too.
    :raises TypeError: likewise.
    """
    unshared = _unshared(operator)
    try:
        summands = _summands(unshared.simplify())
    except (RuntimeError, TypeError):
        summands = [
            (value * factor, summand)
            for value, factors in _distributed(_unshared(operator))
            for factor, summand in _summands(qml.prod(*factors).simplify())
        ]
    parts = []
    for factor, summand in summands:
        if summand.pauli_rep is None:
            parts.append((factor, summand))
        else:
            parts.extend(
                (factor * read_coefficient(value), _word_operator(word, wires))
                for word, value in summand.pauli_rep.items()
            )
    return parts


def _unshared(operator: Operator) -> Operator:
    """A copy of an operator whose torch parameters share no storage with the caller's tensors.

    PennyLane's ``simplify`` accumulates repeated terms into a coefficient in place. A scalar that
    ``qml.dot`` unbound from the caller's coefficient vector is a view of it, so simplifying the
    caller's operator would write into that vector. Clones keep the gradient.

    :param operator: The operator to simplify.
    :return: The operator itself when it has no torch parameter, otherwise a copy with each one cloned.
    :rtype: Operator
    """
    if not any(isinstance(parameter, torch.Tensor) for parameter in operator.data):
        return operator
    return bind_new_parameters(
        operator,
        [parameter.clone() if isinstance(parameter, torch.Tensor) else parameter for parameter in operator.data],
    )


def _distributed(operator: Operator) -> List[Tuple[TensorLike, List[Operator]]]:
    """``(coefficient, factors)`` products of leaves that sum to an operator, multiplied out without ``simplify``.

    Sums, Hamiltonians, scalar products, products and basis changes are multiplied out. An adjoint is
    moved onto the factors of each product, in reverse order, with the coefficient conjugated. A power
    to a positive integer is a product of copies of its base. Any other node is a leaf. Each
    coefficient is a new object, so the caller's are neither written into nor cut from the gradient.

    A term whose coefficient is a constant zero is dropped before it is multiplied, as ``simplify``
    drops it: a Hamiltonian padded with zero-weighted terms would otherwise multiply out into every
    product of them. A trainable zero is kept for its gradient.

    :param operator: The operator.
    :return: Its products, each with at least one factor, in order.
    :rtype: List[Tuple[TensorLike, List[Operator]]]
    """
    if isinstance(operator, LinearCombination):
        weighted = [(read_coefficient(coefficient), term) for coefficient, term in zip(operator.coeffs, operator.ops)]
        return [
            (coefficient * value, factors)
            for coefficient, term in weighted
            if not _is_constant_zero(coefficient)
            for value, factors in _distributed(term)
        ]
    if isinstance(operator, Sum):
        return [product for operand in operator.operands for product in _distributed(operand)]
    if isinstance(operator, SProd):
        scalar = read_coefficient(operator.scalar)
        if _is_constant_zero(scalar):
            return []
        return [(scalar * value, factors) for value, factors in _distributed(operator.base)]
    if isinstance(operator, Adjoint):
        return [
            (_conjugate(value), [qml.adjoint(factor) for factor in reversed(factors)])
            for value, factors in _distributed(operator.base)
        ]
    if isinstance(operator, (Prod, ChangeOpBasis)):
        operands = [_distributed(operand) for operand in operator.operands]
    elif (
        isinstance(operator, Pow)
        and isinstance(operator.z, numbers.Real)
        and float(operator.z).is_integer()
        and operator.z > 0
    ):
        operands = [_distributed(operator.base)] * int(operator.z)
    else:
        return [(1.0, [operator])]
    products: List[Tuple[TensorLike, List[Operator]]] = [(1.0, [])]
    for operand in operands:
        products = [
            (value * other_value, factors + other_factors)
            for value, factors in products
            for other_value, other_factors in operand
        ]
    return products


def _summands(operator: Operator) -> List[Tuple[TensorLike, Operator]]:
    """``(coefficient, operator)`` summands of a simplified operator, sums and scalar products taken apart.

    PennyLane's ``simplify`` is not idempotent: a product of Hamiltonians can come back as a product
    of one-term Hamiltonians, such as ``(2 Z) @ (2 Z)``, which a strategy would read as a Pauli word
    with coefficient one. Such a product is expanded here into the products of its factors' summands,
    and each of those is simplified on its own. Simplifying the whole operator a second time would
    instead accumulate in place into the torch coefficients that the first pass copied, which raises.
    A product of products is expanded the same way: ``(X2 @ X3) @ Y2 @ Y3``, whose phase ``-1`` a
    strategy would drop, and ``(H0 @ Z1) @ (Z1 @ H0)``, which a second pass reduces to the identity.
    A product of leaves other than a plain Pauli word is simplified again, which holds no coefficient
    to accumulate into: ``CZ @ Y0 @ Y1 @ Y0 @ Y1 @ CZ`` settles to ``CZ @ CZ``, then to the identity.
    A product that simplifies back to itself or to a product it came from, such as
    ``Hermitian @ Hermitian @ X``, is a summand as it is.

    :param operator: The operator, as returned by its ``simplify``.
    :return: Its summands, in order, with coefficients read by :func:`read_coefficient`.
    :rtype: List[Tuple[TensorLike, Operator]]
    """
    summands = []
    # Each pending node carries the hashes of the products simplified on its way down.
    pending: List[Tuple[TensorLike, Operator, frozenset]] = [(1.0, operator, frozenset())]
    while pending:
        factor, node, expanded = pending.pop()
        if isinstance(node, SProd):
            pending.append((factor * read_coefficient(node.scalar), node.base, expanded))
        elif isinstance(node, Sum):
            pending.extend((factor, operand, expanded) for operand in reversed(node.operands))
        elif isinstance(node, Prod) and node.hash not in expanded and not _is_plain_pauli_word(node):
            combinations = list(itertools.product(*[_summands(operand) for operand in node.operands]))
            for combination in reversed(combinations):
                scale = factor
                for value, _ in combination:
                    scale = scale * value
                pending.append((scale, qml.prod(*[part for _, part in combination]).simplify(), expanded | {node.hash}))
        else:
            summands.append((factor, node))
    return summands


def _is_constant_zero(value: TensorLike) -> bool:
    """Whether a coefficient read by :func:`read_coefficient` is exactly zero and carries no gradient.

    :param value: The coefficient.
    :return: ``True`` for a zero python number, or a zero torch tensor that does not require a
        gradient; ``False`` for anything else, a tracer included.
    :rtype: bool
    """
    if isinstance(value, torch.Tensor):
        return not value.requires_grad and bool(torch.all(value == 0))
    return isinstance(value, numbers.Number) and value == 0


def _is_negligible(value: TensorLike) -> bool:
    """Whether ``Sum.terms()`` drops a term of this coefficient when it falls back to simplification.

    :param value: The coefficient.
    :return: ``True`` if it is concrete and at or under PennyLane's cutoff in magnitude.
    :rtype: bool
    """
    if qml.math.is_abstract(value):
        return False
    return bool(np.all(np.abs(qml.math.unwrap(value)) <= _SIMPLIFY_CUTOFF))


def _conjugate(value: TensorLike) -> TensorLike:
    """Complex conjugate of a coefficient, in its own interface.

    :param value: The coefficient.
    :return: Its conjugate.
    :rtype: TensorLike
    """
    if isinstance(value, numbers.Number):
        return value.conjugate()
    return qml.math.conj(value)


def _is_plain_pauli_word(operator: Operator) -> bool:
    """Whether an operator is a Pauli, an identity, or a product of them on distinct wires.

    :param operator: The operator.
    :return: ``True`` for a plain Pauli word, which carries no coefficient of its own.
    :rtype: bool
    """
    if type(operator) in _PAULI_LETTERS or isinstance(operator, Identity):
        return True
    if not isinstance(operator, Prod):
        return False
    factors = operator.operands
    if not all(type(factor) in _PAULI_LETTERS or isinstance(factor, Identity) for factor in factors):
        return False
    wires = [wire for factor in factors for wire in factor.wires]
    return len(wires) == len(set(wires))


def _word_operator(word: PauliWord, wires: Sequence) -> Operator:
    """Operator of a Pauli word, with the identity placed on ``wires``.

    :param word: The Pauli word.
    :param wires: Wires the identity acts on, or none.
    :return: The operator.
    :rtype: Operator
    """
    if word:
        return word.operation()
    return qml.Identity(wires=wires) if len(wires) else qml.Identity()


def _describe(operator: Operator) -> str:
    """Printable name of an operator for an error message, falling back to its type.

    PennyLane's own ``repr`` raises on some valid trees (a composite holding an empty Hamiltonian),
    which would replace the error being reported by an unrelated one.

    :param operator: The operator.
    :return: Its ``repr``, or its class name if the ``repr`` fails.
    :rtype: str
    """
    try:
        return repr(operator)
    except Exception:
        return type(operator).__name__


def _power_exponent(operator: Pow) -> int:
    """Integer exponent of a power node.

    A tensor exponent is refused because it can change in place, and with it the set of words,
    which would leave a decomposition cached for the observable out of step with its coefficients.

    :param operator: The power node.
    :return: Its exponent.
    :rtype: int
    :raises TypeError: if the exponent is not a real integer given as a plain number.
    """
    if not isinstance(operator.z, numbers.Number):
        raise TypeError(f"expected a power with a plain number as exponent, got {_describe(operator)}.")
    exponent = complex(operator.z)
    if exponent.imag != 0.0 or not float(exponent.real).is_integer():
        raise TypeError(f"expected integer powers of Pauli observables, got {_describe(operator)}.")
    return int(exponent.real)


def _operands(operator: Operator) -> List[Operator]:
    """Operands a node's expansion is built from, in the order they multiply or add.

    A zero power has none: it is the identity whatever its base, which is not even read.

    :param operator: A node of the expression tree.
    :return: Its operands.
    :rtype: List[Operator]
    """
    if isinstance(operator, LinearCombination):
        return list(operator.ops)
    if isinstance(operator, (Sum, Prod, ChangeOpBasis)):
        return list(operator.operands)
    if isinstance(operator, (SProd, Adjoint)):
        return [operator.base]
    if isinstance(operator, Pow):
        return [] if _power_exponent(operator) == 0 else [operator.base]
    return []


def _operands_or_whole(operator: Operator, keep_non_pauli: bool) -> List[Operator]:
    """Operands of a node, or none when a node that cannot be expanded is to be kept whole.

    :param operator: The node.
    :param keep_non_pauli: Whether a node that cannot be expanded is kept whole.
    :return: Its operands.
    :rtype: List[Operator]
    :raises TypeError: if the node is an invalid power and ``keep_non_pauli`` is ``False``.
    """
    try:
        return _operands(operator)
    except TypeError:
        if not keep_non_pauli:
            raise
        return []


def _accumulate(sentence: Dict[Hashable, TensorLike], key: Hashable, value: TensorLike) -> None:
    """Add ``value`` to the coefficient of ``key``, building a new object rather than adding in place.

    :param sentence: The expansion being built, updated in place.
    :param key: The Pauli word, or a whole term.
    :param value: The coefficient to add.
    """
    sentence[key] = sentence[key] + value if key in sentence else value


def _scaled(value: TensorLike, phase: complex) -> TensorLike:
    """``phase * value`` for a phase in ``{1, -1, i, -i}``, keeping a real value real when it can.

    :param value: The coefficient.
    :param phase: The phase.
    :return: The scaled coefficient, a new object.
    :rtype: TensorLike
    """
    if phase == 1:
        return value * 1
    if phase == -1:
        return -value
    return value * phase


def _sentence_product(
    left: Dict[PauliWord, TensorLike], right: Dict[PauliWord, TensorLike]
) -> Dict[PauliWord, TensorLike]:
    """Product of two Pauli sums, merged by word.

    :param left: Left factor.
    :param right: Right factor.
    :return: The product.
    :rtype: Dict[PauliWord, TensorLike]
    """
    product: Dict[PauliWord, TensorLike] = {}
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word, phase = pauli_word_product(left_word, right_word)
            _accumulate(product, word, _scaled(left_value * right_value, phase))
    return product


def _leaf_sentence(operator: Operator) -> Dict[PauliWord, TensorLike]:
    """Expansion of a leaf: a Pauli, the identity, or a gate with a fixed Pauli representation.

    :param operator: The leaf.
    :return: The expansion.
    :rtype: Dict[PauliWord, TensorLike]
    :raises TypeError: if the leaf has no Pauli representation.
    """
    if isinstance(operator, Identity):
        return {PauliWord({}): 1.0}
    if type(operator) in _PAULI_LETTERS:
        return {PauliWord({operator.wires[0]: _PAULI_LETTERS[type(operator)]}): 1.0}
    if isinstance(operator, (CompositeOp, SymbolicOp)) or operator.pauli_rep is None:
        raise TypeError(
            f"expected a sum of Pauli words, got {_describe(operator)}, which has no Pauli representation "
            f"that can be read live."
        )
    return {word: read_coefficient(value) for word, value in operator.pauli_rep.items()}


def _multiplicative_sentence(
    operator: Operator, operand_sentences: List[Dict[PauliWord, TensorLike]]
) -> Dict[PauliWord, TensorLike]:
    """Expansion of a product, basis change, adjoint, power or leaf, from its operands' expansions.

    :param operator: The node.
    :param operand_sentences: Expansions of :func:`_operands` of the node, in the same order.
    :return: The expansion.
    :rtype: Dict[PauliWord, TensorLike]
    :raises TypeError: if the node has no Pauli representation, or is a negative power of anything
        but a single Pauli word.
    """
    if isinstance(operator, (Prod, ChangeOpBasis)):
        sentence = {PauliWord({}): 1.0}
        for operand in operand_sentences:
            sentence = _sentence_product(sentence, operand)
        return sentence
    if isinstance(operator, Adjoint):
        # Pauli words are Hermitian, so only the coefficients are conjugated.
        return {word: _conjugate(value) for word, value in operand_sentences[0].items()}
    if isinstance(operator, Pow):
        exponent = _power_exponent(operator)
        if exponent == 0:
            return {PauliWord({}): 1.0}
        base = operand_sentences[0]
        if exponent < 0:
            if len(base) != 1:
                raise TypeError(
                    f"can only invert a single Pauli word, got {_describe(operator)}, whose base has {len(base)} terms."
                )
            ((word, value),) = base.items()
            # A Pauli word is its own inverse.
            base = {word: 1 / value}
        sentence = {PauliWord({}): 1.0}
        for _ in range(abs(exponent)):
            sentence = _sentence_product(sentence, base)
        return sentence
    return _leaf_sentence(operator)


def _expand(observable: Operator, keep_non_pauli: bool) -> Dict[Hashable, TensorLike]:
    """Expansion of an observable, with the parts that have no Pauli representation kept whole or refused.

    The tree is cut into regions. A region starts at the observable, at a node shared by several
    parents, or at an operand of a product, basis change, adjoint or power, and runs down through
    sums, Hamiltonians and scalar products. Each region is expanded once, in an order where every
    region it depends on is already done, by carrying the product of the scales met on the way down
    to each node where it stops.

    :param observable: The observable.
    :param keep_non_pauli: Whether to keep a part without a Pauli representation whole.
    :return: The expansion, keyed by Pauli word, or by :class:`_WholeTerm` for a part kept whole.
    :rtype: Dict[Hashable, TensorLike]
    :raises TypeError: as listed by :func:`live_pauli_sentence`, unless ``keep_non_pauli`` covers it.
    """
    operands_of: Dict[int, List[Operator]] = {id(observable): _operands_or_whole(observable, keep_non_pauli)}
    parent_counts: Dict[int, int] = {id(observable): 0}
    nodes: List[Operator] = [observable]
    stack = [observable]
    while stack:
        node = stack.pop()
        for operand in operands_of[id(node)]:
            if id(operand) in parent_counts:
                parent_counts[id(operand)] += 1
                continue
            parent_counts[id(operand)] = 1
            operands_of[id(operand)] = _operands_or_whole(operand, keep_non_pauli)
            nodes.append(operand)
            stack.append(operand)

    region_starts = {id(observable)} | {node_id for node_id, count in parent_counts.items() if count > 1}
    for node in nodes:
        if isinstance(node, _MULTIPLICATIVE):
            region_starts.update(id(operand) for operand in operands_of[id(node)])

    ordered: List[Operator] = []
    started: set = set()
    order_stack: List[Tuple[Operator, bool]] = [(observable, False)]
    while order_stack:
        start, dependencies_done = order_stack.pop()
        if dependencies_done:
            ordered.append(start)
            continue
        if id(start) in started:
            continue
        started.add(id(start))
        order_stack.append((start, True))
        order_stack.extend(
            (dependency, False)
            for dependency in _region_dependencies(start, region_starts, operands_of)
            if id(dependency) not in started
        )

    expansions: Dict[int, Dict[Hashable, TensorLike]] = {}
    for start in ordered:
        expansions[id(start)] = _region_expansion(start, region_starts, operands_of, expansions, keep_non_pauli)
    return expansions[id(observable)]


def _region_dependencies(start: Operator, region_starts: set, operands_of: Dict[int, List[Operator]]) -> List[Operator]:
    """The region starts whose expansions the region beginning at ``start`` reads.

    :param start: First node of the region.
    :param region_starts: Ids of every region's first node.
    :param operands_of: Operands of every node.
    :return: The region starts it depends on.
    :rtype: List[Operator]
    """
    dependencies: List[Operator] = []
    pending = [start]
    while pending:
        node = pending.pop()
        if node is not start and id(node) in region_starts:
            dependencies.append(node)
        elif isinstance(node, _ADDITIVE):
            pending.extend(operands_of[id(node)])
        else:
            dependencies.extend(operands_of[id(node)])
    return dependencies


def _region_expansion(
    start: Operator,
    region_starts: set,
    operands_of: Dict[int, List[Operator]],
    expansions: Dict[int, Dict[Hashable, TensorLike]],
    keep_non_pauli: bool,
) -> Dict[Hashable, TensorLike]:
    """Expansion of the region beginning at ``start``, from the expansions of the regions it reads.

    :param start: First node of the region.
    :param region_starts: Ids of every region's first node.
    :param operands_of: Operands of every node.
    :param expansions: Expansions of the regions already done.
    :param keep_non_pauli: Whether a node that cannot be expanded is kept whole.
    :return: The expansion.
    :rtype: Dict[Hashable, TensorLike]
    """
    expansion: Dict[Hashable, TensorLike] = {}
    pending: List[Tuple[Operator, TensorLike]] = [(start, 1)]
    while pending:
        node, scale = pending.pop()
        if node is not start and id(node) in region_starts:
            terms = expansions[id(node)]
        elif isinstance(node, LinearCombination):
            weighted = list(zip(node.coeffs, operands_of[id(node)]))
            pending.extend(
                (operand, scale * read_coefficient(coefficient)) for coefficient, operand in reversed(weighted)
            )
            continue
        elif isinstance(node, Sum):
            pending.extend((operand, scale) for operand in reversed(operands_of[id(node)]))
            continue
        elif isinstance(node, SProd):
            pending.append((operands_of[id(node)][0], scale * read_coefficient(node.scalar)))
            continue
        else:
            terms = _node_expansion(node, operands_of, expansions, keep_non_pauli)
        for key, value in terms.items():
            _accumulate(expansion, key, scale * value)
    return expansion


def _node_expansion(
    operator: Operator,
    operands_of: Dict[int, List[Operator]],
    expansions: Dict[int, Dict[Hashable, TensorLike]],
    keep_non_pauli: bool,
) -> Dict[Hashable, TensorLike]:
    """Expansion of a product, basis change, adjoint, power or leaf, or the node kept whole.

    :param operator: The node.
    :param operands_of: Operands of every node.
    :param expansions: Expansions of the regions already done, which include its operands.
    :param keep_non_pauli: Whether a node that cannot be expanded is kept whole.
    :return: The expansion.
    :rtype: Dict[Hashable, TensorLike]
    :raises TypeError: if the node cannot be expanded and ``keep_non_pauli`` is ``False``.
    """
    operand_expansions = [expansions[id(operand)] for operand in operands_of[id(operator)]]
    whole = {_WholeTerm(operator): 1.0}
    if any(isinstance(key, _WholeTerm) for operand in operand_expansions for key in operand):
        # An operand kept whole cannot be multiplied through, so neither can this node.
        return whole
    try:
        return _multiplicative_sentence(operator, operand_expansions)
    except TypeError:
        if not keep_non_pauli:
            raise
        return whole
