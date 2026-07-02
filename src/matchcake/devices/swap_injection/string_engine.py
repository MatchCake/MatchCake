import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.pauli import pauli_word_to_string

from ...typing import TensorLike
from ...utils import JordanWigner
from ...utils._pfaffian import signed_pfaffian_complex
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_complex_dtype
from .branch_observables import _build_lambda_y_block, basis_state_probability, transition_cov

# CZ_{jk} = (I + Z_j + Z_k - Z_j Z_k) / 2 with Z_q = -i c_{2q} c_{2q+1}: each CZ expands into
# four Majorana monomials, (coefficient, Z-bilinear qubit list).
CZ_EXPANSION_TERMS: Tuple[Tuple[complex, Tuple[str, ...]], ...] = (
    (0.5 + 0.0j, ()),
    (-0.5j, ("j",)),
    (-0.5j, ("k",)),
    (0.5 + 0.0j, ("j", "k")),
)

# The type-1 (projected) factor of one branching, -2 n_j n_k = -(I - Z_j - Z_k + Z_j Z_k) / 2, in
# the same monomial table form; together with the type-0 identity it recomposes CZ_EXPANSION_TERMS.
TYPE1_EXPANSION_TERMS: Tuple[Tuple[complex, Tuple[str, ...]], ...] = (
    (-0.5 + 0.0j, ()),
    (-0.5j, ("j",)),
    (-0.5j, ("k",)),
    (0.5 + 0.0j, ("j", "k")),
)


class CzStringEngine:
    r"""Overlap-free evaluator for matchgate + ``SWAP``/``CZ`` circuits (swap_injection_theory.md section 11).

    The branch-tensor observables of :mod:`.branch_observables` divide by pairwise branch overlaps
    (through :math:`(\Lambda_\alpha + \Lambda_\beta)^{-1}`), which is ill-defined when wire-sharing
    ``SWAP``\ s drive two branches exactly orthogonal. This engine is the structural cure: it never
    forms a pairwise overlap. Each genuine ``CZ`` is expanded as

    .. math::
        \mathrm{CZ}_{jk} = \tfrac12 (I + Z_j + Z_k - Z_j Z_k), \qquad Z_q = -i\,c_{2q} c_{2q+1},

    and every Majorana factor is pushed through the matchgate layers that follow it (Heisenberg
    picture, via the suffix SPTMs), so the state becomes a sum of at most :math:`4^m` strings of
    linear forms in the final-frame Majoranas acting on the single Gaussian state
    :math:`|G\rangle` of the CZ-free circuit. Every observable then reduces to Wick Pfaffians of
    pairwise contractions against :math:`\Lambda_G` (expectation values, marginals) or against the
    transition covariance to one well-overlapping reference basis state (full-outcome
    probabilities), all manifestly finite for any ``SWAP`` placement.

    The strings are grouped by length at construction and stacked into tensors, so a whole
    observable (all Pauli terms, all string pairs) is evaluated with one batched Pfaffian call per
    distinct sandwich length rather than one call per string pair.

    The engine works in the even ``(2n+2)`` lifted frame throughout, so displaced (product-state)
    inputs and odd-weight Pauli terms (marker rule) are supported. The cost is
    :math:`O(16^m)` string pairs per Hamiltonian term and :math:`O(4^m)` strings per full-outcome
    amplitude, against :math:`O(4^m)` branch pairs for the overlap-normalized path, so the device
    only routes here when branches are (near-)orthogonal.

    :param lifted_covariance: Lifted covariance ``Lambda_G`` of the CZ-free circuit output, of
        shape ``(..., D, D)`` with ``D = 2n + 2``.
    :param cz_events: One ``(j, k, sptm_prefix)`` triple per genuine ``CZ`` in circuit order,
        where ``sptm_prefix`` is the physical ``(..., 2n, 2n)`` SPTM of every matchgate layer
        applied before the event (``None`` for none).
    :param sptm_total: Physical SPTM of every matchgate layer of the circuit (``None`` for none).
    """

    @staticmethod
    def branch_pair_cost(histories: List[Tuple[int, ...]], pair_mask: np.ndarray) -> int:
        """Number of string pairs of the masked branch pairs, ``sum 4^(s_a + s_b)``.

        Used by the device to decide between the hybrid per-pair evaluation and the full
        CZ-expansion engine (``16^m`` string pairs): once most pairs are masked, the branch-pair
        enumeration is the more expensive of the two.

        :param histories: Per-branch type-0/type-1 choices.
        :param pair_mask: Boolean ``(chi, chi)`` mask of the branch pairs to evaluate.
        :return: Total string-pair count of the masked pairs.
        :rtype: int
        """
        return int(
            sum(
                4 ** (sum(histories[bra_index]) + sum(histories[ket_index]))
                for bra_index, ket_index in zip(*np.nonzero(pair_mask))
            )
        )

    @staticmethod
    def _slice_batch(tensor: Optional[TensorLike], index: Tuple[int, ...]) -> Optional[TensorLike]:
        """Select one batch element of an SPTM, passing unbatched (shared) tensors through.

        :param tensor: SPTM of shape ``(..., 2n, 2n)`` or ``None``.
        :param index: Batch index to select.
        :return: SPTM of shape ``(2n, 2n)`` or ``None``.
        :rtype: Optional[TensorLike]
        """
        if tensor is None or qml.math.ndim(tensor) <= 2:
            return tensor
        return tensor[index]

    @staticmethod
    def _expand_product_strings(
        per_factor_terms: Sequence[Sequence[Tuple[complex, List[torch.Tensor]]]],
    ) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Cartesian product of per-factor ``(coefficient, vectors)`` choices, one string per combo.

        Each factor offers a small menu of monomial terms; a string picks one term per factor,
        multiplies the coefficients, and concatenates the vectors in factor order (so the first
        factor sits leftmost). Callers that need a different vector order feed the factors in the
        matching order. Replaces the hand-rolled per-factor doubling loop that the ``4^m``
        expansions and the ``2^k`` marginal expansion each duplicated.

        :param per_factor_terms: Per factor, its list of ``(coefficient, vectors)`` terms.
        :return: One ``(coefficient, vectors)`` string per term combination.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        strings: List[Tuple[complex, List[torch.Tensor]]] = [(1.0 + 0.0j, [])]
        for factor_terms in per_factor_terms:
            strings = [
                (coefficient * term_coefficient, vectors + term_vectors)
                for coefficient, vectors in strings
                for term_coefficient, term_vectors in factor_terms
            ]
        return strings

    @staticmethod
    def _event_terms(
        bilinear_vectors: Dict[str, List[torch.Tensor]],
        expansion_terms: Tuple[Tuple[complex, Tuple[str, ...]], ...],
    ) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Materialize one event's monomial menu from a ``Z``-bilinear expansion table.

        :param bilinear_vectors: The event's ``Z_j`` and ``Z_k`` bilinear vector pairs.
        :param expansion_terms: A ``(coefficient, qubit-name tuple)`` monomial table.
        :return: ``(coefficient, vectors)`` terms of the event.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        return [
            (term_coefficient, [vector for name in term_qubits for vector in bilinear_vectors[name]])
            for term_coefficient, term_qubits in expansion_terms
        ]

    def __init__(
        self,
        lifted_covariance: TensorLike,
        cz_events: Sequence[Tuple[int, int, Optional[TensorLike]]],
        sptm_total: Optional[TensorLike] = None,
    ):
        self._complex_dtype = infer_complex_dtype(lifted_covariance)
        self.lam_g = torch.as_tensor(
            qml.math.toarray(lifted_covariance)
            if not isinstance(lifted_covariance, torch.Tensor)
            else lifted_covariance
        ).to(self._complex_dtype)
        self.dim = self.lam_g.shape[-1]
        self.marker = self.dim - 1
        self.n_phys = (self.dim - 2) // 2
        self._device = self.lam_g.device
        self._eye = torch.eye(self.dim, dtype=self._complex_dtype, device=self._device)
        self._cz_events = list(cz_events)
        self._sptm_total = sptm_total
        self._event_vectors = self._build_event_vectors(cz_events, sptm_total)
        self.strings = self._build_strings()
        self._bra_groups, self._ket_groups = self._build_string_groups(self.strings)
        self._contraction_g: Optional[torch.Tensor] = None
        self._reference: Optional[Tuple[np.ndarray, torch.Tensor, torch.Tensor]] = None
        self._element_engines: Optional[List["CzStringEngine"]] = None
        self._history_groups: Dict[Tuple[int, ...], Tuple[Dict, Dict]] = {}

    def hamiltonian_expval(self, observable: Operator, wires: List) -> TensorLike:
        """Expectation value ``<H>`` of a Pauli-sum observable, exact for any ``SWAP`` placement.

        Mirrors :func:`.branch_observables.hamiltonian_expval`: each Pauli term maps to a Majorana
        monomial through Jordan-Wigner (odd-weight terms append the parity marker), evaluated by
        Wick contractions of the CZ-expansion string pairs against ``Lambda_G``.

        :param observable: A Pauli-decomposable observable (exposing ``observable.terms()``).
        :param wires: Device wire labels in qubit order.
        :return: Real expectation value ``<H>`` (scalar or batched ``(...)``).
        :rtype: TensorLike
        """
        weighted_observables = self._hamiltonian_weighted_observables(observable, wires)
        contraction = self._get_contraction_g()
        expectation = qml.math.real(self._string_pair_sum_many(weighted_observables, contraction))
        return convert_and_cast_like(expectation, qml.math.real(self.lam_g))

    def branch_pair_hamiltonian_expval(
        self,
        observable: Operator,
        wires: List,
        histories: List[Tuple[int, ...]],
        pair_mask: np.ndarray,
    ) -> TensorLike:
        """Overlap-free sum of ``<lambda_a phi_a| H |lambda_b phi_b>`` over the masked branch pairs.

        Each branch's unnormalized state is a sum of at most ``4^s`` projector-monomial strings
        (``s`` type-1 events, see :meth:`_build_branch_strings`), so a masked pair is a plain
        string-pair Wick sum against ``Lambda_G`` with no branch overlap anywhere. This is the
        hybrid complement of the overlap-normalized evaluation restricted to the healthy pairs.

        :param observable: A Pauli-decomposable observable (exposing ``observable.terms()``).
        :param wires: Device wire labels in qubit order.
        :param histories: Per-branch type-0/type-1 choices, aligned with the covariance tensor.
        :param pair_mask: Boolean ``(chi, chi)`` mask of the branch pairs to evaluate.
        :return: Real sum of the masked pair contributions (scalar or batched ``(...)``).
        :rtype: TensorLike
        """
        weighted_observables = self._hamiltonian_weighted_observables(observable, wires)
        contraction = self._get_contraction_g()
        jobs = [
            (self._get_history_groups(histories[bra_index])[0], self._get_history_groups(histories[ket_index])[1])
            for bra_index, ket_index in zip(*np.nonzero(pair_mask))
        ]
        total = self._bucketed_jobs_sum(jobs, weighted_observables, contraction)
        return convert_and_cast_like(qml.math.real(total), qml.math.real(self.lam_g))

    def branch_pair_marginal_probability(
        self,
        target_state: TensorLike,
        measured_qubits: List[int],
        histories: List[Tuple[int, ...]],
        pair_mask: np.ndarray,
    ) -> TensorLike:
        """Overlap-free sum of ``<lambda_a phi_a| P_y |lambda_b phi_b>`` over the masked branch pairs.

        The marginal projector is expanded in ``2^k`` Pauli-``Z`` words (as in
        :meth:`basis_state_probability`) and every word is a string-pair Wick sum over the masked
        pairs' branch strings.

        :param target_state: Outcome bits of the measured qubits, an array of ``k`` bits.
        :param measured_qubits: Qubit indices the bits refer to.
        :param histories: Per-branch type-0/type-1 choices, aligned with the covariance tensor.
        :param pair_mask: Boolean ``(chi, chi)`` mask of the branch pairs to evaluate.
        :return: Real sum of the masked pair contributions (scalar or batched ``(...)``).
        :rtype: TensorLike
        """
        bits = np.asarray(qml.math.toarray(target_state)).astype(int).reshape(-1)
        weighted_observables = self._marginal_weighted_observables(bits, list(measured_qubits))
        contraction = self._get_contraction_g()
        jobs = [
            (self._get_history_groups(histories[bra_index])[0], self._get_history_groups(histories[ket_index])[1])
            for bra_index, ket_index in zip(*np.nonzero(pair_mask))
        ]
        total = self._bucketed_jobs_sum(jobs, weighted_observables, contraction)
        probability = qml.math.real(total) / 2.0 ** len(measured_qubits)
        return convert_and_cast_like(probability, qml.math.real(self.lam_g))

    def basis_state_probability(
        self,
        target_state: TensorLike,
        measured_qubits: Optional[List[int]] = None,
    ) -> TensorLike:
        """Outcome probability ``p(y)``, exact for any ``SWAP`` placement.

        Full outcomes (every physical qubit measured) go through per-string amplitudes relative to
        a well-overlapping reference basis state of the lifted Gaussian; marginals go through the
        ``Z``-expansion of the projector (``2^k`` Pauli terms for ``k`` measured qubits).

        :param target_state: Outcome bits of the measured qubits, an array of ``k`` bits.
        :param measured_qubits: Qubit indices the bits refer to. Defaults to ``range(k)``.
        :return: Real probability ``p(y)`` (scalar or batched ``(...)``).
        :rtype: TensorLike
        """
        bits = np.asarray(qml.math.toarray(target_state)).astype(int).reshape(-1)
        if measured_qubits is None:
            measured_qubits = list(range(len(bits)))
        if sorted(measured_qubits) == list(range(self.n_phys)):
            qubit_order_bits = bits[np.argsort(measured_qubits)]
            probability = self._full_state_probability(qubit_order_bits)
        else:
            probability = self._marginal_probability(bits, list(measured_qubits))
        return convert_and_cast_like(qml.math.real(probability), qml.math.real(self.lam_g))

    def _batch_shape(self) -> Tuple[int, ...]:
        """Leading batch shape of the engine data.

        :return: Broadcast batch shape of ``Lambda_G``.
        :rtype: Tuple[int, ...]
        """
        return tuple(self.lam_g.shape[:-2])

    def _build_event_vectors(
        self,
        cz_events: Sequence[Tuple[int, int, Optional[TensorLike]]],
        sptm_total: Optional[TensorLike],
    ) -> List[Dict[str, List[torch.Tensor]]]:
        """Heisenberg-rotated Majorana bilinear vectors of every event.

        Each vector is a row of the event's suffix SPTM (the matchgate layers applied after the
        event), zero-padded to the lifted dimension.

        :param cz_events: ``(j, k, sptm_prefix)`` triples in circuit order.
        :param sptm_total: Physical SPTM of the whole matchgate stream, or ``None``.
        :return: Per event, the ``Z_j`` and ``Z_k`` bilinear vector pairs of shape ``(..., D)``.
        :rtype: List[Dict[str, List[torch.Tensor]]]
        """
        total = self._to_complex_tensor(sptm_total) if sptm_total is not None else None
        event_vectors = []
        for qubit_j, qubit_k, sptm_prefix in cz_events:
            suffix = self._suffix_sptm(sptm_prefix, total)
            event_vectors.append(
                {
                    "j": [self._mode_vector(suffix, 2 * qubit_j), self._mode_vector(suffix, 2 * qubit_j + 1)],
                    "k": [self._mode_vector(suffix, 2 * qubit_k), self._mode_vector(suffix, 2 * qubit_k + 1)],
                }
            )
        return event_vectors

    def _build_strings(self) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Expand every ``CZ`` event into Heisenberg-rotated Majorana strings.

        Later events sit to the left of earlier ones, so each event's rotated vectors are
        prepended.

        :return: List of ``(coefficient, vectors)`` strings, vectors of shape ``(..., D)``.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        per_event_terms = [
            self._event_terms(bilinear_vectors, CZ_EXPANSION_TERMS) for bilinear_vectors in self._event_vectors
        ]
        return self._expand_product_strings(list(reversed(per_event_terms)))

    def _build_branch_strings(self, history: Tuple[int, ...]) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Strings of one branch's unnormalized state ``lambda |phi> = prod_t [1 or -2 n_j n_k] |G>``.

        A type-0 event contributes the identity; a type-1 event contributes the four monomials of
        ``-2 n_j n_k`` (``TYPE1_EXPANSION_TERMS``, with the ``-2`` weight factor folded in), so a
        branch with ``s`` type-1 events expands into ``4^s`` strings. Later events sit to the left.

        :param history: The branch's type-0/type-1 choice per event, in circuit order.
        :return: List of ``(coefficient, vectors)`` strings, vectors of shape ``(..., D)``.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        per_event_terms = [
            self._event_terms(bilinear_vectors, TYPE1_EXPANSION_TERMS)
            for choice, bilinear_vectors in zip(history, self._event_vectors)
            if choice != 0
        ]
        return self._expand_product_strings(list(reversed(per_event_terms)))

    def _build_string_groups(
        self,
        strings: List[Tuple[complex, List[torch.Tensor]]],
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Stack strings into per-length tensors for batched pairwise evaluation.

        For every distinct string length the ket variant stacks the vectors in string order and the
        bra variant stacks their conjugates in reversed order (the adjoint of the operator string),
        so any bra-observable-ket sandwich is a concatenation along the factor axis.

        :param strings: ``(coefficient, vectors)`` strings to group.
        :return: ``(bra_groups, ket_groups)``, each mapping a length to a ``(coefficients,
            vectors)`` pair of shapes ``(n_strings,)`` and ``(n_strings, *batch, length, D)``.
        :rtype: Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]
        """
        by_length: Dict[int, List[Tuple[complex, List[torch.Tensor]]]] = {}
        for coefficient, vectors in strings:
            by_length.setdefault(len(vectors), []).append((coefficient, vectors))
        bra_groups: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        ket_groups: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for length, group in by_length.items():
            coefficients = torch.as_tensor(
                [coefficient for coefficient, _ in group], dtype=self._complex_dtype, device=self._device
            )
            kets = torch.stack([self._vectors_to_tensor(vectors) for _, vectors in group])
            bras = torch.stack(
                [self._vectors_to_tensor([torch.conj(vector) for vector in reversed(vectors)]) for _, vectors in group]
            )
            ket_groups[length] = (coefficients, kets)
            bra_groups[length] = (coefficients, bras)
        return bra_groups, ket_groups

    def _get_history_groups(
        self, history: Tuple[int, ...]
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Cached per-length string groups of one branch history.

        :param history: The branch's type-0/type-1 choice per event.
        :return: The branch's ``(bra_groups, ket_groups)``.
        :rtype: Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]
        """
        key = tuple(history)
        if key not in self._history_groups:
            self._history_groups[key] = self._build_string_groups(self._build_branch_strings(key))
        return self._history_groups[key]

    def _vectors_to_tensor(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Stack linear-form vectors along a factor axis, broadcast to the engine batch shape.

        :param vectors: Vectors of shape ``(..., D)``.
        :return: Stacked tensor of shape ``(*batch, len(vectors), D)``.
        :rtype: torch.Tensor
        """
        batch_shape = self._batch_shape()
        if not vectors:
            return torch.zeros(*batch_shape, 0, self.dim, dtype=self._complex_dtype, device=self._device)
        return torch.stack([torch.broadcast_to(vector, batch_shape + (self.dim,)) for vector in vectors], dim=-2)

    def _suffix_sptm(
        self, sptm_prefix: Optional[TensorLike], sptm_total: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Physical SPTM of the matchgate layers after an event: ``Q_suffix = Q_prefix^T Q_total``.

        :param sptm_prefix: Physical SPTM before the event, or ``None`` for identity.
        :param sptm_total: Physical SPTM of the whole matchgate stream, or ``None`` for identity.
        :return: Suffix SPTM of shape ``(..., 2n, 2n)``, or ``None`` for identity.
        :rtype: Optional[torch.Tensor]
        """
        if sptm_prefix is None:
            return sptm_total
        prefix = self._to_complex_tensor(sptm_prefix)
        if sptm_total is None:
            return prefix.transpose(-1, -2)
        return prefix.transpose(-1, -2) @ sptm_total

    def _mode_vector(self, suffix: Optional[torch.Tensor], mode: int) -> torch.Tensor:
        """Final-frame linear form of the Majorana ``c_mode`` inserted at an event.

        :param suffix: Physical suffix SPTM ``(..., 2n, 2n)``, or ``None`` for identity.
        :param mode: Physical Majorana index at the event time.
        :return: Coefficient vector of shape ``(..., D)`` in the lifted frame.
        :rtype: torch.Tensor
        """
        if suffix is None:
            return self._eye[mode]
        batch_shape = tuple(suffix.shape[:-2])
        vector = torch.zeros(*batch_shape, self.dim, dtype=self._complex_dtype, device=self._device)
        vector[..., : 2 * self.n_phys] = suffix[..., mode, :]
        return vector

    def _to_complex_tensor(self, tensor: TensorLike) -> torch.Tensor:
        """Convert any tensor-like to a torch tensor in the engine's complex dtype.

        :param tensor: Input tensor-like.
        :return: Torch tensor on the engine device.
        :rtype: torch.Tensor
        """
        tensor_t = torch.as_tensor(qml.math.toarray(tensor) if not isinstance(tensor, torch.Tensor) else tensor)
        return tensor_t.to(dtype=self._complex_dtype, device=self._device)

    def _get_contraction_g(self) -> torch.Tensor:
        """Contraction matrix ``I - i Gamma`` of the single Gaussian ``|G>`` (marker rescaled).

        :return: Contraction matrix of shape ``(..., D, D)``.
        :rtype: torch.Tensor
        """
        if self._contraction_g is None:
            gamma = transition_cov(self.lam_g, self.lam_g, marker=self.marker)
            self._contraction_g = self._eye - 1j * self._to_complex_tensor(gamma)
        return self._contraction_g

    def _stacked_wick_pfaffians(self, stacked_vectors: torch.Tensor, contraction: torch.Tensor) -> torch.Tensor:
        """Wick Pfaffians ``Pf(A)`` with ``A_{ab} = v_a^T (I - i Gamma) v_b`` (``a < b``), batched.

        :param stacked_vectors: Ordered factors of shape ``(..., length, D)`` with even ``length``.
        :param contraction: Contraction matrix of shape ``(..., D, D)``.
        :return: Pfaffians of shape ``(...)``.
        :rtype: torch.Tensor
        """
        raw = stacked_vectors @ contraction @ stacked_vectors.transpose(-1, -2)
        upper = torch.triu(raw, diagonal=1)
        return signed_pfaffian_complex(upper - upper.transpose(-1, -2))

    def _string_pair_sum_many(
        self,
        weighted_observables: List[Tuple[complex, List[torch.Tensor]]],
        contraction: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum over observables and full CZ-expansion string pairs.

        :param weighted_observables: ``(weight, vectors)`` pairs, one per observable monomial.
        :param contraction: Contraction matrix of shape ``(..., D, D)``.
        :return: Complex sum of shape ``(...)``.
        :rtype: torch.Tensor
        """
        return self._bucketed_jobs_sum([(self._bra_groups, self._ket_groups)], weighted_observables, contraction)

    def _stack_sandwich(self, bras: torch.Tensor, observable: torch.Tensor, kets: torch.Tensor) -> torch.Tensor:
        """Stack every bra-observable-ket sandwich of one length bucket along the pair axis.

        Broadcasts the ``n_bras`` bras against the ``n_kets`` kets (with the shared observable in the
        middle) so every string pair becomes one row of the returned factor stack.

        :param bras: Bra factors of shape ``(n_bras, *batch, bra_length, D)``.
        :param observable: Observable factors of shape ``(*batch, n_observable, D)``.
        :param kets: Ket factors of shape ``(n_kets, *batch, ket_length, D)``.
        :return: Stacked factors of shape ``(n_bras * n_kets, *batch, length, D)``.
        :rtype: torch.Tensor
        """
        batch_shape = self._batch_shape()
        n_bras, bra_length = bras.shape[0], bras.shape[-2]
        n_kets, ket_length = kets.shape[0], kets.shape[-2]
        n_observable = observable.shape[-2]
        length = bra_length + n_observable + ket_length
        return torch.cat(
            [
                bras[:, None].expand(n_bras, n_kets, *batch_shape, bra_length, self.dim),
                observable[None, None].expand(n_bras, n_kets, *batch_shape, n_observable, self.dim),
                kets[None, :].expand(n_bras, n_kets, *batch_shape, ket_length, self.dim),
            ],
            dim=-2,
        ).reshape(n_bras * n_kets, *batch_shape, length, self.dim)

    def _bucketed_jobs_sum(
        self,
        jobs: List[Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]],
        weighted_observables: List[Tuple[complex, List[torch.Tensor]]],
        contraction: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of ``w conj(c_a) c_b <bra_a | obs | ket_b>`` over jobs, observables, and string pairs.

        Every (job, observable, bra group, ket group) sandwich of a given total length is stacked
        into one tensor, so the whole sum costs one batched Pfaffian call per distinct sandwich
        length: the Pfaffian eliminates columns sequentially, so few large batches beat many small
        ones.

        :param jobs: ``(bra_groups, ket_groups)`` pairs of per-length string groups.
        :param weighted_observables: ``(weight, vectors)`` pairs, one per observable monomial.
        :param contraction: Contraction matrix of shape ``(..., D, D)``.
        :return: Complex sum of shape ``(...)``.
        :rtype: torch.Tensor
        """
        batch_shape = self._batch_shape()
        total = torch.zeros(batch_shape, dtype=self._complex_dtype, device=self._device)
        observables = [
            (observable_weight, self._vectors_to_tensor(observable_vectors))
            for observable_weight, observable_vectors in weighted_observables
        ]
        buckets: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for (job_bra_groups, job_ket_groups), (observable_weight, observable) in itertools.product(jobs, observables):
            n_observable = observable.shape[-2]
            for (bra_length, (bra_coefficients, bras)), (ket_length, (ket_coefficients, kets)) in itertools.product(
                job_bra_groups.items(), job_ket_groups.items()
            ):
                length = bra_length + n_observable + ket_length
                if length % 2 == 1:
                    continue
                weights = observable_weight * (
                    torch.conj(bra_coefficients)[:, None] * ket_coefficients[None, :]
                ).reshape(-1)
                if length == 0:
                    total = total + weights.sum() * torch.ones(
                        batch_shape, dtype=self._complex_dtype, device=self._device
                    )
                    continue
                bucket_weights, bucket_stacks = buckets.setdefault(length, ([], []))
                bucket_weights.append(weights)
                bucket_stacks.append(self._stack_sandwich(bras, observable, kets))
        for length, (bucket_weights, bucket_stacks) in buckets.items():
            weights = torch.cat(bucket_weights, dim=0)  # (P,)
            stacked = torch.cat(bucket_stacks, dim=0)  # (P, *batch, length, D)
            pfaffians = self._stacked_wick_pfaffians(stacked, contraction)  # (P, *batch)
            total = total + (weights.reshape(-1, *([1] * len(batch_shape))) * pfaffians).sum(dim=0)
        return total

    def _amplitude_sum(self, prefix_vectors: List[torch.Tensor], contraction: torch.Tensor) -> torch.Tensor:
        """Sum over strings of ``c_sigma <prefix, l^sigma_1, ..., l^sigma_r>``.

        :param prefix_vectors: Factors placed left of every string (the bit-flip monomial).
        :param contraction: Contraction matrix of shape ``(..., D, D)``.
        :return: Complex sum of shape ``(...)``.
        :rtype: torch.Tensor
        """
        batch_shape = self._batch_shape()
        prefix = self._vectors_to_tensor(prefix_vectors)  # (*batch, f, D)
        n_prefix = prefix.shape[-2]
        total = torch.zeros(batch_shape, dtype=self._complex_dtype, device=self._device)
        for ket_length, (ket_coefficients, kets) in self._ket_groups.items():
            length = n_prefix + ket_length
            if length % 2 == 1:
                continue
            if length == 0:
                total = total + ket_coefficients.sum() * torch.ones(
                    batch_shape, dtype=self._complex_dtype, device=self._device
                )
                continue
            n_kets = kets.shape[0]
            stacked = torch.cat(
                [prefix[None].expand(n_kets, *batch_shape, n_prefix, self.dim), kets],
                dim=-2,
            )
            pfaffians = self._stacked_wick_pfaffians(stacked, contraction)  # (S, *batch)
            weights = ket_coefficients.reshape(n_kets, *([1] * len(batch_shape)))
            total = total + (weights * pfaffians).sum(dim=0)
        return total

    def _get_reference(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Reference lifted basis state with the largest greedy overlap with ``|G>``.

        Chosen mode by mode so that ``p_G(reference) >= 2^{-(n+1)} > 0``, which keeps the
        transition covariance to the reference finite and well-conditioned.

        :return: ``(bits, p_ref, contraction)`` with ``bits`` over the ``n + 1`` lifted modes,
            ``p_ref = |<ref|G>|^2`` of shape ``(...)``, and the reference contraction matrix.
        :rtype: Tuple[np.ndarray, torch.Tensor, torch.Tensor]
        """
        if self._reference is not None:
            return self._reference
        n_modes = self.dim // 2
        single_branch = qml.math.real(self.lam_g)[None]
        unit_weight = np.ones((1, 1))
        bits: List[int] = []
        for mode in range(n_modes):
            measured = list(range(mode + 1))
            probability_zero = torch.as_tensor(
                qml.math.toarray(basis_state_probability(single_branch, unit_weight, np.asarray(bits + [0]), measured))
            )
            probability_one = torch.as_tensor(
                qml.math.toarray(basis_state_probability(single_branch, unit_weight, np.asarray(bits + [1]), measured))
            )
            bits.append(int((probability_one > probability_zero).flatten()[0]))
        reference_bits = np.asarray(bits)
        reference_probability = self._to_complex_tensor(
            basis_state_probability(single_branch, unit_weight, reference_bits, list(range(n_modes)))
        ).real
        lam_reference = _build_lambda_y_block(reference_bits, self.dim, self.lam_g.real.dtype, self._device)
        gamma = transition_cov(lam_reference, self.lam_g.real, marker=None)
        contraction = self._eye - 1j * self._to_complex_tensor(gamma)
        self._reference = (reference_bits, reference_probability, contraction)
        return self._reference

    def _flip_vectors(self, reference_bits: np.ndarray, target_bits: np.ndarray) -> List[torch.Tensor]:
        """Majorana factors of the bit-flip monomial mapping the reference onto the target.

        ``<y| = <ref| prod_{q in flips} X_q`` with ``X_q = Z_0 ... Z_{q-1} c_{2q}``; the scalar
        phase of the monomial is common to every string and cancels in ``|amplitude|^2``.

        :param reference_bits: Reference bits over the lifted modes.
        :param target_bits: Target bits over the lifted modes.
        :return: Ordered unit vectors of the flip monomial.
        :rtype: List[torch.Tensor]
        """
        mismatched_modes = np.nonzero(reference_bits != target_bits)[0]
        indices = [index for mode in mismatched_modes for index in range(2 * int(mode) + 1)]
        return [self._eye[index] for index in indices]

    def _full_state_probability(self, qubit_order_bits: np.ndarray) -> torch.Tensor:
        """Probability of a full physical outcome via lifted amplitudes, summed over the ancilla.

        ``p(y) = p_ref * sum_a |sum_sigma c_sigma <ref| flip_{(y, a)} L_sigma |G>/<ref|G>|^2``;
        only the parity-allowed ancilla bit contributes (the other parity sector vanishes). The
        greedy reference is a per-state choice, so batched engines evaluate element by element
        (a reference chosen from one batch element can be near-orthogonal to another element's
        Gaussian); expectation values and marginals are reference-free and stay fully batched.

        :param qubit_order_bits: Outcome bits of every physical qubit, in qubit order.
        :return: Probability of shape ``(...)``.
        :rtype: torch.Tensor
        """
        batch_shape = self._batch_shape()
        if batch_shape:
            probabilities = [engine._full_state_probability(qubit_order_bits) for engine in self._get_element_engines()]
            return torch.stack(probabilities).reshape(batch_shape)
        reference_bits, reference_probability, contraction = self._get_reference()
        probability = torch.zeros(batch_shape, dtype=reference_probability.dtype, device=self._device)
        for ancilla_bit in (0, 1):
            target_bits = np.concatenate([qubit_order_bits, [ancilla_bit]])
            flip = self._flip_vectors(reference_bits, target_bits)
            amplitude = self._amplitude_sum(flip, contraction)
            probability = probability + reference_probability * torch.abs(amplitude) ** 2
        return probability

    def _get_element_engines(self) -> List["CzStringEngine"]:
        """Unbatched engines for every batch element, sliced from the constructor inputs.

        :return: One engine per element of the batch shape, in ``np.ndindex`` order.
        :rtype: List[CzStringEngine]
        """
        if self._element_engines is None:
            self._element_engines = [
                CzStringEngine(
                    self.lam_g[index],
                    [
                        (qubit_j, qubit_k, self._slice_batch(prefix, index))
                        for qubit_j, qubit_k, prefix in self._cz_events
                    ],
                    sptm_total=self._slice_batch(self._sptm_total, index),
                )
                for index in np.ndindex(*self._batch_shape())
            ]
        return self._element_engines

    def _marginal_probability(self, bits: np.ndarray, measured_qubits: List[int]) -> torch.Tensor:
        """Marginal probability via the ``Z``-expansion of the projector (``2^k`` Pauli terms).

        ``P_y = prod_i (I + (1 - 2 y_i) Z_{q_i}) / 2`` with ``Z_q = -i c_{2q} c_{2q+1}``, so every
        term is an even Majorana monomial evaluated by the string-pair Wick sum against
        ``Lambda_G``.

        :param bits: Outcome bits of the measured qubits.
        :param measured_qubits: Qubit indices the bits refer to.
        :return: Probability of shape ``(...)``.
        :rtype: torch.Tensor
        """
        contraction = self._get_contraction_g()
        weighted_observables = self._marginal_weighted_observables(bits, measured_qubits)
        total = self._string_pair_sum_many(weighted_observables, contraction)
        return qml.math.real(total) / 2.0 ** len(measured_qubits)

    def _hamiltonian_weighted_observables(
        self, observable: Operator, wires: List
    ) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Weighted Majorana monomials of a Pauli-sum observable (marker appended to odd terms).

        :param observable: A Pauli-decomposable observable (exposing ``observable.terms()``).
        :param wires: Device wire labels in qubit order.
        :return: ``(weight, vectors)`` pairs, one per Pauli term.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        wires = list(wires)
        jordan_wigner = JordanWigner(len(wires))
        wire_map = {wire: index for index, wire in enumerate(wires)}
        try:
            coefficients, operators = observable.terms()
        except TermsUndefinedError:
            coefficients, operators = [1.0], [observable]  # a bare Pauli word has no terms() decomposition

        weighted_observables: List[Tuple[complex, List[torch.Tensor]]] = []
        for coefficient, operator in zip(coefficients, operators):
            pauli_str = pauli_word_to_string(operator, wire_map=wire_map)
            support, kappa = jordan_wigner.pauli_to_majorana(pauli_str, wires)
            observable_vectors = [self._eye[index] for index in support]
            if len(support) % 2 == 1:
                observable_vectors = observable_vectors + [self._eye[self.marker]]
            coefficient_value = complex(coefficient.item() if isinstance(coefficient, torch.Tensor) else coefficient)
            weighted_observables.append((coefficient_value * kappa, observable_vectors))
        return weighted_observables

    def _marginal_weighted_observables(
        self, bits: np.ndarray, measured_qubits: List[int]
    ) -> List[Tuple[complex, List[torch.Tensor]]]:
        """Weighted Majorana monomials of the ``2^k`` Pauli-``Z`` words of a marginal projector.

        ``P_y = prod_i (I + (1 - 2 y_i) Z_{q_i}) / 2`` with ``Z_q = -i c_{2q} c_{2q+1}``; the
        overall ``2^{-k}`` is left to the caller.

        :param bits: Outcome bits of the measured qubits.
        :param measured_qubits: Qubit indices the bits refer to.
        :return: ``(weight, vectors)`` pairs, one per subset of measured qubits.
        :rtype: List[Tuple[complex, List[torch.Tensor]]]
        """
        per_qubit_terms = [
            [
                (1.0 + 0.0j, []),
                (complex((1 - 2 * bits[position]) * (-1j)), [self._eye[2 * qubit], self._eye[2 * qubit + 1]]),
            ]
            for position, qubit in enumerate(measured_qubits)
        ]
        return self._expand_product_strings(per_qubit_terms)
