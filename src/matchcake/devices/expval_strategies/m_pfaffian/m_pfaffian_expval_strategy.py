import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Optional, cast

import numpy as np
import pennylane as qml
import torch
from pennylane import BasisState
from pennylane.operation import Operator, StatePrepBase, TermsUndefinedError
from pennylane.pauli import pauli_word_to_string

from ....operations.state_preparation.product_state import ProductState
from ....typing import TensorLike
from ....utils import JordanWigner
from ....utils._pfaffian import infer_real_dtype, sector_pfaffian_features
from ....utils.math import convert_and_cast_like
from ..expval_strategy import ExpvalStrategy


def extend_majorana_indices(
    mu: np.ndarray,
    alpha: complex,
    parity_index: int,
) -> tuple[np.ndarray, complex]:
    """Return (ext_index_set, ext_phase) per the parity rule.

    For rank even:  ext_index_set = mu,                          ext_phase = alpha.
    For rank odd:   ext_index_set = concat(mu, [parity_index]),
                    ext_phase = alpha * i^rank * (-1)^{rank(rank-1)/2}.

    Since parity_index = 2n is the largest index, appending it keeps ext_index_set sorted.
    """
    rank = len(mu)
    if rank % 2 == 0:
        return mu, alpha
    extra_phase = (1j) ** rank * (-1) ** ((rank * (rank - 1)) // 2)
    return np.concatenate([mu, [parity_index]]), alpha * extra_phase


class _DecompositionCache:
    """Memoizes the Pauli -> Majorana decomposition of an observable.

    Which Majorana index set each term maps to, its sector and its constant phase
    depend only on the observable's Pauli operators and the device wires -- never
    on the per-forward covariance matrix. For a fixed circuit/Hamiltonian this is
    recomputed on every forward pass, which dominates the runtime when the Pfaffian
    kernel is cheap (e.g. float32 internals). This class memoizes it.

    Both tiers cache only the *coefficient-independent* structure; the coefficients
    are read live from ``observable.terms()`` on every call (which is cheap and does
    not rebuild anything) and folded in afterwards. An observable whose coefficients
    change in place between calls -- even when the same object is reused -- therefore
    always produces correct values, never a stale cached coefficient.

    Two tiers, both bounded (LRU) and lock-protected:
      * ``_id_cache``: keyed by ``id(observable)`` and *validated by a weak
        reference* to the original object. A bare ``id()`` key is unsafe -- once an
        object is garbage-collected its id can be reused by an unrelated object,
        which would return a stale (wrong) entry. Checking ``wref() is observable``
        makes a hit impossible unless the *same live object* is presented again, so
        id reuse can never produce a false hit. A hit skips both the Pauli-word
        string construction (the structural key) and the Majorana decomposition.
      * ``_struct_cache``: keyed structurally by the tuple of Pauli-word strings
        (relative to the device wire order) plus the wires and matrix size. Correct
        even when callers pass freshly built observable objects every forward, or
        reuse a Pauli structure with different coefficients.

    State is held at the class level, so a single cache is shared across all
    strategy instances within a process. Under multiprocessing ``spawn`` each
    process gets its own class state, so the caches are never shared across
    processes; the lock guards concurrent threads.
    """

    MAXSIZE = 64
    _lock = threading.Lock()
    # Both caches map to the same coefficient-independent structure:
    #   {sector: (index_sets, ext_phases, term_indices, wick_phase)}
    _struct_cache: "OrderedDict[tuple, dict]" = OrderedDict()  # struct_key -> structure
    _id_cache: "OrderedDict[int, tuple]" = OrderedDict()  # id(obs) -> (weakref, context_sig, structure)

    @classmethod
    def clear(cls) -> None:
        """Empty both cache tiers (primarily for tests)."""
        with cls._lock:
            cls._struct_cache.clear()
            cls._id_cache.clear()

    @classmethod
    def is_decomposed(cls, observable: Operator) -> bool:
        """Return True iff ``observable`` has a live entry in the identity cache.

        Validated by weakref identity, so a recycled id can never yield a false
        positive. A True result means the observable is Pauli-decomposable.
        """
        with cls._lock:
            entry = cls._id_cache.get(id(observable))
            return entry is not None and entry[0]() is observable

    @staticmethod
    def _coeff_to_complex(coeff: Any) -> complex:
        """Pull a Hamiltonian coefficient out as a plain Python complex (severs grad).

        Mirrors the original per-term ``complex(h_coeff.item() if Tensor else h_coeff)``;
        coefficients are treated as constants exactly as before, so gradients flow only
        through the covariance matrix.
        """
        if isinstance(coeff, torch.Tensor):
            return complex(coeff.item())
        return complex(coeff)

    @staticmethod
    def _build_structure(
        pauli_strs: tuple,
        device_wires: list,
        parity_index: int,
        n_qubits: int,
    ) -> dict:
        """Coefficient-independent decomposition grouped by parity sector.

        Returns ``{sector: (index_sets, ext_phases, term_indices, wick_phase)}`` where
        ``index_sets`` is ``(n_terms_in_sector, sector)`` int, ``ext_phases`` is the
        complex phase per term, ``term_indices`` maps back to the Hamiltonian term
        order, and ``wick_phase = (-1j)**(sector//2)``.
        """
        jw = JordanWigner(n_qubits)
        by_sector: dict[int, tuple[list, list, list]] = {}
        for term_idx, pauli_str in enumerate(pauli_strs):
            mu, alpha = jw.pauli_to_majorana(pauli_str, device_wires)
            ext_index_set, ext_phase = extend_majorana_indices(mu, alpha, parity_index)
            sector = len(ext_index_set)
            idx_list, phase_list, term_list = by_sector.setdefault(sector, ([], [], []))
            idx_list.append(ext_index_set)
            phase_list.append(ext_phase)
            term_list.append(term_idx)

        structure: dict[int, tuple] = {}
        for sector, (idx_list, phase_list, term_list) in by_sector.items():
            index_sets = np.stack(idx_list)  # (n_terms, sector); (n_terms, 0) for sector 0
            ext_phases = np.asarray(phase_list, dtype=complex)
            term_indices = np.asarray(term_list, dtype=int)
            wick_phase = (-1j) ** (sector // 2)
            structure[sector] = (index_sets, ext_phases, term_indices, wick_phase)
        return structure

    @classmethod
    def _build_payload(cls, structure: dict, h_coeffs: Any) -> "OrderedDict[int, tuple]":
        """Bake the (constant) coefficients into per-sector ``scalar_coeffs``.

        ``scalar_coeffs[k] = Re(coeff_k * ext_phase_k * wick_phase)`` -- exactly the
        real scalar the original per-term loop computed, just vectorized.
        """
        payload: "OrderedDict[int, tuple]" = OrderedDict()
        for sector, (index_sets, ext_phases, term_indices, wick_phase) in structure.items():
            coeffs = np.asarray([cls._coeff_to_complex(h_coeffs[int(t)]) for t in term_indices], dtype=complex)
            scalar_coeffs = np.real(coeffs * ext_phases * wick_phase).astype(np.float64)
            payload[sector] = (index_sets, scalar_coeffs)
        return payload

    @classmethod
    def get_payload(
        cls,
        observable: Operator,
        terms_of: Callable[[Operator], tuple],
        device_wires_tuple: tuple,
        n_total: int,
    ) -> "OrderedDict[int, tuple]":
        """Return ``{sector: (index_sets, scalar_coeffs)}`` for this observable.

        The coefficient-independent structure is taken from the weakref-validated
        identity cache, then the structural cache, and only falls through to the full
        Pauli -> Majorana decomposition on a miss. Coefficients are read live from
        ``terms_of`` on every call (cheap) and folded in last, so a reused object
        whose coefficients change in place never yields a stale value.
        """
        parity_index = n_total - 1
        n_qubits = parity_index // 2
        oid = id(observable)
        context_sig = (device_wires_tuple, n_total)

        h_coeffs, h_ops = terms_of(observable)

        structure = None
        with cls._lock:
            entry = cls._id_cache.get(oid)
            if entry is not None:
                wref, sig, cached_structure = entry
                if sig == context_sig and wref() is observable:
                    cls._id_cache.move_to_end(oid)
                    structure = cached_structure

        if structure is None:
            device_wires = list(device_wires_tuple)
            wire_map = {w: i for i, w in enumerate(device_wires)}
            pauli_strs = tuple(pauli_word_to_string(op, wire_map=wire_map) for op in h_ops)
            struct_key = (pauli_strs, device_wires_tuple, n_total)

            with cls._lock:
                structure = cls._struct_cache.get(struct_key)
                if structure is not None:
                    cls._struct_cache.move_to_end(struct_key)
            if structure is None:
                structure = cls._build_structure(pauli_strs, device_wires, parity_index, n_qubits)
                with cls._lock:
                    cls._struct_cache[struct_key] = structure
                    cls._struct_cache.move_to_end(struct_key)
                    while len(cls._struct_cache) > cls.MAXSIZE:
                        cls._struct_cache.popitem(last=False)

            try:
                wref = weakref.ref(observable)
            except TypeError:
                wref = None
            if wref is not None:
                with cls._lock:
                    cls._id_cache[oid] = (wref, context_sig, structure)
                    cls._id_cache.move_to_end(oid)
                    while len(cls._id_cache) > cls.MAXSIZE:
                        cls._id_cache.popitem(last=False)

        return cls._build_payload(structure, h_coeffs)


class MPfaffianExpvalStrategy(ExpvalStrategy):
    """Compute <P> for arbitrary Pauli observables via the extended-encoding m-Pfaffian.

    Uses the extended (2n+1)-Majorana algebra so that arbitrary qubit product states
    are supported as initial states (not just computational basis states).

    The ``extended_covariance_matrix`` kwarg must be the extended covariance matrix
    of shape (..., 2n+1, 2n+1), laid out with the parity index at position 2n:

        ext_cov_matrix = [[ cov_matrix,  d   ],
                          [ -d^T,        0   ]]

    where cov_matrix is the 2n x 2n standard covariance matrix and d[mu] = <c_mu> is
    the displacement vector for the initial product state.

    The expectation value formula for each Pauli term P = coeff * pauli_op is:

        <P> = Re( coeff * ext_phase * (-i)^{|ext_index_set|/2} * Pf(ext_cov_matrix|_{ext_index_set}) )

    The Pauli -> Majorana decomposition (index sets, sectors, phases) is a pure
    function of the observable's Pauli operators and the device wires, so it is
    memoized (see :class:`_DecompositionCache`); each forward pass folds in the live
    coefficients and re-evaluates only the Pfaffian math against the current
    covariance matrix.

    Mid-circuit gates must all be matchgates (matchgate evolution preserves the
    lift's identity-block structure on the parity index). This is a device-level
    invariant enforced upstream.
    """

    NAME = "MPfaffianExpvalStrategy"

    @staticmethod
    def _terms_of(observable: Operator) -> tuple:
        """Return ``(coeffs, ops)`` for the observable as a sum of Pauli terms.

        Uses ``observable.terms()`` directly (which does not rebuild anything),
        falling back to a trivial single-term wrapping for operators that do not
        define a terms decomposition. Reading the coefficients here every call is
        what keeps the cache correct under in-place coefficient changes.
        """
        try:
            return observable.terms()
        except TermsUndefinedError:
            return (torch.ones(1),), [observable]

    def __call__(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
        **kwargs,
    ) -> TensorLike:
        extended_covariance_matrix: TensorLike = kwargs["extended_covariance_matrix"]
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(
                f"Cannot execute {self.NAME} strategy for observable {observable} with state_prep_op {state_prep_op}."
            )

        ext_cov_matrix = cast(np.ndarray, extended_covariance_matrix)  # (..., 2n+1, 2n+1)
        n_total = ext_cov_matrix.shape[-1]  # 2n + 1
        device_wires_tuple = tuple(sorted(state_prep_op.wires.tolist()))

        payload = _DecompositionCache.get_payload(observable, self._terms_of, device_wires_tuple, n_total)

        r_dtype = infer_real_dtype(ext_cov_matrix)
        ext_cov_matrix_t: Optional[TensorLike] = None  # lazily built for the sector-2 read
        total_re: Any = np.float64(0.0)
        for sector, (index_sets, scalar_coeffs) in payload.items():
            if sector == 0:
                pf_values = torch.ones(len(scalar_coeffs), dtype=r_dtype)
            elif sector == 2:
                if ext_cov_matrix_t is None:
                    ext_cov_matrix_t = torch.as_tensor(qml.math.real(ext_cov_matrix), dtype=r_dtype)
                i_idx = index_sets[:, 0]
                j_idx = index_sets[:, 1]
                pf_values = ext_cov_matrix_t[..., i_idx, j_idx]  # (..., n_terms)
            else:
                pf_values = sector_pfaffian_features(ext_cov_matrix, index_sets, dtype=r_dtype)  # (..., n_terms)

            scalar_coeffs_t = torch.as_tensor(scalar_coeffs, dtype=pf_values.dtype, device=pf_values.device)
            total_re = total_re + pf_values @ scalar_coeffs_t  # (...)

        return convert_and_cast_like(total_re, extended_covariance_matrix)

    def can_execute(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> bool:
        """Return True iff this strategy can compute <observable>.

        Conditions:
          1. state_prep_op is a BasisState or a ProductState.
          2. observable decomposes into a sum of Pauli strings.
        """
        if not isinstance(state_prep_op, (BasisState, ProductState)):
            return False
        # Fast path: an observable already decomposed (and cached) is, by
        # construction, Pauli-decomposable.
        if _DecompositionCache.is_decomposed(observable):
            return True
        try:
            _, ops = self._terms_of(observable)
            for op in ops:
                _ = pauli_word_to_string(op, wire_map={w: i for i, w in enumerate(op.wires)})
        except Exception:
            return False
        return True
