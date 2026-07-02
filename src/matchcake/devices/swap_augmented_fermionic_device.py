from typing import Iterable, List, Optional, Union

import numpy as np
import pennylane as qml
from pennylane import BasisState
from pennylane.exceptions import DeviceError
from pennylane.operation import StatePrepBase
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.wires import Wires

from .. import utils
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..observables.batch_projector import BatchProjector
from ..operations.state_preparation import ProductState
from ..typing import TensorLike
from .expval_strategies.m_pfaffian._extended_covariance import displacement_vector
from .nif_device import NonInteractingFermionicDevice
from .swap_injection import (
    CzStringEngine,
    SwapBranchState,
    basis_state_probability,
    hamiltonian_expval,
    lift_from_product_state,
    lift_sptm,
)


class SwapAugmentedFermionicDevice(NonInteractingFermionicDevice):
    r"""Matchgate simulator with genuine qubit ``SWAP`` support via the branch-tensor formalism.

    Matchgates + ``SWAP`` is universal (Jozsa-Miyake), so a circuit with ``m`` injected ``SWAP``s is
    no longer free fermions; it is simulated as a sum of ``chi <= 2^m`` fermionic Gaussian branches
    (covariance tensor ``(chi, D, D)`` plus a Hermitian weight matrix ``W``), classically efficient
    while ``m`` is small. This device is a strict superset of :class:`NonInteractingFermionicDevice`:
    on a circuit with no ``SWAP`` it reproduces ``nif.qubit`` exactly.

    The initial product state is lifted once to the even ``(2n+2)`` parity-purified frame (so
    arbitrary product-state inputs work, not just basis states) and then propagated: matchgate layers
    accumulate into a single SPTM that is flushed onto every branch, and each ``SWAP`` branches the
    state. A genuine qubit ``CZ`` is the non-Gaussian factor of a ``SWAP`` (``CZ = 1 - 2 n_j n_k``,
    while ``SWAP = fSWAP . CZ``), so it branches the state identically to a ``SWAP`` but without the
    trailing ``fSWAP`` matchgate; both contribute to the ``chi <= 2^m`` branch count. A ``SWAP`` on
    non-adjacent wires also carries the parity phase of every crossed mode,
    ``SWAP_{jk} = M_{jk} CZ_{jk} prod_{j<l<k} CZ_{jl} CZ_{kl}`` with ``M_{jk}`` the (Gaussian)
    fermionic mode swap, so it costs ``2|j - k| - 1`` genuine ``CZ`` branchings; a non-adjacent
    ``CZ`` costs one, like an adjacent one. Observables
    (``probs``, ``expval``) are read off the branch tensor; sampling, batching, and the rest of the
    plumbing are inherited from NIF unchanged.

    The branch-tensor observables divide by pairwise branch overlaps, which is ill-defined when
    ``SWAP``s that share a wire drive two branches exactly orthogonal. The device tracks the
    affected branch pairs at every branching (see :attr:`SwapBranchState.string_pair_mask`) and
    evaluates them overlap-free through the :class:`CzStringEngine` (the Heisenberg
    ``CZ``-expansion of ``swap_injection_theory.md`` section 11), so ``probs`` and ``expval`` are
    exact for arbitrary ``SWAP`` placement. Healthy pairs keep the fast overlap-normalized sum
    (hybrid per-pair dispatch); when most pairs are masked, the full engine takes over, and full
    outcome probabilities always use its per-branch amplitude path.

    See ``docs/swap_injection_theory.md`` for the derivation.
    """

    name = "nif.swap.qubit"

    _supported_ops = NonInteractingFermionicDevice._supported_ops | {"SWAP", "CZ"}
    DEFAULT_CONTRACTION_METHOD = None  # SWAPs are barriers; matchgates accumulate in the global SPTM

    @staticmethod
    def _mode_swap_sptm(j: int, k: int, num_wires: int) -> np.ndarray:
        """SPTM of the fermionic mode swap ``M_{jk}``: the exchange of the Majorana pairs of ``j`` and ``k``.

        Valid for any (also non-adjacent) pair: the mode swap is string-free in the Majorana
        algebra. Note ``SptmCompZX(wires=[j, k]).pad(...)`` is not usable here for non-adjacent
        wires (its padding produces a cycle over the intermediate modes, not the pair exchange).

        :param j: First qubit index.
        :param k: Second qubit index.
        :param num_wires: Number of physical qubits.
        :return: Orthogonal permutation SPTM of shape ``(2n, 2n)``.
        :rtype: np.ndarray
        """
        permutation = np.arange(2 * num_wires)
        permutation[[2 * j, 2 * j + 1, 2 * k, 2 * k + 1]] = [2 * k, 2 * k + 1, 2 * j, 2 * j + 1]
        return np.eye(2 * num_wires)[permutation]

    @staticmethod
    def _normalize_target_states(target_binary_states: TensorLike, num_wires: int) -> np.ndarray:
        """Normalize a single/batched basis outcome to an integer ndarray.

        :param target_binary_states: Outcome(s) as an ``int``, binary ``str``, list, or array.
        :param num_wires: Number of bits an integer outcome is expanded to.
        :return: Integer array of shape ``(k,)`` (single) or ``(B, k)`` (batch).
        :rtype: np.ndarray
        """
        if isinstance(target_binary_states, int):
            return utils.binary_string_to_vector(utils.state_to_binary_string(target_binary_states, num_wires))
        if isinstance(target_binary_states, str):
            return utils.binary_string_to_vector(target_binary_states)
        return np.asarray(target_binary_states).astype(int)

    def __init__(
        self,
        wires: Optional[Union[int, Wires, List[int]]] = None,
        *,
        shots: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(wires, shots=shots, **kwargs)
        self._branch_state: Optional[SwapBranchState] = None
        self._lifted_input_cov: Optional[TensorLike] = None
        self._sptm_prefix: Optional[TensorLike] = None
        self._cz_events: List[tuple] = []
        self._string_engine: Optional[CzStringEngine] = None

    def apply_generator(
        self, op_iterator: Iterable[qml.operation.Operation], **kwargs
    ) -> "SwapAugmentedFermionicDevice":
        """Apply operations, routing each ``SWAP`` to the branch tensor.

        Matchgate layers accumulate into the global SPTM (reusing
        :meth:`NonInteractingFermionicDevice.apply_op`); at every ``SWAP`` the accumulated SPTM is
        flushed onto every branch and the branching step is applied. The trailing accumulation is
        flushed lazily on the first observable access.

        :param op_iterator: The operations to apply.
        :return: ``self``.
        :rtype: SwapAugmentedFermionicDevice
        """
        ops = list(op_iterator)
        if self._wires is None:
            all_wires = Wires.all_wires([op.wires for op in ops if len(op.wires) > 0])
            assert len(all_wires) > 1, "At least two wires are required for this device."
            self._wires = all_wires
            self._setup_wire_dependent_state()

        for index, op in enumerate(ops):
            if isinstance(op, qml.Identity):
                continue
            if self.apply_state_prep(op, index=index):
                continue
            if isinstance(op, qml.SWAP):
                self._flush_sptm_to_branches()
                j, k = sorted(self.wires.index(wire) for wire in op.wires)
                # SWAP_{jk} = M_{jk} CZ_{jk} prod_{j<l<k} CZ_{jl} CZ_{kl}: the qubit swap is the
                # fermionic mode swap M_{jk} times the parity phase of every crossed mode
                # (-1)^{(n_j + n_k) n_l + n_j n_k}, so a SWAP at distance d costs 2d - 1 genuine
                # CZ branchings. For adjacent wires this reduces to the usual fSWAP . CZ.
                for middle in range(j + 1, k):
                    self._apply_cz_event(j, middle)
                    self._apply_cz_event(k, middle)
                self._apply_cz_event(j, k)
                mode_swap_sptm = self._mode_swap_sptm(j, k, self.num_wires)
                self._branch_state.apply_matchgate_sptm(mode_swap_sptm)
                self._sptm_prefix = self.update_single_particle_transition_matrix(self._sptm_prefix, mode_swap_sptm)
                continue
            if isinstance(op, qml.CZ):
                self._flush_sptm_to_branches()
                j, k = sorted(self.wires.index(wire) for wire in op.wires)
                self._apply_cz_event(j, k)
                continue
            self.apply_op(self.convert_op_to_supported(op))
        return self

    def exact_expval(self, observable: qml.operation.Operator) -> TensorLike:
        """Expectation value via the branch tensor for Pauli observables, NIF fallback otherwise.

        :param observable: The observable to measure.
        :return: The expectation value.
        :rtype: TensorLike
        """
        # Pauli observables go to the branch tensor; BatchHamiltonian (recurses per term), projectors
        # (route to probability), and any non-Pauli observable fall back to the NIF implementation.
        is_routed = isinstance(observable, (BatchHamiltonian, BasisStateProjector, BatchProjector))
        if not is_routed and self.m_pfaffian_expval_strategy.can_execute(self.state_prep_op, observable):
            branch = self.branch_state
            if branch.degenerate:
                return self._degenerate_expval(observable)
            return hamiltonian_expval(branch.cov, branch.weights, observable, list(self.wires), marker=branch.marker)
        return super().exact_expval(observable)

    def get_states_probability(
        self,
        target_binary_states: TensorLike,
        wires: Optional[Wires] = None,
        **kwargs,
    ) -> TensorLike:
        """Probabilities of one or a batch of basis outcomes from the branch tensor.

        :param target_binary_states: Binary outcome(s) of shape ``(k,)`` (single) or ``(B, k)`` (batch).
        :param wires: Measured wires. Defaults to all device wires.
        :return: Scalar for a single outcome, ``(B,)`` for a batch.
        :rtype: TensorLike
        """
        branch = self.branch_state
        target = self._normalize_target_states(target_binary_states, self.num_wires)
        if wires is None:
            wires = self.wires

        if target.ndim == 1:
            measured_qubits = [self.wires.index(wire) for wire in Wires(wires)]
            if branch.degenerate:
                return self._degenerate_probability(target, measured_qubits)
            return basis_state_probability(branch.cov, branch.weights, target, measured_qubits)

        wires_array = np.broadcast_to(np.asarray(wires), target.shape)
        if branch.degenerate:
            probabilities = [
                self._degenerate_probability(
                    target[index],
                    [self.wires.index(wire) for wire in wires_array[index]],
                )
                for index in range(target.shape[0])
            ]
        else:
            probabilities = [
                basis_state_probability(
                    branch.cov,
                    branch.weights,
                    target[index],
                    [self.wires.index(wire) for wire in wires_array[index]],
                )
                for index in range(target.shape[0])
            ]
        return qml.math.stack(probabilities)

    def reset(self) -> None:
        """Reset the device, discarding the branch state and the recorded ``CZ`` events.

        :return: None
        """
        super().reset()
        self._branch_state = None
        self._lifted_input_cov = None
        self._sptm_prefix = None
        self._cz_events = []
        self._string_engine = None

    def _ensure_branch_state(self) -> None:
        """Build the initial single-branch lifted state from the product-state input if needed.

        :return: None
        :raises DeviceError: if the state preparation is neither a ``BasisState`` nor a ``ProductState``.
        """
        if self._branch_state is not None:
            return
        state_prep: StatePrepBase = self.state_prep_op
        if isinstance(state_prep, BasisState):
            state_prep = ProductState.from_basis_state(state_prep)
        if not isinstance(state_prep, ProductState):
            raise DeviceError(f"{self.name} requires a ProductState or BasisState input, got {type(state_prep)}.")
        amplitudes = qml.math.cast(state_prep.data[0], self._c_dtype_name)  # (n, 2)
        covariance = qml.math.cast(state_prep.covariance_matrix, self._r_dtype_name)  # (2n, 2n)
        displacement = displacement_vector(amplitudes, self.wires)  # (2n,)
        lifted = lift_from_product_state(covariance, displacement)  # (2n+2, 2n+2)
        cov_tensor = lifted[None, ...]  # (1, 2n+2, 2n+2)
        weights = qml.math.convert_like(qml.math.cast(np.array([[1.0]]), self._c_dtype_name), cov_tensor)  # (1, 1)
        self._lifted_input_cov = lifted
        self._branch_state = SwapBranchState(cov_tensor, weights, lifted=True)

    def _flush_sptm_to_branches(self) -> None:
        """Apply the accumulated matchgate SPTM to every branch and reset the accumulator.

        The flushed SPTM is also composed into ``_sptm_prefix``, the running product of every
        matchgate layer since the circuit start that the string engine needs to Heisenberg-rotate
        the recorded ``CZ`` insertions.

        :return: None
        """
        self._ensure_branch_state()
        if self._global_sptm is not None:
            sptm = self._global_sptm.matrix(self.wires)  # (2n, 2n) physical block (or batched)
            self._branch_state.apply_matchgate_sptm(sptm)
            self._sptm_prefix = self.update_single_particle_transition_matrix(self._sptm_prefix, sptm)
            self._string_engine = None
            self._global_sptm = None
            self._transition_matrix = None
            self._lookup_table = None

    def _degenerate_expval(self, observable: qml.operation.Operator) -> TensorLike:
        """Expectation value when some branch pairs are masked: hybrid per-pair or full engine.

        The healthy pairs keep the overlap-normalized branch-tensor sum (with the masked pairs
        excluded), and the masked pairs are added back overlap-free from the branch histories;
        when most pairs are masked the full ``CZ``-expansion engine is cheaper and used instead.

        :param observable: The observable to measure.
        :return: The expectation value.
        :rtype: TensorLike
        """
        branch = self.branch_state
        pair_mask = branch.string_pair_mask
        if self._prefer_full_engine(pair_mask):
            return self.string_engine.hamiltonian_expval(observable, list(self.wires))
        healthy = hamiltonian_expval(
            branch.cov, branch.weights, observable, list(self.wires), marker=branch.marker, pair_mask=pair_mask
        )
        masked = self.string_engine.branch_pair_hamiltonian_expval(
            observable, list(self.wires), branch.histories, pair_mask
        )
        return healthy + masked

    def _degenerate_probability(self, target_state: TensorLike, measured_qubits: List[int]) -> TensorLike:
        """Outcome probability when some branch pairs are masked.

        Full outcomes go through the engine's amplitude path (per-branch amplitudes never form a
        pair overlap, at ``4^m`` strings); marginals use the hybrid per-pair split or, when most
        pairs are masked, the engine's ``Z``-expansion.

        :param target_state: Outcome bits of the measured qubits.
        :param measured_qubits: Qubit indices the bits refer to.
        :return: The probability.
        :rtype: TensorLike
        """
        if sorted(measured_qubits) == list(range(self.num_wires)):
            return self.string_engine.basis_state_probability(target_state, measured_qubits)
        branch = self.branch_state
        pair_mask = branch.string_pair_mask
        if self._prefer_full_engine(pair_mask):
            return self.string_engine.basis_state_probability(target_state, measured_qubits)
        healthy = basis_state_probability(
            branch.cov, branch.weights, target_state, measured_qubits, pair_mask=pair_mask
        )
        masked = self.string_engine.branch_pair_marginal_probability(
            target_state, measured_qubits, branch.histories, pair_mask
        )
        return healthy + masked

    def _prefer_full_engine(self, pair_mask: np.ndarray) -> bool:
        """Whether the full ``CZ``-expansion engine beats the hybrid per-pair evaluation.

        The hybrid costs ``sum 4^(s_a + s_b)`` string pairs over the masked branch pairs, against
        ``16^m`` for the engine; branch enumeration loses once most pairs are masked.

        :param pair_mask: Boolean ``(chi, chi)`` mask of the branch pairs needing string evaluation.
        :return: True when the engine is the cheaper evaluation.
        :rtype: bool
        """
        branch = self.branch_state
        hybrid_cost = CzStringEngine.branch_pair_cost(branch.histories, pair_mask)
        return 16 ** len(self._cz_events) <= hybrid_cost

    def _apply_cz_event(self, j: int, k: int) -> None:
        """Apply a genuine ``CZ`` to the branch state and record it for the string engine.

        :param j: First qubit index.
        :param k: Second qubit index.
        :return: None
        """
        self._cz_events.append((j, k, self._sptm_prefix))
        self._string_engine = None
        self._branch_state.apply_cz(j, k)

    @property
    def branch_state(self) -> SwapBranchState:
        """The current branch state, flushing any accumulated matchgate SPTM first.

        :return: The branch state.
        :rtype: SwapBranchState
        """
        self._flush_sptm_to_branches()
        return self._branch_state

    @property
    def branch_covariances(self) -> TensorLike:
        """The ``(chi, D, D)`` branch covariance tensor (shadows NIF's single-matrix property).

        :return: The branch covariance tensor.
        :rtype: TensorLike
        """
        return self.branch_state.cov

    @property
    def string_engine(self) -> CzStringEngine:
        """The overlap-free ``CZ``-expansion evaluator, built lazily from the recorded events.

        Used for observables whenever the branch state is degenerate (two branches
        (near-)orthogonal); exact for arbitrary ``SWAP`` placement.

        :return: The string engine.
        :rtype: CzStringEngine
        """
        self._flush_sptm_to_branches()
        if self._string_engine is None:
            lifted_input = self._lifted_input_cov  # (2n+2, 2n+2)
            if self._sptm_prefix is None:
                lifted_covariance = lifted_input
            else:
                lifted_total = lift_sptm(self._sptm_prefix)  # (..., 2n+2, 2n+2)
                lifted_transposed = qml.math.swapaxes(lifted_total, -1, -2)
                lifted_covariance = qml.math.einsum("...ij,jk->...ik", lifted_transposed, lifted_input)
                lifted_covariance = qml.math.einsum("...ij,...jk->...ik", lifted_covariance, lifted_total)
            self._string_engine = CzStringEngine(lifted_covariance, self._cz_events, sptm_total=self._sptm_prefix)
        return self._string_engine
