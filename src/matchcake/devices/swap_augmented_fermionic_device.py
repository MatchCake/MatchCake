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
    SwapBranchState,
    basis_state_probability,
    hamiltonian_expval,
    lift_from_product_state,
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
    trailing ``fSWAP`` matchgate; both contribute to the ``chi <= 2^m`` branch count. Observables
    (``probs``, ``expval``) are read off the branch tensor; sampling, batching, and the rest of the
    plumbing are inherited from NIF unchanged.

    The branch-tensor observables are correct for matchgate circuits with no ``SWAP`` (then identical
    to ``nif.qubit``), a single ``SWAP``, or ``SWAP``s acting on disjoint wire pairs. ``SWAP``s that
    share a wire can drive two branches to be orthogonal, which the overlap-normalized observable
    formula does not yet handle; see ``swap_injection_theory.md`` section 12.

    See ``docs/swap_injection_theory.md`` for the derivation.
    """

    name = "nif.swap.qubit"

    _supported_ops = NonInteractingFermionicDevice._supported_ops | {"SWAP", "CZ"}
    DEFAULT_CONTRACTION_METHOD = None  # SWAPs are barriers; matchgates accumulate in the global SPTM

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
                j, k = (self.wires.index(wire) for wire in op.wires)
                self._branch_state.apply_swap(j, k)
                continue
            if isinstance(op, qml.CZ):
                self._flush_sptm_to_branches()
                j, k = (self.wires.index(wire) for wire in op.wires)
                self._branch_state.apply_cz(j, k)
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
            return basis_state_probability(branch.cov, branch.weights, target, measured_qubits)

        wires_array = np.broadcast_to(np.asarray(wires), target.shape)
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
        """Reset the device, discarding the branch state.

        :return: None
        """
        super().reset()
        self._branch_state = None

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
        self._branch_state = SwapBranchState(cov_tensor, weights, lifted=True)

    def _flush_sptm_to_branches(self) -> None:
        """Apply the accumulated matchgate SPTM to every branch and reset the accumulator.

        :return: None
        """
        self._ensure_branch_state()
        if self._global_sptm is not None:
            sptm = self._global_sptm.matrix(self.wires)  # (2n, 2n) physical block (or batched)
            self._branch_state.apply_matchgate_sptm(sptm)
            self._global_sptm = None
            self._transition_matrix = None
            self._lookup_table = None

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
