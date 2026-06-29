import numpy as np
import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.ops.qubit.state_preparation import BasisState
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from ...base.lookup_table import NonInteractingFermionicLookupTable
from ...operations.state_preparation import StatePrepFromGates
from ...operations.state_preparation.product_state import ProductState
from .probability_strategy import ProbabilityStrategy


class LookupTableStrategy(ProbabilityStrategy):
    NAME: str = "LookupTable"
    REQUIRES_KWARGS = ["lookup_table"]

    def __call__(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_states: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        """Compute probabilities via the pre-built lookup table.

        Accepts a single outcome ``(k,)`` and returns a scalar, or a batch ``(B, k)``
        and returns ``(B,)``.

        :param state_prep_op: State preparation operation (must be a basis state).
        :type state_prep_op: StatePrepBase
        :param target_binary_states: Binary outcomes of shape ``(k,)`` or ``(B, k)``.
        :type target_binary_states: TensorLike
        :param wires: Measured wires — ``Wires`` for single, ``ndarray(B, k)`` for batch.
        :type wires: Wires
        :return: Scalar or ``(B,)`` probabilities.
        :rtype: TensorLike
        """
        self.check_required_kwargs(kwargs)

        target_arr = np.asarray(target_binary_states)
        is_single = target_arr.ndim == 1

        lookup_table: NonInteractingFermionicLookupTable = kwargs["lookup_table"]
        show_progress = kwargs.get("show_progress", False)
        system_state = self.system_basis_state_from_state_prep_op(state_prep_op)

        if isinstance(wires, int):
            wires = [wires]
        batch_wires = np.asarray(wires)
        all_wires = Wires(kwargs.get("all_wires", Wires(batch_wires.reshape(-1)[: target_arr.shape[-1]])))

        # Convert measured wire labels to Majorana wire indices, matching the target's
        # dimensionality. The lookup table itself squeezes the leading axis for 1-D
        # targets (via its ``initial_ndim`` handling), so no manual squeeze is needed here.
        if is_single:
            wire_indices = np.asarray(all_wires.indices(Wires(batch_wires)))  # (k,)
        else:
            if batch_wires.ndim == 1:
                batch_wires = np.broadcast_to(batch_wires, target_arr.shape)
            same_wires = len(target_arr) <= 1 or np.all(batch_wires == batch_wires[0])
            if same_wires:
                row = np.asarray(all_wires.indices(Wires(batch_wires[0])))
                wire_indices = np.broadcast_to(row, target_arr.shape)  # (B, k)
            else:
                wire_indices = np.array([all_wires.indices(Wires(w)) for w in batch_wires])  # (B, k)

        obs = lookup_table(system_state, target_arr, wire_indices, show_progress=show_progress)
        chunk_size = kwargs.get("pfaffian_chunk_size", None)
        return qml.math.real(utils.pfaffian(obs, sign=False, chunk_size=chunk_size))

    def can_execute(self, state_prep_op: StatePrepBase) -> bool:
        """Return True for basis-state inputs.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :return: True when the state encodes a computational-basis state.
        :rtype: bool
        """
        if isinstance(state_prep_op, (BasisState, StatePrepFromGates)):
            return True
        if isinstance(state_prep_op, ProductState):
            is_basis = state_prep_op.is_basis_state
            return bool(is_basis) if isinstance(is_basis, bool) else bool(qml.math.all(is_basis))
        return False
