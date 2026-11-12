from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, StatePrepBase
from pennylane.wires import Wires, WiresLike

from ..operations.single_particle_transition_matrices import SingleParticleTransitionMatrixOperation
from ..operations.state_preparation import ProductState
from ..typing import TensorLike
from ..utils.math import convert_and_cast_like, convert_tensors_to_same_type_and_cast_to


def merge_blocks(
    *blocks: Callable[[], Any],
    wires: Optional[WiresLike] = None,
    contract: bool = False,
) -> List[Operation]:
    r"""
    Merge circuit blocks acting on disjoint wires into a single circuit.

    Each block is a quantum function called without arguments (use a ``lambda`` or
    :func:`functools.partial` to bind its parameters and wires). A block may start with its own
    state preparations (:class:`~pennylane.BasisState` or
    :class:`~matchcake.operations.state_preparation.ProductState`). Since the blocks act on disjoint
    wires, every state preparation commutes with the gates of the other blocks, so all of them are
    merged into one :class:`~matchcake.operations.state_preparation.ProductState` placed first,
    followed by the gates of every block in order. This is required by the
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`, which accepts a single state
    preparation at the start of the circuit.

    Wires of blocks without a state preparation start in :math:`|0\rangle`. The merged operations are
    queued when called inside a QNode, so the output of the blocks can be used as the input of the
    next circuit:

    .. code-block:: python

        @qml.qnode(qml.device("nif.qubit", wires=4))
        def circuit():
            merge_blocks(lambda: block_a(params_a, wires=[0, 1]), lambda: block_b(params_b, wires=[2, 3]))
            next_block(params_next, wires=[0, 1, 2, 3])
            return qml.probs(wires=[0, 1, 2, 3])

    :param blocks: The quantum functions to merge, each called without arguments.
    :type blocks: Callable[[], Any]
    :param wires: Extra wires the merged state preparation must cover, starting in :math:`|0\rangle`.
        Defaults to the wires of the blocks only. Only used when a block has a state preparation.
    :type wires: Optional[WiresLike]
    :param contract: Whether to contract the gates of all the blocks into a single
        :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`.
    :type contract: bool
    :return: The merged operations: the merged state preparation (if any) followed by the gates.
    :rtype: List[Operation]
    :raises ValueError: If two blocks share a wire, or if a state preparation is not a product state
        or comes after a gate in its block.
    """
    with qml.QueuingManager.stop_recording():
        tapes = [qml.tape.make_qscript(block)() for block in blocks]
        state_preps, gates = _split_blocks(tapes)
        merged_ops: List[Operation] = []
        if state_preps:
            block_wires = Wires.all_wires([tape.wires for tape in tapes])
            extra_wires = Wires([]) if wires is None else Wires(wires)
            merged_ops.append(_merge_state_preps(state_preps, Wires.all_wires([block_wires, extra_wires])))
        if contract and gates:
            merged_ops.append(
                SingleParticleTransitionMatrixOperation.from_circuit(lambda: [qml.apply(g) for g in gates])
            )
        else:
            merged_ops.extend(gates)
    if qml.QueuingManager.recording():
        merged_ops = [qml.apply(op) for op in merged_ops]
    return merged_ops


def _split_blocks(tapes: List[qml.tape.QuantumScript]) -> Tuple[List[ProductState], List[Operation]]:
    """
    Separate the state preparations of the recorded blocks from their gates.

    :param tapes: The recorded blocks.
    :type tapes: List[qml.tape.QuantumScript]
    :return: The state preparations as product states and the gates, both in block order.
    :rtype: Tuple[List[ProductState], List[Operation]]
    :raises ValueError: If two blocks share a wire, or if a state preparation is not a product state
        or comes after a gate in its block.
    """
    state_preps: List[ProductState] = []
    gates: List[Operation] = []
    used_wires = Wires([])
    for block_index, tape in enumerate(tapes):
        shared_wires = Wires.shared_wires([used_wires, tape.wires])
        if len(shared_wires) > 0:
            raise ValueError(f"Blocks must act on disjoint wires, but block {block_index} reuses wires {shared_wires}.")
        used_wires = Wires.all_wires([used_wires, tape.wires])
        has_gate = False
        for op in tape.operations:
            if not isinstance(op, StatePrepBase):
                has_gate = has_gate or not isinstance(op, qml.Identity)
                gates.append(op)
                continue
            if has_gate:
                raise ValueError(
                    f"The state preparation {op} of block {block_index} must come before the gates of its block."
                )
            state_preps.append(_as_product_state(op, f"block {block_index}"))
    return state_preps, gates


def _as_product_state(op: StatePrepBase, owner: str) -> ProductState:
    """
    Convert a state preparation to a product state.

    :param op: The state preparation.
    :type op: StatePrepBase
    :param owner: The block or generator the state preparation comes from, used in error messages.
    :type owner: str
    :return: The state preparation as a product state.
    :rtype: ProductState
    :raises ValueError: If the state preparation is not a product state.
    """
    if isinstance(op, qml.BasisState):
        return ProductState.from_basis_state(op)
    if not isinstance(op, ProductState):
        raise ValueError(
            f"Only product state preparations (BasisState or ProductState) can be merged, but {owner} uses {op}."
        )
    return op


def _merge_state_preps(state_preps: List[ProductState], wires: Wires) -> ProductState:
    r"""
    Merge product states on disjoint wires into a single product state on ``wires``.

    :param state_preps: The product states to merge.
    :type state_preps: List[ProductState]
    :param wires: The wires of the merged state. Wires not covered by ``state_preps`` start in
        :math:`|0\rangle`.
    :type wires: Wires
    :return: The merged product state.
    :rtype: ProductState
    :raises ValueError: If two state preparations share a wire or have different batch sizes.
    """
    prep_wires = Wires.all_wires([op.wires for op in state_preps])
    if len(prep_wires) != sum(len(op.wires) for op in state_preps):
        raise ValueError(f"State preparations must act on disjoint wires, got {state_preps}.")
    # Each amplitude tensor has shape (..., n_k, 2).
    amplitudes = convert_tensors_to_same_type_and_cast_to([op.parameters[0] for op in state_preps])
    zero_wires = Wires([wire for wire in wires if wire not in prep_wires])
    if len(zero_wires) > 0:
        zero_amplitudes = np.zeros((len(zero_wires), 2), dtype=complex)
        zero_amplitudes[:, 0] = 1.0
        amplitudes.append(convert_and_cast_like(zero_amplitudes, amplitudes[0]))
    batch_sizes = {qml.math.shape(amplitude)[0] for amplitude in amplitudes if qml.math.ndim(amplitude) == 3}
    if len(batch_sizes) > 1:
        raise ValueError(f"State preparations must have the same batch size, got {sorted(batch_sizes)}.")
    if batch_sizes:
        batch_size = batch_sizes.pop()
        amplitudes = [_broadcast_to_batch(amplitude, batch_size) for amplitude in amplitudes]
    state = qml.math.concatenate(amplitudes, axis=-2)
    return ProductState(state, wires=Wires.all_wires([prep_wires, zero_wires]), validate_norm=False)


def _broadcast_to_batch(amplitude: TensorLike, batch_size: int) -> TensorLike:
    """
    Add a leading batch axis to unbatched amplitudes.

    :param amplitude: The amplitudes, shape (n, 2) or (batch_size, n, 2).
    :type amplitude: TensorLike
    :param batch_size: The batch size.
    :type batch_size: int
    :return: The batched amplitudes, shape (batch_size, n, 2).
    :rtype: TensorLike
    """
    if qml.math.ndim(amplitude) == 3:
        return amplitude
    return qml.math.stack([amplitude] * batch_size)
