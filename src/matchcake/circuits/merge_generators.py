from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pennylane as qml
from pennylane.operation import Operation, StatePrepBase
from pennylane.wires import Wires, WiresLike

from ..operations.state_preparation import ProductState
from .merge_blocks import _as_product_state, _merge_state_preps


def merge_generators(
    *generators: Iterable[Operation],
    wires: Optional[WiresLike] = None,
) -> Iterator[Operation]:
    r"""
    Merge generator-based circuits acting on disjoint wires into a single generator of operations.

    This is the counterpart of :func:`~matchcake.circuits.merge_blocks` for circuits given as iterables
    of operations, such as generators, lists or
    :class:`~matchcake.circuits.RandomOperationsGenerator` instances. The result is meant for
    :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.execute_generator` and
    :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.apply_generator`.

    A generator may start with its own state preparations (:class:`~pennylane.BasisState` or
    :class:`~matchcake.operations.state_preparation.ProductState`). Since the generators act on
    disjoint wires, every state preparation commutes with the gates of the other generators, so all of
    them are merged into one :class:`~matchcake.operations.state_preparation.ProductState` yielded
    first, followed by the gates of every generator in order. This is required by the
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`, which accepts a single state
    preparation at the start of the circuit.

    The merge is lazy: only the leading state preparations of each generator are read before the first
    operation is yielded, and the gates are then streamed one by one, so a long circuit is never stored
    in memory. For the same reason, shared wires and state preparations placed after a gate are only
    detected when the offending operation is reached, and raise a ``ValueError`` at that point.

    The merged state preparation covers the wires of the state preparations, the ``wires`` attribute of
    the generators that have one (as :class:`~matchcake.circuits.RandomOperationsGenerator`), and the
    extra ``wires``. Those extra wires start in :math:`|0\rangle`, like every device wire the merged
    state preparation does not cover.

    Operations are created without being queued, so the merged generator can be consumed inside a QNode
    with :func:`~pennylane.apply`.

    .. code-block:: python

        from matchcake import NIFDevice
        from matchcake.circuits import RandomSptmOperationsGenerator, merge_generators

        left = RandomSptmOperationsGenerator(wires=[0, 1, 2], seed=0)
        right = RandomSptmOperationsGenerator(wires=[3, 4, 5], seed=1)
        device = NIFDevice(wires=6)
        probs = device.execute_generator(merge_generators(left, right), output_type="probs", reset=True)

    :param generators: The circuits to merge, each an iterable of operations.
    :type generators: Iterable[Operation]
    :param wires: Extra wires the merged state preparation must cover, starting in :math:`|0\rangle`.
        Only used when a generator has a state preparation.
    :type wires: Optional[WiresLike]
    :return: A generator of the merged operations: the merged state preparation (if any) followed by
        the gates.
    :rtype: Iterator[Operation]
    :raises ValueError: If two generators share a wire, or if a state preparation is not a product state
        or comes after a gate in its generator.
    """
    with qml.QueuingManager.stop_recording():
        iterators = [iter(generator) for generator in generators]
        heads = [_read_leading_state_preps(iterator, index) for index, iterator in enumerate(iterators)]
    wire_owners: Dict[Any, int] = {}
    for index, (state_preps, _) in enumerate(heads):
        for state_prep in state_preps:
            _claim_wires(wire_owners, state_prep.wires, index)
    all_state_preps = [state_prep for state_preps, _ in heads for state_prep in state_preps]
    if all_state_preps:
        declared_wires = [Wires(generator.wires) for generator in generators if hasattr(generator, "wires")]
        extra_wires = Wires([]) if wires is None else Wires(wires)
        prep_wires = Wires.all_wires([state_prep.wires for state_prep in all_state_preps])
        with qml.QueuingManager.stop_recording():
            merged_state_prep = _merge_state_preps(
                all_state_preps, Wires.all_wires([prep_wires, *declared_wires, extra_wires])
            )
        yield merged_state_prep
    for index, (iterator, (_, buffered_ops)) in enumerate(zip(iterators, heads)):
        for op in buffered_ops:
            _claim_wires(wire_owners, op.wires, index)
            yield op
        while True:
            with qml.QueuingManager.stop_recording():
                op = next(iterator, None)
            if op is None:
                break
            if isinstance(op, StatePrepBase):
                raise ValueError(
                    f"The state preparation {op} of generator {index} must come before the gates of its generator."
                )
            _claim_wires(wire_owners, op.wires, index)
            yield op


def _read_leading_state_preps(iterator: Iterator[Operation], index: int) -> Tuple[List[ProductState], List[Operation]]:
    """
    Read the state preparations at the start of a generator.

    Leading :class:`~pennylane.Identity` operations are skipped over and kept, so they do not end the
    state preparations.

    :param iterator: The iterator of the generator. It is advanced up to and including its first gate.
    :type iterator: Iterator[Operation]
    :param index: The position of the generator, used in error messages.
    :type index: int
    :return: The state preparations as product states, and the operations read after them (the leading
        identities and the first gate, if any).
    :rtype: Tuple[List[ProductState], List[Operation]]
    :raises ValueError: If a state preparation is not a product state.
    """
    state_preps: List[ProductState] = []
    buffered_ops: List[Operation] = []
    for op in iterator:
        if isinstance(op, StatePrepBase):
            state_preps.append(_as_product_state(op, f"generator {index}"))
            continue
        buffered_ops.append(op)
        if not isinstance(op, qml.Identity):
            break
    return state_preps, buffered_ops


def _claim_wires(wire_owners: Dict[Any, int], wires: Wires, index: int) -> None:
    """
    Record that a generator acts on some wires.

    :param wire_owners: The generator index that acts on each wire seen so far. Updated in place.
    :type wire_owners: Dict[Any, int]
    :param wires: The wires of the operation.
    :type wires: Wires
    :param index: The position of the generator.
    :type index: int
    :raises ValueError: If one of the wires belongs to another generator.
    """
    for wire in wires:
        owner = wire_owners.setdefault(wire, index)
        if owner != index:
            raise ValueError(
                f"Generators must act on disjoint wires, but generators {owner} and {index} share wire {wire}."
            )
