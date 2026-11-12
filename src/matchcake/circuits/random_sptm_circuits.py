from collections.abc import Sequence as SequenceABC
from typing import Any, Iterator, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml
from pennylane.operation import Operation

from ..operations import (
    SingleParticleTransitionMatrixOperation,
    SptmCompHH,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
    SptmFermionicSuperposition,
    SptmIdentity,
)
from .random_generator import RandomOperationsGenerator, neighbouring_wire_pairs

DEFAULT_SPTM_OP_TYPES = (
    SptmCompRxRx,
    SptmCompZX,
    SptmCompRzRz,
    SptmIdentity,
    SptmCompHH,
    SptmCompRyRy,
    SptmFermionicSuperposition,
)


# pylint: disable=too-many-arguments
def random_sptm_operations_generator(
    n_ops: int,
    wires: Union[Sequence[int], int],
    batch_size: Optional[int] = None,
    op_types: SequenceABC[Type[SingleParticleTransitionMatrixOperation]] = DEFAULT_SPTM_OP_TYPES,
    *,
    use_cuda: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Iterator[SingleParticleTransitionMatrixOperation]:
    """
    Generate a random circuit of single-particle transition matrix (SPTM) operations.

    Each operation has a type drawn uniformly from ``op_types``, acts on a random pair of neighbouring
    wires ``(w, w + 1)`` of ``wires`` (see :func:`~matchcake.circuits.random_generator.neighbouring_wire_pairs`),
    and has random parameters drawn with the ``random`` class method of its type. No state preparation is yielded: use
    :class:`RandomSptmOperationsGenerator` for a circuit that starts with a random basis state.

    .. code-block:: python

        from matchcake import NIFDevice
        from matchcake.circuits import random_sptm_operations_generator

        device = NIFDevice(wires=4)
        ops = random_sptm_operations_generator(n_ops=10, wires=4, seed=0)
        probs = device.execute_generator(ops, output_type="probs", reset=True)

    :param n_ops: The number of operations to generate.
    :type n_ops: int
    :param wires: The wires of the circuit, or their number ``n`` for the wires ``0, ..., n - 1``. At
        least two of them must be neighbours.
    :type wires: Union[Sequence[int], int]
    :param batch_size: The batch size of the parameters of each operation. Unbatched if None.
    :type batch_size: Optional[int]
    :param op_types: The operation types to draw from.
    :type op_types: Sequence[Type[SingleParticleTransitionMatrixOperation]]
    :param use_cuda: Whether to move each operation to the GPU.
    :type use_cuda: bool
    :param seed: The seed of the random generator. The same seed gives the same circuit. Each
        operation gets its own seed drawn from this generator, so two operations of the same type
        have different parameters.
    :type seed: Optional[int]
    :param kwargs: Ignored. Kept for compatibility with the other generators.
    :return: A generator of the ``n_ops`` random operations.
    :rtype: Iterator[SingleParticleTransitionMatrixOperation]
    :raises ValueError: If no two wires of ``wires`` are neighbours.
    """
    pairs = neighbouring_wire_pairs(np.arange(wires) if isinstance(wires, int) else wires)
    rn_gen = np.random.default_rng(seed)
    for _ in range(n_ops):
        cls = op_types[int(rn_gen.integers(0, len(op_types)))]
        pair = pairs[int(rn_gen.integers(0, len(pairs)))]
        op_seed = int(rn_gen.integers(0, 2**32))
        op = cls.random(wires=pair, batch_size=batch_size, seed=op_seed)
        if use_cuda:
            op = op.to_cuda()
        yield op
    return


class RandomSptmOperationsGenerator(RandomOperationsGenerator):
    """
    Random circuit of single-particle transition matrix (SPTM) operations, starting in a random basis
    state.

    Iterating over the generator yields a :class:`~pennylane.BasisState` on all its wires, followed by ``n_ops``
    operations. Each operation has a type drawn uniformly from ``op_types``, acts on a random pair of neighbouring
    wires ``(w, w + 1)`` of ``wires``, and has random parameters. The same ``seed`` gives the same circuit at every
    iteration.

    The SPTM operations are only supported by the
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`. Use
    :class:`~matchcake.circuits.RandomMatchgateOperationsGenerator` for a circuit that also runs on qubit
    devices.

    .. code-block:: python

        from matchcake import NIFDevice
        from matchcake.circuits import RandomSptmOperationsGenerator

        generator = RandomSptmOperationsGenerator(wires=4, n_ops=10, seed=0)
        device = NIFDevice(wires=4)
        probs = device.execute_generator(generator, output_type="probs", reset=True)
    """

    def __init__(
        self,
        wires: Union[Sequence[int], int],
        n_ops: Optional[int] = None,
        batch_size: Optional[int] = None,
        op_types: SequenceABC[Type[SingleParticleTransitionMatrixOperation]] = DEFAULT_SPTM_OP_TYPES,
        *,
        use_cuda: bool = False,
        seed: Optional[int] = None,
        output_type: Optional[str] = None,
        observable: Optional[Any] = None,
        output_wires: Optional[Sequence[int]] = None,
        initial_state: Optional[Union[Sequence[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Configure the random circuit.

        :param wires: The wires of the circuit, or their number ``n`` for the wires ``0, ..., n - 1``.
            At least two of them must be neighbours.
        :type wires: Union[Sequence[int], int]
        :param n_ops: The number of operations after the state preparation. Defaults to
            ``2 * len(wires) * len(op_types)``.
        :type n_ops: Optional[int]
        :param batch_size: The batch size of the parameters of each operation. Unbatched if None.
        :type batch_size: Optional[int]
        :param op_types: The operation types to draw from.
        :type op_types: Sequence[Type[SingleParticleTransitionMatrixOperation]]
        :param use_cuda: Whether to move each operation to the GPU.
        :type use_cuda: bool
        :param seed: The seed of the random generator, for the initial state, the operation types, their
            wires and their parameters.
        :type seed: Optional[int]
        :param output_type: The measurement returned by :meth:`get_output_op`: ``"samples"``,
            ``"probs"``, ``"expval"``, or None for no measurement.
        :type output_type: Optional[str]
        :param observable: The observable of the measurement.
        :type observable: Optional[Any]
        :param output_wires: The wires of the measurement. Defaults to all the wires.
        :type output_wires: Optional[Sequence[int]]
        :param initial_state: The bits of the initial basis state, one per wire. Drawn at random if None.
        :type initial_state: Optional[Union[Sequence[int], np.ndarray]]
        :param kwargs: Additional keyword arguments, stored in ``self.kwargs``.
        """
        super().__init__(
            wires=wires,
            n_ops=n_ops,
            batch_size=batch_size,
            op_types=op_types,
            use_cuda=use_cuda,
            seed=seed,
            output_type=output_type,
            observable=observable,
            output_wires=output_wires,
            initial_state=initial_state,
            **kwargs,
        )


class RandomSptmHaarOperationsGenerator(RandomSptmOperationsGenerator):
    r"""
    Brickwork circuit of random rotations approximating a Haar-random free-fermion evolution.

    Iterating over the generator yields a :class:`~pennylane.BasisState` on all its wires, an
    :class:`~matchcake.operations.SptmIdentity` on the first pair of neighbouring wires, and then ``n_ops``
    operations. These operations are blocks of three rotations :math:`R_{ZZ} R_{YY} R_{ZZ}` with random angles
    (:class:`~matchcake.operations.SptmCompRzRz`, :class:`~matchcake.operations.SptmCompRyRy`,
    :class:`~matchcake.operations.SptmCompRzRz`), placed on the pairs of neighbouring wires ``(w, w + 1)`` of
    ``wires`` in turn, in increasing order and starting over after the last pair. With ``add_swap_noise``, a
    fermionic swap (:class:`~matchcake.operations.SptmCompZX`) on a random pair of neighbouring wires is added each
    time the number of operations reaches a multiple of the number of wires. The circuit stops after exactly
    ``n_ops`` operations, even in the middle of a block. The same ``seed`` gives the same circuit at every
    iteration.

    .. code-block:: python

        from matchcake import NIFDevice
        from matchcake.circuits import RandomSptmHaarOperationsGenerator

        generator = RandomSptmHaarOperationsGenerator(wires=4, seed=0)
        device = NIFDevice(wires=4)
        probs = device.execute_generator(generator, output_type="probs", reset=True)
    """

    def __init__(
        self,
        wires: Union[Sequence[int], int],
        n_ops: Optional[int] = None,
        batch_size: Optional[int] = None,
        *,
        use_cuda: bool = False,
        seed: Optional[int] = None,
        add_swap_noise: bool = True,
        initial_state: Optional[Union[Sequence[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Configure the brickwork circuit.

        :param wires: The wires of the circuit, or their number ``n`` for the wires ``0, ..., n - 1``.
            At least two of them must be neighbours.
        :type wires: Union[Sequence[int], int]
        :param n_ops: The number of operations after the state preparation and the identity. Defaults
            to ``3 * n * (n - 1)`` for ``n`` wires, which is ``n`` sweeps of rotation blocks over the
            ``n - 1`` pairs of neighbouring wires of ``0, ..., n - 1``.
        :type n_ops: Optional[int]
        :param batch_size: The batch size of the rotation angles. Unbatched if None.
        :type batch_size: Optional[int]
        :param use_cuda: Whether to move each operation to the GPU with its ``to_cuda`` method.
        :type use_cuda: bool
        :param seed: The seed of the random generator, for the initial state, the rotation angles and the
            wires of the swaps.
        :type seed: Optional[int]
        :param add_swap_noise: Whether to add the random fermionic swaps.
        :type add_swap_noise: bool
        :param initial_state: The bits of the initial basis state, one per wire. Drawn at random if None.
        :type initial_state: Optional[Union[Sequence[int], np.ndarray]]
        :param kwargs: Additional keyword arguments passed to :class:`RandomSptmOperationsGenerator`.
        """
        super().__init__(
            wires=wires,
            n_ops=n_ops,
            batch_size=batch_size,
            op_types=[],
            use_cuda=use_cuda,
            seed=seed,
            initial_state=initial_state,
            **kwargs,
        )
        if n_ops is None:
            self.n_ops = 3 * self.n_qubits * (self.n_qubits - 1)
        self.add_swap_noise = add_swap_noise

    def __iter__(self) -> Iterator[Operation]:
        """
        Iterate over the operations of the circuit.

        :return: A generator of the state preparation followed by the operations of
            :meth:`haar_circuit_gen`. Empty if ``n_ops`` is 0.
        :rtype: Iterator[Operation]
        """
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        yield qml.BasisState(self.get_initial_state(rn_gen), wires=self.wires)
        yield from self.haar_circuit_gen(rn_gen)
        return

    def haar_circuit_gen(self, rn_gen: Optional[np.random.Generator] = None) -> Iterator[Operation]:
        """
        Generate the identity and the ``n_ops`` operations of the brickwork circuit, without the state
        preparation.

        :param rn_gen: The random generator of the angles and the wires of the swaps. Defaults to a new
            generator seeded with ``seed``.
        :type rn_gen: Optional[np.random.Generator]
        :return: A generator of the operations.
        :rtype: Iterator[Operation]
        :raises ValueError: If no two wires of ``wires`` are neighbours.
        """
        if rn_gen is None:
            rn_gen = np.random.default_rng(self.seed)
        pairs = neighbouring_wire_pairs(self.wires)
        yield self._to_device(SptmIdentity(wires=pairs[0]))
        n_ops = 0
        n_blocks = 0
        while n_ops < self.n_ops:
            pair = pairs[n_blocks % len(pairs)]
            n_blocks += 1
            for cls in (SptmCompRzRz, SptmCompRyRy, SptmCompRzRz):
                if n_ops >= self.n_ops:
                    return
                params = cls.random_params(self.batch_size, seed=int(rn_gen.integers(0, 2**32)))
                yield self._to_device(cls(params, wires=pair))
                n_ops += 1

            if n_ops < self.n_ops and n_ops % self.n_qubits == 0 and self.add_swap_noise:
                yield self._to_device(SptmCompZX(wires=pairs[int(rn_gen.integers(0, len(pairs)))]))
                n_ops += 1
        return
