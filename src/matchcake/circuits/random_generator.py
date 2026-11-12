import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation


def neighbouring_wire_pairs(wires: Union[Sequence[int], np.ndarray]) -> np.ndarray:
    """
    Return the pairs of neighbouring wires ``(w, w + 1)`` whose two wires are both in ``wires``.

    Random circuits place their two-wire gates on these pairs, so every gate is a nearest-neighbour
    gate that stays on the wires of the circuit.

    :param wires: The wires of the circuit.
    :type wires: Union[Sequence[int], np.ndarray]
    :return: The pairs, shape (n_pairs, 2), in increasing order.
    :rtype: np.ndarray
    :raises ValueError: If no two wires of ``wires`` are neighbours.
    """
    sorted_wires = np.unique(np.asarray(wires))
    pairs = np.stack([sorted_wires[:-1], sorted_wires[1:]], axis=-1)
    pairs = pairs[pairs[:, 1] - pairs[:, 0] == 1]
    if len(pairs) == 0:
        raise ValueError(
            f"A random circuit needs at least two neighbouring wires (w and w + 1), got {np.asarray(wires).tolist()}."
        )
    return pairs


class RandomOperationsGenerator:
    """
    Random circuit of two-wire operations, starting in a random basis state.

    Iterating over the generator yields a :class:`~pennylane.BasisState` on all its wires, followed by
    ``n_ops`` operations. Each operation has a type drawn uniformly from ``op_types``, acts on a random
    pair of neighbouring wires ``(w, w + 1)`` of ``wires`` (see :func:`neighbouring_wire_pairs`), and
    has random parameters drawn with the ``random`` class method of its type. The same ``seed`` gives
    the same circuit at every iteration, and the generator can be iterated over several times.

    The generator is an iterable of operations, so it can be given to
    :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.execute_generator`, merged with
    :func:`~matchcake.circuits.merge_generators`, or used as the body of a QNode with :meth:`circuit`.
    The subclasses :class:`~matchcake.circuits.RandomMatchgateOperationsGenerator` and
    :class:`~matchcake.circuits.RandomSptmOperationsGenerator` set default operation types.

    .. code-block:: python

        import pennylane as qml

        from matchcake import NIFDevice
        from matchcake.circuits import RandomMatchgateOperationsGenerator

        generator = RandomMatchgateOperationsGenerator(wires=4, n_ops=10, seed=0, output_type="probs")
        circuit = qml.QNode(generator.circuit, NIFDevice(wires=4))
        probs = circuit()
    """

    def __init__(
        self,
        wires: Union[Sequence[int], int],
        n_ops: Optional[int] = None,
        batch_size: Optional[int] = None,
        op_types: Sequence[Type[Any]] = (),
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
            At least two of them must be neighbours. They are stored sorted.
        :type wires: Union[Sequence[int], int]
        :param n_ops: The number of operations after the state preparation. Defaults to
            ``2 * len(wires) * len(op_types)``.
        :type n_ops: Optional[int]
        :param batch_size: The batch size of the parameters of each operation. Unbatched if None.
        :type batch_size: Optional[int]
        :param op_types: The operation types to draw from. Each must have a
            ``random(wires, batch_size, seed)`` class method.
        :type op_types: Sequence[Type[Any]]
        :param use_cuda: Whether to move each operation to the GPU with its ``to_cuda`` method.
        :type use_cuda: bool
        :param seed: The seed of the random generator, for the initial state, the operation types, their
            wires and their parameters. A new random circuit at every iteration if None.
        :type seed: Optional[int]
        :param output_type: The measurement returned by :meth:`get_output_op`: ``"samples"``,
            ``"probs"``, ``"expval"``, or None for no measurement.
        :type output_type: Optional[str]
        :param observable: The observable of the measurement. Required for ``"expval"``.
        :type observable: Optional[Any]
        :param output_wires: The wires of the ``"samples"`` and ``"probs"`` measurements. Defaults to all
            the wires.
        :type output_wires: Optional[Sequence[int]]
        :param initial_state: The bits of the initial basis state, one per wire in the sorted wire order.
            Drawn at random if None.
        :type initial_state: Optional[Union[Sequence[int], np.ndarray]]
        :param kwargs: Additional keyword arguments, stored in ``self.kwargs``.
        """
        wires_arr: np.ndarray
        if isinstance(wires, int):
            wires_arr = np.arange(wires)
        else:
            wires_arr = np.sort(np.asarray(wires))
        self.wires = wires_arr
        self.n_ops = n_ops if n_ops is not None else 2 * len(wires_arr) * len(op_types)
        self.batch_size = batch_size
        self.op_types = op_types
        self.use_cuda = use_cuda
        self.seed = seed
        self.output_type = output_type
        self.observable = observable
        self.output_wires = output_wires
        self.kwargs = kwargs
        self.initial_state = initial_state

    def __iter__(self) -> Iterator[Operation]:
        """
        Iterate over the operations of the circuit.

        The circuit starts with a :class:`~pennylane.BasisState` on all the wires, followed by ``n_ops``
        operations. Each operation has a type drawn uniformly from ``op_types``, acts on a random pair of
        neighbouring wires ``(w, w + 1)`` of ``wires`` (see :func:`neighbouring_wire_pairs`), and gets its
        own seed drawn from the random generator, so two operations of the same type have different
        parameters. The same ``seed`` gives the same circuit at every iteration.

        :return: A generator of the operations. Empty if ``n_ops`` is 0.
        :rtype: Iterator[Operation]
        :raises ValueError: If no two wires of ``wires`` are neighbours, or if ``initial_state`` does not
            have one bit per wire.
        """
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        initial_state = self.get_initial_state(rn_gen)
        yield qml.BasisState(initial_state, wires=self.wires)

        pairs = neighbouring_wire_pairs(self.wires)
        for _ in range(self.n_ops):
            cls = self.op_types[int(rn_gen.integers(0, len(self.op_types)))]
            pair = pairs[int(rn_gen.integers(0, len(pairs)))]
            op_seed = int(rn_gen.integers(0, 2**32))
            op = cls.random(wires=pair, batch_size=self.batch_size, seed=op_seed)
            yield self._to_device(op)
        return

    def __len__(self) -> int:
        """
        Return the number of operations after the state preparation.

        :return: ``n_ops``. Iterating yields one more operation, the state preparation, when ``n_ops``
            is not 0.
        :rtype: int
        """
        return self.n_ops

    def __repr__(self) -> str:
        """
        Return the class name and the attributes of the generator.

        :return: The representation, with the attributes as sorted JSON.
        :rtype: str
        """
        return f"{self.__class__.__name__}({json.dumps(self.__dict__, default=str, sort_keys=True)})"

    def tolist(self) -> List[Operation]:
        """
        Return the operations of the circuit as a list.

        Inside a QNode, the operations are queued when they are created, so this also applies them to
        the circuit.

        :return: The state preparation followed by the operations.
        :rtype: List[Operation]
        """
        return list(self)

    def get_ops(self) -> List[Operation]:
        """
        Return the operations of the circuit as a list. Same as :meth:`tolist`.

        :return: The state preparation followed by the operations.
        :rtype: List[Operation]
        """
        return self.tolist()

    def get_output_op(self) -> Optional[MeasurementProcess]:
        """
        Return the measurement set by ``output_type``.

        :return: :func:`~pennylane.sample` or :func:`~pennylane.probs` of ``observable`` on
            ``output_wires``, :func:`~pennylane.expval` of ``observable``, or None when ``output_type``
            is None.
        :rtype: Optional[MeasurementProcess]
        :raises ValueError: If ``output_type`` is not supported.
        """
        wires = self.output_wires if self.output_wires is not None else self.wires
        if self.output_type is None:
            return None
        elif self.output_type == "samples":
            return qml.sample(self.observable, wires=wires)
        elif self.output_type == "probs":
            return qml.probs(op=self.observable, wires=wires)
        elif self.output_type == "expval":
            return qml.expval(self.observable)
        else:
            raise ValueError(f"Invalid output_type: {self.output_type}")

    def circuit(self) -> Optional[MeasurementProcess]:
        """
        Apply the operations of the circuit and return its measurement, as the body of a QNode.

        .. code-block:: python

            circuit = qml.QNode(generator.circuit, NIFDevice(wires=generator.n_wires))

        :return: The measurement of :meth:`get_output_op`.
        :rtype: Optional[MeasurementProcess]
        """
        self.tolist()
        return self.get_output_op()

    def get_initial_state(self, rn_gen: np.random.Generator) -> np.ndarray:
        """
        Return the bits of the initial basis state.

        :param rn_gen: The random generator used to draw the bits when ``initial_state`` is None.
        :type rn_gen: np.random.Generator
        :return: One bit per wire, in the sorted wire order.
        :rtype: np.ndarray
        :raises ValueError: If ``initial_state`` does not have one bit per wire.
        """
        if self.initial_state is None:
            initial_state = rn_gen.choice([0, 1], size=self.n_wires).astype(int)
        else:
            initial_state = np.asarray(self.initial_state, dtype=int)
            if len(initial_state) != self.n_wires:
                raise ValueError(f"Initial state has {len(initial_state)} qubits, but {self.n_wires} are required.")
        return initial_state

    def _to_device(self, op: Operation) -> Operation:
        """
        Move an operation to the GPU when ``use_cuda`` is set.

        :param op: The operation.
        :type op: Operation
        :return: The operation, on the GPU when ``use_cuda`` is set.
        :rtype: Operation
        """
        if self.use_cuda:
            return op.to_cuda()
        return op

    @property
    def n_qubits(self) -> int:
        """
        The number of wires of the circuit.

        :return: ``len(wires)``.
        :rtype: int
        """
        return len(self.wires)

    @property
    def n_wires(self) -> int:
        """
        The number of wires of the circuit. Same as :attr:`n_qubits`.

        :return: ``len(wires)``.
        :rtype: int
        """
        return self.n_qubits

    @property
    def output_kwargs(self) -> Dict[str, Any]:
        """
        The measurement settings of the generator.

        :return: ``output_type``, ``observable`` and ``output_wires``, keyed by name.
        :rtype: Dict[str, Any]
        """
        return dict(
            output_type=self.output_type,
            observable=self.observable,
            output_wires=self.output_wires,
        )
