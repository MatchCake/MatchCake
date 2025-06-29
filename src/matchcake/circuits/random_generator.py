import json
from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml


class RandomOperationsGenerator:
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
        **kwargs,
    ):
        if isinstance(wires, int):
            wires = np.arange(wires)
        wires = np.sort(np.asarray(wires))
        self.wires = wires
        self.n_ops = n_ops if n_ops is not None else 2 * len(wires) * len(op_types)
        self.batch_size = batch_size
        self.op_types = op_types
        self.use_cuda = use_cuda
        self.seed = seed
        self.output_type = output_type
        self.observable = observable
        self.output_wires = output_wires
        self.kwargs = kwargs
        self.initial_state = initial_state

    @property
    def n_qubits(self):
        return len(self.wires)

    @property
    def n_wires(self):
        return self.n_qubits

    @property
    def output_kwargs(self):
        return dict(
            output_type=self.output_type,
            observable=self.observable,
            output_wires=self.output_wires,
        )

    def __iter__(self):
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        initial_state = self.get_initial_state(rn_gen)
        yield qml.BasisState(initial_state, wires=self.wires)

        wires = np.sort(np.asarray(self.wires))
        for _ in range(self.n_ops):
            cls = rn_gen.choice(self.op_types)
            rn_wire0 = rn_gen.choice(wires[:-1])
            rn_wire1 = rn_wire0 + 1
            op = cls.random(wires=[rn_wire0, rn_wire1], batch_size=self.batch_size, seed=self.seed)
            if self.use_cuda:
                op = op.to_cuda()
            yield op
        return

    def __len__(self):
        return self.n_ops

    def tolist(self):
        return list(self)

    def get_ops(self):
        return self.tolist()

    def get_output_op(self):
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

    def circuit(self):
        self.tolist()
        return self.get_output_op()

    def get_initial_state(self, rn_gen: np.random.Generator):
        if self.initial_state is None:
            initial_state = rn_gen.choice([0, 1], size=self.n_wires).astype(int)
        else:
            initial_state = np.asarray(self.initial_state, dtype=int)
            if len(initial_state) != self.n_wires:
                raise ValueError(f"Initial state has {len(initial_state)} qubits, but {self.n_wires} are required.")
        return initial_state

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.__dict__, default=str, sort_keys=True)})"
