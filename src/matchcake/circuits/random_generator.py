from typing import Optional, Type, List, Sequence, Union, Any

import numpy as np
import pennylane as qml


class RandomOperationsGenerator:
    def __init__(
            self,
            wires: Union[Sequence[int], int],
            n_ops: Optional[int] = None,
            batch_size: Optional[int] = None,
            op_types: List[Type[Any]] = (),
            *,
            use_cuda: bool = False,
            seed: Optional[int] = None,
            output_type: Optional[str] = None,
            observable: Optional[Any] = None,
            output_wires: Optional[Sequence[int]] = None,
            **kwargs
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

    @property
    def n_qubits(self):
        return len(self.wires)

    @property
    def n_wires(self):
        return self.n_qubits

    @property
    def output_kwargs(self):
        return dict(output_type=self.output_type, observable=self.observable, output_wires=self.output_wires)

    def __iter__(self):
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        initial_state = rn_gen.choice([0, 1], size=self.n_wires)
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
