from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml

from ..operations import (
    CompHH,
    CompRxRx,
    CompRyRy,
    CompRzRz,
    FermionicSuperposition,
    MatchgateOperation,
    fSWAP,
)
from .random_generator import RandomOperationsGenerator


class RandomMatchgateOperationsGenerator(RandomOperationsGenerator):
    def __init__(
        self,
        wires: Union[Sequence[int], int],
        n_ops: Optional[int] = None,
        batch_size: Optional[int] = None,
        op_types: List[Type[MatchgateOperation]] = (
            MatchgateOperation,
            CompRxRx,
            fSWAP,
            CompRzRz,
            CompHH,
            CompRyRy,
            FermionicSuperposition,
        ),
        *,
        use_cuda: bool = False,
        seed: Optional[int] = None,
        output_type: Optional[str] = None,
        observable: Optional[Any] = None,
        output_wires: Optional[Sequence[int]] = None,
        initial_state: Optional[Union[Sequence[int], np.ndarray]] = None,
        **kwargs,
    ):
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


class RandomMatchgateHaarOperationsGenerator(RandomMatchgateOperationsGenerator):
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
        **kwargs,
    ):
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
        self.add_swap_noise = add_swap_noise

    def haar_circuit_gen(self):
        yield qml.Identity(wires=[0, 1])
        n_ops = 0
        rn_gen = np.random.default_rng(self.seed)
        while n_ops < self.n_ops:
            i = n_ops % (self.n_qubits - 1)
            yield CompRzRz(
                CompRzRz.random_params(self.batch_size, seed=self.seed),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield CompRyRy(
                CompRyRy.random_params(self.batch_size, seed=self.seed),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield CompRzRz(
                CompRzRz.random_params(self.batch_size, seed=self.seed),
                wires=[i, i + 1],
            )
            n_ops += 1

            if n_ops % self.n_qubits == 0 and self.add_swap_noise:
                wire0 = rn_gen.choice(self.wires[:-1])
                wire1 = wire0 + 1
                yield fSWAP(wires=[wire0, wire1])
                n_ops += 1
        return

    def __iter__(self):
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        initial_state = self.get_initial_state(rn_gen)
        yield qml.BasisState(initial_state, wires=self.wires)
        yield from self.haar_circuit_gen()
        return
