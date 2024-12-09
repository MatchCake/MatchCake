from typing import Optional, Type, List, Sequence, Union, Any

import numpy as np
import pennylane as qml

from .random_generator import RandomOperationsGenerator

from ..operations import (
    MatchgateOperation,
    fRXX,
    fSWAP,
    fRZZ,
    fH,
    fRYY,
    FermionicSuperposition,
)


class RandomMatchgateOperationsGenerator(RandomOperationsGenerator):
    def __init__(
            self,
            wires: Union[Sequence[int], int],
            n_ops: Optional[int] = None,
            batch_size: Optional[int] = None,
            op_types: List[Type[MatchgateOperation]] = (
                MatchgateOperation,
                fRXX,
                fSWAP,
                fRZZ,
                fH,
                fRYY,
                FermionicSuperposition,
            ),
            *,
            use_cuda: bool = False,
            seed: Optional[int] = None,
            output_type: Optional[str] = None,
            observable: Optional[Any] = None,
            output_wires: Optional[Sequence[int]] = None,
            **kwargs
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
            **kwargs
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
            **kwargs
    ):
        super().__init__(
            wires=wires,
            n_ops=n_ops,
            batch_size=batch_size,
            op_types=[],
            use_cuda=use_cuda,
            seed=seed,
            **kwargs
        )
        self.add_swap_noise = add_swap_noise

    def haar_circuit_gen(self):
        yield qml.Identity(wires=[0, 1])
        n_ops = 0
        rn_gen = np.random.default_rng(self.seed)
        while n_ops < self.n_ops:
            i = n_ops % (self.n_qubits - 1)
            yield fRZZ(
                fRZZ.random_params(self.batch_size, seed=self.seed),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield fRYY(
                fRYY.random_params(self.batch_size, seed=self.seed),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield fRZZ(
                fRZZ.random_params(self.batch_size, seed=self.seed),
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
        yield qml.BasisState(rn_gen.choice([0, 1], size=self.n_wires), wires=self.wires)
        yield from self.haar_circuit_gen()
        return



