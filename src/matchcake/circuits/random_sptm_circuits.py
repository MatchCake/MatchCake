from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml

from ..operations import (
    SingleParticleTransitionMatrixOperation,
    SptmCompHH,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
    SptmFermionicSuperposition,
    SptmFSwapCompRzRz,
    SptmIdentity,
)
from .random_generator import RandomOperationsGenerator


# pylint: disable=too-many-arguments
def random_sptm_operations_generator(
    n_ops: int,
    wires: Union[Sequence[int], int],
    batch_size: Optional[int] = None,
    op_types: List[Type[SingleParticleTransitionMatrixOperation]] = (
        SptmCompRxRx,
        SptmCompZX,
        SptmCompRzRz,
        SptmIdentity,
        SptmCompHH,
        SptmCompRyRy,
        SptmFermionicSuperposition,
        SptmFSwapCompRzRz,
    ),
    *,
    use_cuda: bool = False,
    seed: Optional[int] = None,
    **kwargs,
):
    if isinstance(wires, int):
        wires = np.arange(wires)
    wires = np.sort(np.asarray(wires))
    rn_gen = np.random.default_rng(seed)
    for _ in range(n_ops):
        cls = rn_gen.choice(op_types)
        rn_wire0 = rn_gen.choice(wires[:-1])
        rn_wire1 = rn_wire0 + 1
        op = cls.random(wires=[rn_wire0, rn_wire1], batch_size=batch_size, seed=seed)
        if use_cuda:
            op = op.to_cuda()
        yield op
    return


class RandomSptmOperationsGenerator(RandomOperationsGenerator):
    def __init__(
        self,
        wires: Union[Sequence[int], int],
        n_ops: Optional[int] = None,
        batch_size: Optional[int] = None,
        op_types: List[Type[SingleParticleTransitionMatrixOperation]] = (
            SptmCompRxRx,
            SptmCompZX,
            SptmCompRzRz,
            SptmIdentity,
            SptmCompHH,
            SptmCompRyRy,
            SptmFermionicSuperposition,
            SptmFSwapCompRzRz,
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


class RandomSptmHaarOperationsGenerator(RandomSptmOperationsGenerator):
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
        yield SptmIdentity(wires=[0, 1])
        n_ops = 0
        rn_gen = np.random.default_rng(self.seed)
        while n_ops < self.n_ops:
            i = n_ops % (self.n_qubits - 1)
            yield SptmCompRzRz(
                SptmCompRzRz.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield SptmCompRyRy(
                SptmCompRyRy.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield SptmCompRzRz(
                SptmCompRzRz.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1

            if n_ops % self.n_qubits == 0 and self.add_swap_noise:
                wire0 = rn_gen.choice(self.wires[:-1])
                wire1 = wire0 + 1
                # wire1 = rn_gen.choice(self.wires[wire0+1:])
                yield SptmCompZX(wires=[wire0, wire1])
                n_ops += 1
        return

    def __iter__(self):
        if self.n_ops == 0:
            return
        rn_gen = np.random.default_rng(self.seed)
        yield qml.BasisState(self.get_initial_state(rn_gen), wires=self.wires)
        yield from self.haar_circuit_gen()
        return
