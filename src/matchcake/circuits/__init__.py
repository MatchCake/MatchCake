from typing import Optional, Type, List, Sequence, Union

import numpy as np

from ..operations import (
    SingleParticleTransitionMatrixOperation,
    SptmRxRx,
    SptmFSwap,
    SptmRzRz,
    SptmIdentity,
    SptmFHH,
    SptmRyRy,
    SptmFermionicSuperposition,
    SptmFSwapRzRz,
)


# pylint: disable=too-many-arguments
def random_sptm_operations_generator(
        n_ops: int,
        wires: Union[Sequence[int], int],
        batch_size: Optional[int] = None,
        op_types: List[Type[SingleParticleTransitionMatrixOperation]] = (
            SptmRxRx,
            SptmFSwap,
            SptmRzRz,
            SptmIdentity,
            SptmFHH,
            SptmRyRy,
            SptmFermionicSuperposition,
            SptmFSwapRzRz,
        ),
        *,
        use_cuda: bool = False,
        seed: Optional[int] = None,
        **kwargs
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


class RandomSptmOperationsGenerator:
    def __init__(
            self,
            wires: Union[Sequence[int], int],
            n_ops: Optional[int] = None,
            batch_size: Optional[int] = None,
            op_types: List[Type[SingleParticleTransitionMatrixOperation]] = (
                SptmRxRx,
                SptmFSwap,
                SptmRzRz,
                SptmIdentity,
                SptmFHH,
                SptmRyRy,
                SptmFermionicSuperposition,
                SptmFSwapRzRz,
            ),
            *,
            use_cuda: bool = False,
            seed: Optional[int] = None,
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
        self.kwargs = kwargs

    @property
    def n_qubits(self):
        return len(self.wires)

    @property
    def n_wires(self):
        return self.n_qubits

    def __iter__(self):
        return random_sptm_operations_generator(
            self.n_ops,
            self.wires,
            batch_size=self.batch_size,
            op_types=self.op_types,
            use_cuda=self.use_cuda,
            seed=self.seed,
            **self.kwargs
        )

    def __len__(self):
        return self.n_ops


