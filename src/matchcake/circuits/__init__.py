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
        **kwargs
):
    if isinstance(wires, int):
        wires = np.arange(wires)
    wires = np.sort(np.asarray(wires))
    for _ in range(n_ops):
        cls = np.random.choice(op_types)
        rn_wire0 = np.random.choice(wires[:-1])
        rn_wire1 = rn_wire0 + 1
        op = cls.random(wires=[rn_wire0, rn_wire1], batch_size=batch_size)
        if use_cuda:
            op = op.to_cuda()
        yield op
    return

