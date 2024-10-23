from typing import Optional, Type, List, Sequence

import numpy as np

from ..operations import (
    SingleParticleTransitionMatrixOperation,
    SptmRxRx,
    SptmFSwap,
    SptmRzRz,
    SptmIdentity,
    SptmFHH,
    SptmRyRy,
)


def random_sptm_operations_generator(
        n_ops: int,
        wires: Sequence[int],
        batch_size: Optional[int] = None,
        op_types: List[Type[SingleParticleTransitionMatrixOperation]] = (
            SptmRxRx,
            SptmFSwap,
            SptmRzRz,
            SptmIdentity,
            SptmFHH,
            SptmRyRy,
        )
):
    for _ in range(n_ops):
        cls = np.random.choice(op_types)
        rn_wire0 = np.random.choice(wires[:-1])
        rn_wire1 = rn_wire0 + 1
        yield cls.random(wires=[rn_wire0, rn_wire1], batch_size=batch_size)
    return

