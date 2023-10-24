import pytest
import numpy as np
from msim import utils

ALL_PAULI = [utils.PAULI_X, utils.PAULI_Y, utils.PAULI_Z, utils.PAULI_I]


@pytest.mark.parametrize(
    "__input,target,lib",
    [
        ([P], P, lib)
        for P in ALL_PAULI
        for lib in [np, "numpy"]
    ] + [
        ([P, Q], np.kron(P, Q), lib)
        for P in ALL_PAULI
        for Q in ALL_PAULI
        for lib in [np, "numpy"]
    ] + [
        ([P, Q, R], np.kron(np.kron(P, Q), R), lib)
        for P in ALL_PAULI
        for Q in ALL_PAULI
        for R in ALL_PAULI
        for lib in [np, "numpy"]
    ]
)
def test_recursive_kron(__input, target, lib):
    out = utils.recursive_kron(__input, lib=lib)
    shapes = [i.shape for i in __input]
    max_dim = max([len(s) for s in shapes])
    shapes = [s + (1,) * (max_dim - len(s)) for s in shapes]
    target_shape = tuple(np.prod(shapes, axis=0).astype(int))
    assert out.shape == target_shape, f"Output shape is {out.shape} instead of {target_shape}"
    assert np.allclose(out, target)




