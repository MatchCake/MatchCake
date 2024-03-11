import pytest
from matchcake import ml
import numpy as np
import pennylane as qml
from matchcake.ml.ml_kernel import FermionicPQCKernel, PennylaneFermionicPQCKernel
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)


@pytest.mark.parametrize(
    "x, rotations",
    [
        (np.random.rand(2, f), rot)
        for rot in ["X", "Y", "Z", "X,Y", "X,Z", "Y,Z", "X,Y,Z"]
        for f in range(2, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_gram_equal_pennylane(x, rotations):
    x = qml.math.array(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel = FermionicPQCKernel(rotations=rotations)
    pkernel = PennylaneFermionicPQCKernel(rotations=rotations)
    fkernel.fit(x, y)
    pkernel.fit(x, y)
    pkernel.parameters = fkernel.parameters
    f_gram = fkernel.compute_gram_matrix(x)
    p_gram = pkernel.compute_gram_matrix(x)
    np.testing.assert_allclose(
        f_gram, p_gram,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
