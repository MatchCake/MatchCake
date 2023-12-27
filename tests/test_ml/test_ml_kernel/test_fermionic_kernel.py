import pytest
from msim import ml
import numpy as np
import pennylane as qml
from msim.ml.ml_kernel import FermionicPQCKernel, PennylaneFermionicPQCKernel
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)


@pytest.mark.parametrize("x", [
    np.random.rand(2, 2)
    # for _ in range(N_RANDOM_TESTS_PER_CASE)
])
def test_fermionic_pqck_equal_pennylane(x):
    x = qml.math.array(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel = FermionicPQCKernel()
    pkernel = PennylaneFermionicPQCKernel()
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



