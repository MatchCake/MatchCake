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


@pytest.mark.parametrize(
    "n_qubit, n_features, entangling_mth",
    [
        (n_q, n_feat, ent_mth)
        for n_q in np.arange(2, 64, 2)
        for n_feat in np.arange(n_q, 64, 2)
        for ent_mth in ["identity", "fswap", "hadamard"]
    ]
)
def test_fermionic_pqc_n_gates(n_qubit, n_features, entangling_mth):
    fkernel = FermionicPQCKernel(size=n_qubit, entangling_mth=entangling_mth, parameter_scaling=1, data_scaling=1)

    x = np.stack([np.arange(n_features) for _ in range(2)])
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    fkernel._depth = int(max(1, np.ceil(n_features / n_qubit)))
    fkernel.parameters = np.zeros(n_features)
    fkernel.single_distance(x[0], x[-1])
    qscript = fkernel.qnode.tape.expand()
    n_gates = len(qscript.operations) // 2  # remove the adjoint gates
    gates = [op.name for op in qscript.operations[:n_gates]]
    rotations = fkernel.rotations.split(',')
    for k in rotations:
        n_k = len([g for g in gates if k in g])
        assert n_k == n_features // 2, (
            f"We expect {n_features // 2} gates of type {k} but got {n_k} "
            f"with n_qubit={n_qubit}, n_features={n_features}, entangling_mth={entangling_mth}"
        )
    is_entangling = entangling_mth != "identity"
    half_depth, half_even_qubit = fkernel.depth // 2, n_qubit // 2
    n_expected_entangling_gates = (half_even_qubit * fkernel.depth - half_depth) * int(is_entangling)
    n_expected_gates = (n_features//2) * len(rotations) + n_expected_entangling_gates
    assert n_gates == n_expected_gates, \
        (f"n_gates={n_gates}, n_expected_gates={n_expected_gates} "
         f"with n_qubit={n_qubit}, n_features={n_features}, entangling_mth={entangling_mth}")
