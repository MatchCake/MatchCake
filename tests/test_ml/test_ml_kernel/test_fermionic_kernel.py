import pytest
from typing import Literal, List, Union, Dict
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from matchcake.ml.kernels import FermionicPQCKernel, StateVectorFermionicPQCKernel
from matchcake.utils.torch_utils import to_numpy
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "x, rotations",
    [
        (np.random.rand(2, f), rot)
        for rot in ["X", "Y", "X,Z", "Y,Z", "X,Y,Z"]
        for f in range(2, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_gram_equal_pennylane(x, rotations):
    x = qml.math.array(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel = FermionicPQCKernel(rotations=rotations, parameter_scaling=0, data_scaling=1)
    pkernel = StateVectorFermionicPQCKernel(rotations=rotations, parameter_scaling=0, data_scaling=1)
    fkernel.fit(x, y)
    pkernel.fit(x, y)
    pkernel.parameters = to_numpy(fkernel.parameters)
    f_gram = fkernel.compute_gram_matrix(x)
    p_gram = pkernel.compute_gram_matrix(x)
    np.testing.assert_allclose(
        f_gram, p_gram,
        atol=10*ATOL_APPROX_COMPARISON,
        rtol=10*RTOL_APPROX_COMPARISON,
        err_msg=f"Gram matrices are not equal for rotations={rotations}",
    )


@pytest.mark.parametrize(
    "n_qubit, n_features, entangling_mth",
    [
        (n_q, n_feat, ent_mth)
        for n_q in np.arange(2, 16, 2)
        for n_feat in np.arange(n_q, 16, 2)
        for ent_mth in ["identity", "fswap", "hadamard"]
    ]
)
def test_fermionic_pqc_n_gates(
        n_qubit: int,
        n_features: int,
        entangling_mth: Literal["identity", "fswap", "hadamard"]
):
    """
    Test that the number of gates is as expected for the FermionicPQCKernel.

    :param n_qubit: Number of qubits in the quantum circuit
    :type n_qubit: int
    :param n_features: Number of features in the data
    :type n_features: int
    :param entangling_mth: Entangling method
    :type entangling_mth: Literal["identity", "fswap", "hadamard"]
    :return: None

    :raises AssertionError: If the number of rotations is not as expected
    :raises AssertionError: If the number of gates is not as expected
    """
    fkernel = FermionicPQCKernel(size=n_qubit, entangling_mth=entangling_mth, parameter_scaling=1, data_scaling=1)

    x = np.stack([np.arange(n_features) for _ in range(2)])
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
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


@pytest.mark.parametrize(
    "n_qubit, n_features, entangling_mth, rotations, expected_arrangement",
    [
        (2, 2, "identity", ["X"], [dict(gate_subname="X", wires=[0, 1], parameters=[0, 1])]),
        (2, 2, "identity", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
        ]),
        (2, 4, "identity", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[2, 3]),
        ]),
        (4, 6, "identity", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[4, 5]),
        ]),
        (4, 6, "fswap", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="fswap", wires=[0, 1], parameters=[]),
            dict(gate_subname="fswap", wires=[2, 3], parameters=[]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="fswap", wires=[1, 2], parameters=[]),
        ]),
        (4, 8, "fswap", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="fswap", wires=[0, 1], parameters=[]),
            dict(gate_subname="fswap", wires=[2, 3], parameters=[]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[6, 7]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[6, 7]),
            dict(gate_subname="fswap", wires=[1, 2], parameters=[]),
        ]),
        (4, 6, "hadamard", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="h", wires=[0, 1], parameters=[]),
            dict(gate_subname="h", wires=[2, 3], parameters=[]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="h", wires=[1, 2], parameters=[]),
        ]),
        (4, 8, "hadamard", ["Y,Z"], [
            dict(gate_subname="Y", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[0, 1]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[2, 3]),
            dict(gate_subname="h", wires=[0, 1], parameters=[]),
            dict(gate_subname="h", wires=[2, 3], parameters=[]),
            dict(gate_subname="Y", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Z", wires=[0, 1], parameters=[4, 5]),
            dict(gate_subname="Y", wires=[2, 3], parameters=[6, 7]),
            dict(gate_subname="Z", wires=[2, 3], parameters=[6, 7]),
            dict(gate_subname="h", wires=[1, 2], parameters=[]),
        ]),
    ]
)
def test_fermionic_pqc_arrangement_of_gates(
        n_qubit: int,
        n_features: int,
        entangling_mth: Literal["identity", "fswap", "hadamard"],
        rotations: List[Literal["X", "Y", "Z"]],
        expected_arrangement: List[
            Dict[Literal["gate_subname", "wires", "parameters"], Union[str, List[int], List[float]]]
        ]
):
    """
    Test that the arrangement of the gates is as expected for the FermionicPQCKernel. Note that the
    parameters of this test function are about the non-adjoint gates only (i.e., the first half of the gates).

    :param n_qubit: Number of qubits in the quantum circuit
    :type n_qubit: int
    :param n_features: Number of features in the data
    :type n_features: int
    :param entangling_mth: Entangling method
    :type entangling_mth: Literal["identity", "fswap", "hadamard"]
    :param rotations: List of rotations
    :type rotations: List[Literal["X", "Y", "Z"]]
    :param expected_arrangement: Expected arrangement of the gates
    :type expected_arrangement: List[Dict[Literal["gate_subname", "wires", "parameters"], Union[str, List[int], List[float]]]]
    :return: None

    :raises AssertionError: If the arrangement of the gates is not as expected
    """
    rotations = ','.join(rotations)
    fkernel = FermionicPQCKernel(
        size=n_qubit, entangling_mth=entangling_mth, rotations=rotations, parameter_scaling=1, data_scaling=1
    )

    x = np.stack([np.arange(n_features) for _ in range(2)])
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    fkernel.parameters = np.zeros(n_features)
    fkernel.single_distance(x[0], x[-1])
    qscript = fkernel.qnode.tape.expand()
    n_gates = len(qscript.operations) // 2  # remove the adjoint gates
    gates = [op for op in qscript.operations[:n_gates]]
    assert len(gates) == len(expected_arrangement), (
        f"Expected {len(expected_arrangement)} gates but got {len(gates)} "
    )
    for gate, expected_gate in zip(gates, expected_arrangement):
        assert expected_gate["gate_subname"].lower() in gate.name.lower(), (
            f"Expected gate {expected_gate['gate_subname']} but got {gate.name} "
        )
        assert Wires(expected_gate["wires"]) == gate.wires, (
            f"Expected wires {expected_gate['wires']} but got {gate} "
        )
        if hasattr(gate, "_given_params"):
            np.testing.assert_allclose(
                expected_gate["parameters"], gate._given_params,
                atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )


@pytest.mark.parametrize(
    "n_qubit, x, entangling_mth, rotations",
    [
        (n_q, np.random.rand(np.random.randint(n_q, 3*n_q+1)), ent_mth, rot)
        for n_q in [2, 6]
        for ent_mth in ["identity", "fswap", "hadamard"]
        for rot in [["X"], ["Y"], ["Z"], ["X", "Y"], ["X", "Z"], ["Y", "Z"], ["X", "Y", "Z"]]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_identity_test(
        n_qubit: int,
        x: np.ndarray,
        entangling_mth: Literal["identity", "fswap", "hadamard"],
        rotations: List[Literal["X", "Y", "Z"]],
):
    """
    Test that the FermionicPQCKernel always return an 1.0 when the inputs are the same.

    :param n_qubit: The number of qubits
    :type n_qubit: int
    :param x: The input data
    :type x: np.ndarray
    :param entangling_mth: The entangling method
    :type entangling_mth: Literal["identity", "fswap", "hadamard"]
    :param rotations: The rotations
    :type rotations: List[Literal["X", "Y", "Z"]]
    :return: None
    """
    rotations = ','.join(rotations)
    fkernel = FermionicPQCKernel(
        size=n_qubit,
        entangling_mth=entangling_mth,
        rotations=rotations,
        device_kwargs=dict(contraction_method=None),
    )
    x = np.stack([x, x], axis=0)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    np.testing.assert_allclose(
        fkernel.single_distance(x[0], x[-1]), 1.0,
        atol=2*ATOL_APPROX_COMPARISON, rtol=2*RTOL_APPROX_COMPARISON,
        err_msg=f"single_distance failed for identity test with "
                f"n_qubit={n_qubit}, entangling_mth={entangling_mth}, rotations={rotations}"
    )
    np.testing.assert_allclose(
        fkernel.compute_gram_matrix(x), np.ones((2, 2)),
        atol=2*ATOL_APPROX_COMPARISON, rtol=2*RTOL_APPROX_COMPARISON,
        err_msg=f"compute_gram_matrix failed for identity test with "
                f"n_qubit={n_qubit}, entangling_mth={entangling_mth}, rotations={rotations}"
    )


@pytest.mark.parametrize(
    "n_qubit, x0, x1, entangling_mth, rotations",
    [
        (
                n_q,
                np.random.rand(rn_size),
                np.random.rand(rn_size),
                ent_mth, rot,
        )
        for n_q in [2, 6]
        for ent_mth in ["identity", "fswap", "hadamard"]
        for rot in [["X"], ["Y"], ["Z"], ["X", "Y"], ["X", "Z"], ["Y", "Z"], ["X", "Y", "Z"]]
        for rn_size in np.random.randint(n_q, 2*n_q+1, size=N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_swap_test(
        n_qubit: int,
        x0: np.ndarray,
        x1: np.ndarray,
        entangling_mth: Literal["identity", "fswap", "hadamard"],
        rotations: List[Literal["X", "Y", "Z"]],
):
    """
    Test that the FermionicPQCKernel always return the same value when the inputs are swapped.

    :param n_qubit: The number of qubits
    :type n_qubit: int
    :param x0: The first input data
    :type x0: np.ndarray
    :param x1: The second input data
    :type x1: np.ndarray
    :param entangling_mth: The entangling method
    :type entangling_mth: Literal["identity", "fswap", "hadamard"]
    :param rotations: The rotations
    :type rotations: List[Literal["X", "Y", "Z"]]
    :return: None
    """
    rotations = ','.join(rotations)
    fkernel = FermionicPQCKernel(
        size=n_qubit, entangling_mth=entangling_mth, rotations=rotations,
        device_kwargs=dict(contraction_method=None),
    )
    x = np.stack([x0, x1], axis=0)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    # np.testing.assert_allclose(
    #     np.abs(fkernel.single_distance(x0, x1) - fkernel.single_distance(x1, x0)), 0.0,
    #     atol=2*ATOL_APPROX_COMPARISON, rtol=2*RTOL_APPROX_COMPARISON
    # )
    gram = fkernel.compute_gram_matrix(x)
    np.testing.assert_allclose(
        gram - gram.T, np.zeros((2, 2)),
        atol=2*ATOL_APPROX_COMPARISON, rtol=2*RTOL_APPROX_COMPARISON
    )


@pytest.mark.parametrize(
    "x",
    [
        np.stack([np.random.rand(2), np.random.rand(2)], axis=0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_single_distance_gradient(x):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    fkernel = FermionicPQCKernel(
        size=2, device_kwargs=dict(contraction_method=None),
        qnode_kwargs=dict(interface="torch", diff_method="backprop")
    )
    x = torch.from_numpy(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    expval = fkernel.single_distance(x[0], x[-1])
    assert expval.grad_fn is not None, "The gradient is not computed correctly."
    expval.backward()
    assert fkernel.parameters.grad is not None, "The gradient is not computed correctly."


@pytest.mark.parametrize(
    "x, contraction_method",
    [
        [
            np.stack([np.random.rand(n_qubits), np.random.rand(n_qubits)], axis=0),
            contraction_method
        ]
        for n_qubits in [2, 4, 6]
        for contraction_method in [
            None,
            "vertical",
            "horizontal",
            "neighbours",
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_compute_gram_matrix_gradient(x, contraction_method):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    fkernel = FermionicPQCKernel(
        size=x.shape[-1], device_kwargs=dict(contraction_method=contraction_method),
        qnode_kwargs=dict(interface="torch", diff_method="backprop"),
    )
    x = torch.from_numpy(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    expvals = fkernel.compute_gram_matrix(x)
    assert expvals.grad_fn is not None, "The gradient is not computed correctly."
    expvals.sum().backward()
    assert torch.all(torch.isfinite(fkernel.parameters.grad)), "The gradient is not computed correctly."
    assert fkernel.parameters.grad is not None, "The gradient is not computed correctly."


@pytest.mark.parametrize(
    "x",
    [
        np.stack([np.random.rand(2), np.random.rand(2)], axis=0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_fermionic_pqc_compute_gram_matrix_gradient_against_state_vec_sim(x):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")

    fkernel = FermionicPQCKernel(
        size=2, device_kwargs=dict(contraction_method=None),
        qnode_kwargs=dict(interface="torch", diff_method="backprop")
    )
    x = torch.from_numpy(x)
    y = qml.math.array(np.zeros(x.shape[0]))
    fkernel.fit(x, y)
    expval = fkernel.single_distance(x[0], x[-1])
    expval.backward()
    grad = fkernel.parameters.grad

    pkernel = StateVectorFermionicPQCKernel(
        size=2,
        qnode_kwargs=dict(interface="torch", diff_method="backprop")
    )
    pkernel.fit(x, y)
    pkernel.parameters = fkernel.parameters.detach().clone().requires_grad_(True)
    p_expval = pkernel.single_distance(x[0], x[-1])
    p_expval.backward()
    p_grad = pkernel.parameters.grad
    np.testing.assert_allclose(grad, p_grad, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)
