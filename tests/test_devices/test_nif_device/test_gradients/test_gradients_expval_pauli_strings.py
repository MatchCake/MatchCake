import numpy as np
import pennylane as qml
import pytest
import torch
from torch.autograd import gradcheck

from matchcake import BatchHamiltonian
from matchcake.devices.contraction_strategies import contraction_strategy_map
from matchcake.operations import Rxx

from ....configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from ... import devices_init

set_seed(TEST_SEED)

hand_made_test_data = [
    ([0, 0], [qml.PauliZ(0), qml.PauliZ(1)]),
    ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(1) @ qml.PauliZ(2)]),
]

rn_test_data = [
    (
        np.random.choice([0, 1], size=n_wires),
        [p0(w) @ p1(w + np.random.randint(1, n_wires - w)) for w in np.random.randint(n_wires - 1, size=n_wires)],
        contraction_strategy,
    )
    for n_wires in range(2, 6)
    for i in range(N_RANDOM_TESTS_PER_CASE)
    for contraction_strategy in contraction_strategy_map.keys()
    for p0 in [qml.PauliX, qml.PauliY, qml.PauliZ]
    for p1 in [qml.PauliX, qml.PauliY, qml.PauliZ]
]


@pytest.mark.parametrize(
    "basis_state, pauli_string, init_param",
    [
        (basis_state, pauli_string, init_param)
        for basis_state, pauli_string in hand_made_test_data
        for init_param in np.linspace(0, np.pi, num=N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_nif_pauli_strings_grads_with_random_circuit_against_torch_gradcheck_handmade_batch_hamiltonian(
    basis_state, pauli_string, init_param
):
    nif_device, _ = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=None,
        name="lightning.qubit",
    )
    hamiltonian = BatchHamiltonian(np.ones(len(pauli_string)), pauli_string)

    @qml.qnode(nif_device)
    def circuit(params):
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        Rxx(params, wires=[0, 1])
        return qml.expval(hamiltonian)

    init_params_nif = torch.tensor([init_param], dtype=torch.float64).requires_grad_()
    assert gradcheck(
        circuit,
        (init_params_nif,),
        eps=1e-3,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "basis_state, pauli_string, init_param",
    [
        (basis_state, pauli_string, init_param)
        for basis_state, pauli_string in hand_made_test_data
        for init_param in np.linspace(0, np.pi, num=N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_nif_pauli_strings_grads_with_random_circuit_against_torch_gradcheck_handmade_sum_hamiltonian(
    basis_state, pauli_string, init_param
):
    nif_device, _ = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=None,
        name="lightning.qubit",
    )
    hamiltonian = sum(pauli_string)

    @qml.qnode(nif_device)
    def circuit(params):
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        Rxx(params, wires=[0, 1])
        return qml.expval(hamiltonian)

    init_params_nif = torch.tensor([init_param], dtype=torch.float64).requires_grad_()
    assert gradcheck(
        circuit,
        (init_params_nif,),
        eps=1e-3,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
