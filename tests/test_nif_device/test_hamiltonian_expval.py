import numpy as np
import pennylane as qml
import pytest
import psutil

from pennylane import PauliZ as Z

from matchcake import MatchgateOperation, utils, BatchHamiltonian
from matchcake.devices.contraction_strategies import contraction_strategy_map
from matchcake.operations import SptmfRxRx
from matchcake.utils import torch_utils
from matchcake import matchgate_parameter_sets as mps
from matchcake.circuits import random_sptm_operations_generator, RandomMatchgateHaarOperationsGenerator, \
    RandomMatchgateOperationsGenerator
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "basis_state,hamiltonian",
    [
        ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ]
)
def test_nif_sum_hamiltonian_expval_zz_on_basis_state_against_qubit_device(basis_state, hamiltonian):
    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(hamiltonian))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    energy = sum(
        nif_device.execute_generator(circuit_gen(), observable=obs, output_type="expval")
        for obs in hamiltonian
    )

    q_node = qml.QNode(circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "basis_state,hamiltonian",
    [
        ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ]
)
def test_nif_pennylane_hamiltonian_expval_zz_on_basis_state_against_qubit_device(basis_state, hamiltonian):
    pennylane_hamiltonian = qml.Hamiltonian(coeffs=[1.0] * len(hamiltonian), observables=hamiltonian)

    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(pennylane_hamiltonian)

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    energy = nif_device.execute_generator(circuit_gen(), observable=pennylane_hamiltonian, output_type="expval")

    q_node = qml.QNode(circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "basis_state,hamiltonian",
    [
        ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
        ([0, 1, 0, 1, 1], [qml.PauliZ(2) @ qml.PauliZ(3), qml.PauliZ(1) @ qml.PauliZ(2), qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 1, 1, 0], [qml.PauliZ(2) @ qml.PauliZ(3), qml.PauliZ(1) @ qml.PauliZ(2), qml.PauliZ(0) @ qml.PauliZ(1)]),
        ([0, 1, 1, 0, 1, 0, 0], [Z(0) @ Z(1), Z(1) @ Z(2), Z(2) @ Z(3)]),
    ]
)
def test_nif_batched_hamiltonian_expval_zz_on_basis_state_against_qubit_device(basis_state, hamiltonian):
    batched_hamiltonian = BatchHamiltonian(np.ones(len(hamiltonian)), hamiltonian)

    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(hamiltonian))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    nif_device.reset()
    energy = nif_device.execute_generator(circuit_gen(), observable=batched_hamiltonian, output_type="expval")

    q_node = qml.QNode(circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "basis_state,hamiltonian,contraction_strategy",
    [
        (
                np.random.choice([0, 1], size=n_wires),
                [
                    qml.PauliZ(w) @ qml.PauliZ(w+np.random.randint(1, n_wires-w))
                    for w in np.random.randint(n_wires-1, size=n_wires)
                ],
                contraction_strategy
        )
        for n_wires in range(2, 12)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for contraction_strategy in contraction_strategy_map.keys()
    ]
)
def test_nif_batched_hamiltonian_expval_zz_on_rn_basis_state_against_qubit_device(basis_state, hamiltonian, contraction_strategy):
    set_of_wires = set([obs.wires for obs in hamiltonian])
    dict_of_obs = {w: [obs for obs in hamiltonian if obs.wires == w] for w in set_of_wires}
    hamiltonian = [obs_list[0] for obs_list in dict_of_obs.values()]

    batched_hamiltonian = BatchHamiltonian(np.ones(len(hamiltonian)), hamiltonian)

    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(hamiltonian))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=contraction_strategy, name="lightning.qubit"
    )
    energy = nif_device.execute_generator(circuit_gen(), observable=batched_hamiltonian, output_type="expval")

    q_node = qml.QNode(circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)



@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_state,hamiltonian,contraction_strategy,op_gen",
    [
        (
                np.random.choice([0, 1], size=n_wires),
                [
                    qml.PauliZ(w) @ qml.PauliZ(w+np.random.randint(1, n_wires-w))
                    for w in np.random.randint(n_wires-1, size=n_wires)
                ],
                contraction_strategy,
                gen_cls(wires=n_wires, n_ops=2*n_wires, seed=i)
        )
        for n_wires in range(2, 6)
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for contraction_strategy in contraction_strategy_map.keys()
        for gen_cls in [RandomMatchgateHaarOperationsGenerator, RandomMatchgateOperationsGenerator]
    ]
)
def test_nif_batched_hamiltonian_expval_zz_on_rn_mop_gen_against_qubit_device(
        basis_state, hamiltonian, contraction_strategy, op_gen
):
    set_of_wires = set([obs.wires for obs in hamiltonian])
    dict_of_obs = {w: [obs for obs in hamiltonian if obs.wires == w] for w in set_of_wires}
    hamiltonian = [obs_list[0] for obs_list in dict_of_obs.values()]

    op_gen.initial_state = basis_state

    batched_hamiltonian = BatchHamiltonian(np.ones(len(hamiltonian)), hamiltonian)
    nif_device, qubit_device = devices_init(
        wires=op_gen.wires, shots=None, contraction_strategy=contraction_strategy,
        name="lightning.qubit"
    )
    energy = nif_device.execute_generator(op_gen, observable=batched_hamiltonian, output_type="expval")

    op_gen.observable = sum(hamiltonian)
    op_gen.output_type = "expval"
    q_node = qml.QNode(op_gen.circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_state,hamiltonian,contraction_strategy,op_gen",
    [
        (
                np.random.choice([0, 1], size=n_wires),
                [
                    qml.PauliZ(w) @ qml.PauliZ(w+np.random.randint(1, n_wires-w))
                    for w in np.random.randint(n_wires-1, size=n_wires)
                ],
                contraction_strategy,
                gen_cls(wires=n_wires, n_ops=2*n_wires, seed=i)
        )
        for n_wires in range(2, 6)
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for contraction_strategy in contraction_strategy_map.keys()
        for gen_cls in [RandomMatchgateHaarOperationsGenerator, RandomMatchgateOperationsGenerator]
    ]
)
def test_nif_batched_hamiltonian_expval_zz_on_rn_mop_gen_against_qubit_device_ham(
        basis_state, hamiltonian, contraction_strategy, op_gen
):
    set_of_wires = set([obs.wires for obs in hamiltonian])
    dict_of_obs = {w: [obs for obs in hamiltonian if obs.wires == w] for w in set_of_wires}
    hamiltonian = [obs_list[0] for obs_list in dict_of_obs.values()]

    op_gen.initial_state = basis_state

    batched_hamiltonian = BatchHamiltonian(np.ones(len(hamiltonian)), hamiltonian)
    nif_device, qubit_device = devices_init(
        wires=op_gen.wires, shots=None, contraction_strategy=contraction_strategy,
        name="lightning.qubit"
    )
    energy = nif_device.execute_generator(op_gen, observable=batched_hamiltonian, output_type="expval")

    op_gen.observable = qml.Hamiltonian(np.ones(len(hamiltonian)), hamiltonian)
    op_gen.output_type = "expval"
    q_node = qml.QNode(op_gen.circuit, qubit_device)
    expected_energy = q_node()

    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)
