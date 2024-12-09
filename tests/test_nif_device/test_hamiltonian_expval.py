import numpy as np
import pennylane as qml
import pytest
import psutil

from matchcake import MatchgateOperation, utils, BatchHamiltonian
from matchcake.operations import SptmfRxRx
from matchcake.utils import torch_utils
from matchcake import matchgate_parameter_sets as mps
from matchcake.circuits import random_sptm_operations_generator
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
    "basis_state,hamiltonian,expected_energy",
    [
        ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)], 1.0),
        ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)], -1.0),
        ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)], -1.0),
        ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)], 1.0),
        ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
    ]
)
def test_nif_batched_hamiltonian_expval_zz_on_basis_state(basis_state, hamiltonian, expected_energy):
    hamiltonian = BatchHamiltonian(np.ones(len(hamiltonian)), hamiltonian)

    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    nif_device, _ = devices_init(wires=len(basis_state), shots=None, contraction_strategy=None)
    energy = nif_device.execute_generator(circuit_gen(), observable=hamiltonian, output_type="expval")
    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "basis_state,hamiltonian,expected_energy",
    [
        ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)], 1.0),
        ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)], -1.0),
        ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)], -1.0),
        ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)], 1.0),
        ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
        ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], 2.0),
        ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)], -2.0),
    ]
)
def test_nif_sum_hamiltonian_expval_zz_on_basis_state(basis_state, hamiltonian, expected_energy):
    def circuit_gen():
        yield qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return

    nif_device, _ = devices_init(wires=len(basis_state), shots=None, contraction_strategy=None)
    energy = sum(
        nif_device.execute_generator(circuit_gen(), observable=obs, output_type="expval")
        for obs in hamiltonian
    )
    np.testing.assert_allclose(energy, expected_energy, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

