import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.ops.qubit.observables import BasisStateProjector

from matchcake import NonInteractingFermionicDevice
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import torch_utils
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_apply_op_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        global_sptm, batched = None, False
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        new_op, batched = nif_device._apply_op(op, batched, global_sptm)
        return new_op

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_apply_generator_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device.apply_generator([op])
        return nif_device.transition_matrix

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_execute_generator_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        nif_device.execute_generator([op])
        return nif_device.transition_matrix

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix, obs",
    [
        (np.random.random((batch_size, 2 * size, 2 * size)), obs)
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
        for obs in [
            qml.PauliZ(0),
            sum([qml.PauliZ(i) for i in range(size)]),
            BasisStateProjector(np.zeros(size, dtype=int), wires=np.arange(size)),
        ]
    ]
)
def test_sptm_circuit_expval_gradient_check(matrix, obs):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        return nif_device.execute_generator(
            [SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))],
            observable=obs,
            output_type="expval",
        )

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix, obs",
    [
        (np.random.random((batch_size, 2 * size, 2 * size)), obs)
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
        for obs in [
            qml.PauliZ(0),
            sum([qml.PauliZ(i) for i in range(size)]),
            BasisStateProjector(np.zeros(size, dtype=int), wires=np.arange(size)),
        ]
    ]
)
def test_sptm_circuit_exact_expval_gradient_check(matrix, obs):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2, pfaffian_method="det")
        nif_device.execute_generator([op])
        return nif_device.exact_expval(obs)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_circuit_probability_gradient_check(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2, pfaffian_method="det")
        nif_device.execute_generator([op])
        return nif_device.probability(nif_device.wires)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_circuit_analytic_probability_gradient_check(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2, pfaffian_method="det")
        nif_device.execute_generator([op])
        return nif_device.analytic_probability(nif_device.wires)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_circuit_get_states_probability_gradient_check(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2, pfaffian_method="det")
        nif_device.execute_generator([op])
        return nif_device.get_states_probability(np.arange(nif_device.num_wires, dtype=int))

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_circuit_get_states_probability_gradient_check(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        nif_device.execute_generator([op])
        target_binary_states = np.arange(nif_device.num_wires, dtype=int).reshape(1, -1)
        batch_wires = np.stack([[nif_device.wires] for _ in range(target_binary_states.shape[0])])
        return nif_device.prob_strategy.batch_call(
                system_state=nif_device.binary_state,
                target_binary_states=target_binary_states,
                batch_wires=batch_wires,
                all_wires=nif_device.wires,
                lookup_table=nif_device.lookup_table,
                transition_matrix=nif_device.transition_matrix,
                pfaffian_method="det",
                majorana_getter=nif_device.majorana_getter,
            )

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )
