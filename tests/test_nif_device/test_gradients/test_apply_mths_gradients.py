import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.ops.qubit.observables import BasisStateProjector
from scipy.linalg import expm

from matchcake import NonInteractingFermionicDevice, utils
from matchcake.base import NonInteractingFermionicLookupTable
from matchcake.devices.probability_strategies import LookupTableStrategy
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import torch_utils
from ... import get_slow_test_mark
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    set_seed,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_apply_op_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        new_op = nif_device.apply_op(op)
        return new_op.matrix()

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
    ],
)
def test_apply_op_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device.apply_op(op)
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
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_apply_generator_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
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
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_execute_generator_gradient_check_with_sptm_circuit(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        nif_device.execute_generator([op])
        return nif_device.transition_matrix

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "input_matrix, system_state, target_binary_states",
    [
        (
            expm(np.random.randn(batch_size, 2 * size, 2 * size)),
            np.random.choice([0, 1], size=size),
            np.random.choice([0, 1], size=(batch_size, size)),
        )
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [1, 4]
    ],
)
def test_sptm_circuit_prob_strategy_batch_call_rn_transition_matrix_gradient_check(
    input_matrix, system_state, target_binary_states
):

    def get_output(matrix):
        nif_device = NonInteractingFermionicDevice(wires=input_matrix.shape[-1] // 2)
        transition_matrix = utils.make_transition_matrix_from_action_matrix(matrix)
        nif_device.transition_matrix = transition_matrix
        return nif_device.prob_strategy.batch_call(
            system_state=system_state,
            target_binary_states=target_binary_states,
            batch_wires=None,
            lookup_table=nif_device.lookup_table,
            pfaffian_method="det",
        )

    assert torch.autograd.gradcheck(
        get_output,
        torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "input_matrix, system_state, target_binary_states",
    [
        (
            expm(np.random.randn(batch_size, 2 * size, 2 * size)),
            np.random.choice([0, 1], size=size),
            np.random.choice([0, 1], size=(batch_size, size)),
        )
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [1, 4]
    ],
)
def test_sptm_circuit_prob_strategy_batch_call_rn_sptm_gradient_check(
    input_matrix, system_state, target_binary_states
):

    def circuit(sptm_matrix):
        nif_device = NonInteractingFermionicDevice(wires=input_matrix.shape[-1] // 2)
        nif_device.global_sptm = sptm_matrix
        return nif_device.prob_strategy.batch_call(
            system_state=nif_device.binary_state,
            target_binary_states=target_binary_states,
            batch_wires=None,
            lookup_table=nif_device.lookup_table,
            pfaffian_method="det",
        )

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "input_matrix, system_state, target_binary_states",
    [
        (
            expm(np.random.randn(batch_size, 2 * size, 2 * size)),
            np.random.choice([0, 1], size=size),
            np.random.choice([0, 1], size=(batch_size, size)),
        )
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [1, 4]
    ],
)
def test_sptm_circuit_prob_strategy_batch_call_gradient_check(
    input_matrix, system_state, target_binary_states
):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(wires=input_matrix.shape[-1] // 2)
        nif_device.apply_op(op)
        return nif_device.prob_strategy.batch_call(
            system_state=nif_device.binary_state,
            target_binary_states=target_binary_states,
            batch_wires=None,
            lookup_table=nif_device.lookup_table,
            pfaffian_method="det",
        )

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "input_matrix",
    [
        expm(np.random.randn(batch_size, 2 * size, 2 * size))
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [1, 4]
    ],
)
def test_sptm_circuit_probability_gradient_check(input_matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(
            wires=input_matrix.shape[-1] // 2, pfaffian_method="det"
        )
        nif_device.execute_generator([op])
        return nif_device.probability(nif_device.wires)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "matrix, obs",
    [
        (np.random.random((batch_size, 2 * size, 2 * size)), obs)
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for obs in [
            qml.PauliZ(0),
            sum([qml.PauliZ(i) for i in range(size)]),
            BasisStateProjector(np.zeros(size, dtype=int), wires=np.arange(size)),
        ]
    ],
)
def test_sptm_circuit_expval_gradient_check(matrix, obs):
    def circuit(p):
        nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        return nif_device.execute_generator(
            [op],
            observable=obs,
            output_type="expval",
        )

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "matrix, obs",
    [
        (np.random.random((batch_size, 2 * size, 2 * size)), obs)
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for obs in [
            qml.PauliZ(0),
            sum([qml.PauliZ(i) for i in range(size)]),
            BasisStateProjector(np.zeros(size, dtype=int), wires=np.arange(size)),
        ]
    ],
)
def test_sptm_circuit_exact_expval_gradient_check(matrix, obs):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(
            wires=matrix.shape[-1] // 2, pfaffian_method="det"
        )
        nif_device.execute_generator([op])
        return nif_device.exact_expval(obs)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_sptm_circuit_analytic_probability_gradient_check(matrix):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(
            wires=matrix.shape[-1] // 2, pfaffian_method="det"
        )
        nif_device.execute_generator([op])
        return nif_device.analytic_probability(nif_device.wires)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "input_matrix, target_binary_states",
    [
        (
            expm(np.random.randn(batch_size, 2 * size, 2 * size)),
            np.random.choice([0, 1], size=(batch_size, size)),
        )
        for size in [2, 4, 6]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [1, 4]
    ],
)
def test_sptm_circuit_get_states_probability_gradient_check(
    input_matrix, target_binary_states
):
    def circuit(p):
        op = SingleParticleTransitionMatrixOperation(
            matrix=p, wires=np.arange(p.shape[-1] // 2)
        )
        nif_device = NonInteractingFermionicDevice(
            wires=input_matrix.shape[-1] // 2, pfaffian_method="det"
        )
        nif_device.execute_generator([op])
        return nif_device.get_states_probability(target_binary_states)

    assert torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
