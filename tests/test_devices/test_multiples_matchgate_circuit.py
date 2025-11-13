import numpy as np
import pennylane as qml
import psutil
import pytest

from matchcake import MatchgateOperation
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from matchcake.circuits import (
    RandomMatchgateHaarOperationsGenerator,
    RandomMatchgateOperationsGenerator,
    RandomOperationsGenerator,
    RandomSptmOperationsGenerator,
)
from matchcake.devices.contraction_strategies import contraction_strategy_map

from ..configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from . import devices_init
from .test_rotations_zyz_circuit import specific_matchgate_circuit

set_seed(TEST_SEED)


def multiples_matchgate_circuit(params_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = [0, 1]
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params in params_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        wire0 = np.random.choice(all_wires[:-1], size=1).item()
        wire1 = wire0 + 1
        MatchgateOperation(mg_params, wires=[wire0, wire1])
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "params_list,n_wires",
    [([mps.MatchgatePolarParams.random(), mps.MatchgatePolarParams.random()], 4)]
    + [
        (
            [mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)],
            num_wires,
        )
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
    ]
    + [
        (
            [mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)],
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
    ],
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, n_wires):
    nif_device, qubit_device = devices_init(wires=n_wires)
    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(
        specific_matchgate_circuit,
        qubit_device,
    )

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    qubit_probs = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=qubit_device.wires,
    )
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=nif_device.wires,
    )
    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "op_gen,contraction_strategy",
    [
        (
            RandomMatchgateHaarOperationsGenerator(wires=num_wires, n_ops=num_gates, output_type="probs", seed=i),
            contraction_strategy,
        )
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
        for contraction_strategy in contraction_strategy_map.keys()
    ],
)
def test_multiples_matchgate_probs_with_qbit_device_op_gen(op_gen, contraction_strategy):
    nif_device, qubit_device = devices_init(wires=op_gen.wires, contraction_strategy=contraction_strategy)
    qubit_qnode = qml.QNode(op_gen.circuit, qubit_device)
    qubit_probs = qubit_qnode()
    nif_probs = nif_device.execute_generator(op_gen, output_type=op_gen.output_type, observable=op_gen.observable)
    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        (
            [mps.MatchgatePolarParams.random().to_numpy() for _ in range(2 * num_wires)],
            num_wires,
            0,
        )
        # for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 5)
    ],
)
def test_multiples_matchgate_probs_with_qbit_device_mp(params_list, n_wires, prob_wires):
    pytest.skip()  # takes to long
    if psutil.cpu_count() < 2:
        pytest.skip("This test requires at least 2 CPUs.")
    n_workers = 2
    nif_device, qubit_device = devices_init(wires=n_wires, n_workers=n_workers)
    assert nif_device.n_workers == n_workers

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]

    qubit_state = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )

    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        (
            [mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)],
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 5)
        for num_gates in [1, 2 * num_wires]
    ],
)
def test_multiples_matchgate_apply_vs_apply_gen(params_list, n_wires):
    nif_device, _ = devices_init(wires=n_wires, contraction_strategy="neighbours")
    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
    )

    nif_device.reset()
    nif_device.apply(nif_qnode._tape.operations)
    apply_transition_matrix = nif_device.transition_matrix
    apply_metadata = nif_device.apply_metadata.copy()

    nif_device.reset()
    nif_device.apply_generator(nif_qnode._tape.operations)
    apply_gen_transition_matrix = nif_device.transition_matrix
    apply_gen_metadata = nif_device.apply_metadata.copy()

    np.testing.assert_allclose(
        apply_transition_matrix,
        apply_gen_transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
