import matchcake as mc
import pennylane as qml
import numpy as np
from pennylane.ops.qubit.observables import BasisStateProjector
import time


def circuit(params, wires, initial_state=None):
    """
    Circuit that applies the fRXX, fRYY, fRZZ, and fSWAP operations to the initial state.

    :param params: The parameters of the circuit. It should have shape (batch_size, 2, n_layers). 
    :param wires: The wires of the circuit.
    :param initial_state: The initial state of the circuit. It should be a numpy array with shape (len(wires),) of zeros and ones.
    :return: The expectation value of the circuit.
    """
    qml.BasisState(initial_state, wires=wires)
    batch_size, n_gate_params, n_layers = qml.math.shape(params)
    if n_gate_params != 2:
        raise ValueError("The number of gate parameters should be 2.")
    for layer in range(n_layers):
        layer_params = params[..., layer]
        for i, even_wire in enumerate(wires[:-1:2]):
            idx = list(wires).index(even_wire)
            curr_wires = [wires[idx], wires[idx + 1]]
            mc.operations.fRXX(layer_params, wires=curr_wires)
            mc.operations.fRYY(layer_params, wires=curr_wires)
            mc.operations.fRZZ(layer_params, wires=curr_wires)
        for i, odd_wire in enumerate(wires[1:-1:2]):
            idx = list(wires).index(odd_wire)
            mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)


def run_circuit(use_less_einsum_for_transition_matrix: bool):
    nif_device = mc.NonInteractingFermionicDevice(wires=10, show_progress=True)
    initial_state = np.zeros(len(nif_device.wires), dtype=int)
    nif_qnode = qml.QNode(circuit, nif_device)
    n_layers = 32  # Number of layers
    n_gate_params = 2  # Number of parameters per gate
    params = np.random.random((1000, n_gate_params, n_layers))
    mc.Matchgate.DEFAULT_USE_LESS_EINSUM_FOR_TRANSITION_MATRIX = use_less_einsum_for_transition_matrix
    start_time = time.perf_counter()
    expval = nif_qnode(params, wires=nif_device.wires, initial_state=initial_state)
    end_time = time.perf_counter()
    print(mc.Matchgate.SPTM_CONTRACTION_PATH)
    return end_time - start_time


if __name__ == '__main__':
    time_with_einsum = run_circuit(use_less_einsum_for_transition_matrix=False)
    time_with_less_einsum = run_circuit(use_less_einsum_for_transition_matrix=True)
    print(f"Time with einsum: {time_with_einsum}")
    print(f"Time with less einsum: {time_with_less_einsum}")
    print(f"Speedup: {time_with_einsum / time_with_less_einsum}")

