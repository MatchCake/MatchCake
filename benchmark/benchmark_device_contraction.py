from typing import Optional
import os
import sys

try:
    import matchcake as mc
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    import matchcake as mc

import pennylane as qml
import numpy as np
from pennylane.ops.qubit.observables import BasisStateProjector
import time
import datetime
import pandas as pd


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
            mc.operations.fRYY(layer_params, wires=curr_wires)
            mc.operations.fRZZ(layer_params, wires=curr_wires)
        for i, odd_wire in enumerate(wires[1:-1:2]):
            idx = list(wires).index(odd_wire)
            mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)


def run_circuit(contraction_method: Optional[str], **kwargs):
    nif_device = mc.NonInteractingFermionicDevice(
        wires=kwargs.get("wires", 128),
        show_progress=True,
        contraction_method=contraction_method
    )
    initial_state = np.zeros(len(nif_device.wires), dtype=int)
    nif_qnode = qml.QNode(circuit, nif_device)
    n_features = kwargs.get("n_features", 4096)
    n_layers = int(np.ceil(n_features / nif_device.num_wires))  # Number of layers
    n_gate_params = kwargs.get("n_gate_params", 2)
    params = np.random.random((kwargs.get("batch_size", 8192), n_gate_params, n_layers))
    start_time = time.perf_counter()
    expval = nif_qnode(params, wires=nif_device.wires, initial_state=initial_state)
    end_time = time.perf_counter()
    return end_time - start_time


def run_one_batch(**sim_params):
    time_neighbours = run_circuit(contraction_method="neighbours", **sim_params)
    print(f"Time with neighbours contraction: {datetime.timedelta(seconds=time_neighbours)}")
    time_horizontal = run_circuit(contraction_method="horizontal", **sim_params)
    print(f"Time with horizontal contraction: {datetime.timedelta(seconds=time_horizontal)}")
    time_vertical = run_circuit(contraction_method="vertical", **sim_params)
    print(f"Time with vertical contraction: {datetime.timedelta(seconds=time_vertical)}")
    time_wo_contraction = run_circuit(contraction_method=None, **sim_params)
    print(f"Time without contraction: {datetime.timedelta(seconds=time_wo_contraction)}")
    return dict(
        neighbours=time_neighbours,
        horizontal=time_horizontal,
        vertical=time_vertical,
        none=time_wo_contraction
    )


def run_n_batches(n_batches: int, **sim_params):
    results = []
    for _ in range(n_batches):
        results.append(run_one_batch(**sim_params))
    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sim_params = dict(
        wires=128,
        n_features=256,
        batch_size=32,
    )

    df = run_n_batches(5, **sim_params)
    time_neighbours = df["neighbours"].mean()
    time_horizontal = df["horizontal"].mean()
    time_vertical = df["vertical"].mean()
    time_wo_contraction = df["none"].mean()

    fig, ax = plt.subplots()
    ax.bar(
        ["Neighbours", "Horizontal", "Vertical", "None"],
        [time_neighbours, time_horizontal, time_vertical, time_wo_contraction]
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Time to run the circuit with different contraction methods")
    figures_folder = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_folder, exist_ok=True)
    fig.savefig(os.path.join(figures_folder, "time_to_run_circuit_new_matmul.pdf"), bbox_inches="tight")
    plt.show()


