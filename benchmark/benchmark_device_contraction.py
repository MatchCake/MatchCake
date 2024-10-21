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
            # mc.operations.fRYY(layer_params, wires=curr_wires)
            # mc.operations.fRZZ(layer_params, wires=curr_wires)
            mc.operations.SptmRyRy(layer_params, wires=curr_wires)
            mc.operations.SptmRzRz(layer_params, wires=curr_wires)
        for i, odd_wire in enumerate(wires[1:-1:2]):
            idx = list(wires).index(odd_wire)
            # mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
            mc.operations.SptmFSwap(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)


def run_circuit(contraction_method: Optional[str], **kwargs):
    nif_device = mc.NonInteractingFermionicDevice(
        wires=kwargs.get("wires", 128),
        show_progress=True,
        contraction_strategy=contraction_method
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
    n_ops = len(nif_qnode.tape.operations)
    n_contracted_ops = len(nif_device.contraction_strategy(nif_qnode.tape.operations))
    return dict(
        time=end_time - start_time,
        n_ops=n_ops,
        n_contracted_ops=n_contracted_ops,
        delta_ops=n_ops - n_contracted_ops,
        contraction_method=str(contraction_method),
    )


def run_one_batch(**sim_params):
    data_list = []
    for key in ["forward", "neighbours", "horizontal", "vertical", None]:
        key_data = run_circuit(contraction_method=key, **sim_params)
        data_list.append(key_data)
        print(f"Time with {key} contraction: {datetime.timedelta(seconds=key_data['time'])}")
    return data_list


def run_n_batches(n_batches: int, **sim_params):
    results = []
    for _ in range(n_batches):
        results.extend(run_one_batch(**sim_params))
    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sim_params = dict(
        wires=42,
        n_features=128,
        batch_size=32,
    )

    df = run_n_batches(5, **sim_params)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = np.asarray([axes]).ravel()

    axes[0] = sns.barplot(data=df, x="contraction_method", y="time", ax=axes[0], capsize=0.5)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Time to run the circuit with different contraction methods")

    axes[1] = sns.barplot(data=df, x="contraction_method", y="delta_ops", ax=axes[1], capsize=0.5)
    axes[1].set_ylabel("Number of operations")
    axes[1].set_title("Number of operations removed by the contraction method")

    figures_folder = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_folder, exist_ok=True)
    fig.savefig(os.path.join(figures_folder, "time_to_run_circuit_new_matmul.pdf"), bbox_inches="tight")
    plt.show()


