import matchcake as mc
import numpy as np
import pennylane as qml
from pennylane.ops.qubit.observables import BasisStateProjector

# Create a Non-Interacting Fermionic Device with 4 qubits/wires
nif_device = mc.NonInteractingFermionicDevice(wires=4)
# Define the initial state as the all-zero computational basis state: |0000>
initial_state = np.zeros(len(nif_device.wires), dtype=int)


# Define a quantum circuit
def circuit(params, wires, initial_state):
    # Prepare the initial state
    qml.BasisState(initial_state, wires=wires)
    for i, even_wire in enumerate(wires[:-1:2]):
        idx = list(wires).index(even_wire)
        curr_wires = [wires[idx], wires[idx + 1]]
        # Apply the matchgate M(Rx(params), Rx(params))
        mc.operations.CompRxRx(params, wires=curr_wires)
        # Apply the matchgate M(Ry(params), Ry(params))
        mc.operations.CompRyRy(params, wires=curr_wires)
        # Apply the matchgate M(Rz(params), Rz(params))
        mc.operations.CompRzRz(params, wires=curr_wires)
    for i, odd_wire in enumerate(wires[1:-1:2]):
        idx = list(wires).index(odd_wire)
        mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)


# Create a QNode
nif_qnode = qml.QNode(circuit, nif_device)
qml.draw_mpl(nif_qnode)(
    params=np.array([0.1, 0.2]), wires=nif_device.wires, initial_state=initial_state
)

# Evaluate the QNode
expval = nif_qnode(
    params=np.array([0.1, 0.2]), wires=nif_device.wires, initial_state=initial_state
)
print(f"Expectation value: {expval:.4f}")

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    plt.savefig("../images/minimal_example_quantum_circuit_simulation.svg", dpi=900)
    plt.show()
