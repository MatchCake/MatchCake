import pennylane as qml
from pennylane import numpy as pnp

from msim import MatchgateOperator
from msim import matchgate_parameter_sets as mps

dev = qml.device('default.qubit', wires=2, shots=100)
dev.operations.add(MatchgateOperator)


@qml.qnode(dev)
def circuit(theta):
    mg_params = mps.MatchgatePolarParams(theta, *pnp.ones(5))
    op = MatchgateOperator(mg_params, wires=[0, 1])
    # qml.apply(op)
    qml.QubitUnitary(op.matrix(), wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


if __name__ == '__main__':
    print(qml.draw(circuit)(0.6))
    print(circuit(0.6))










