import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import matchgate_parameter_sets as mps

dev = qml.device('default.qubit', wires=2, shots=100)
dev.operations.add(MatchgateOperator)


@qml.qnode(dev)
def circuit(params):
    mg_params = mps.MatchgatePolarParams(*params)
    op = MatchgateOperator(mg_params, wires=[0, 1])
    qml.apply(op)
    # qml.QubitUnitary(op.matrix(), wires=[0, 1])
    # return qml.expval(qml.Identity(wires=0))
    return qml.probs(wires=0), qml.expval(qml.Identity(wires=0))


mdev = NonInteractingFermionicDevice(wires=2)


@qml.qnode(mdev)
def mdev_circuit(params):
    mg_params = mps.MatchgatePolarParams(*params)
    op = MatchgateOperator(mg_params, wires=[0, 1])
    qml.apply(op)
    # return qml.expval(qml.Identity(wires=0))
    return qml.probs(wires=0), qml.expval(qml.Identity(wires=0))


if __name__ == '__main__':
    
    print(f"{dev.state = }")
    print(f"{mdev.state = }")
    
    # input_params = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    input_params = np.random.rand(6)
    # print(qml.draw(circuit)(input_params))
    print(f"{circuit(input_params) = }")
    
    # print(qml.draw(mdev_circuit)(input_params))
    print(f"{mdev_circuit(input_params) = }")










