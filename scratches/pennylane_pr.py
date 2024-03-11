import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.ops.qubit.observables import BasisStateProjector
import numpy as np


def circuit(x, y):
    n_wires = qml.math.shape(x)[-1]
    for i in range(n_wires):
        qml.RX(x[:, i], wires=i)
        qml.RY(y[:, i], wires=i)
    projector: BasisStateProjector = qml.Projector(np.zeros(n_wires), wires=range(n_wires))
    return qml.expval(projector)


if __name__ == '__main__':
    device = DefaultQubit(wires=2)
    qnode = qml.QNode(circuit, device)
    batch_size = len(device.wires) * 3 + 1
    x = np.random.uniform(0, 2 * np.pi, size=(batch_size, len(device.wires)))
    y = np.random.uniform(0, 2 * np.pi, size=(batch_size, len(device.wires)))
    out = qnode(x, y)
    np.testing.assert_allclose(qml.math.shape(out), (batch_size,))

