import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


def circuit(params):
    qml.QubitUnitary(U, wires=0)
    return qml.expval(qml.PauliZ(0))


if __name__ == '__main__':
    simple_U = np.array([[1, 0], [0, 1]])




