import pennylane as qml
import matplotlib.pyplot as plt


def circuit(params):
    qml.QubitUnitary(U, wires=0)
    return qml.expval(qml.PauliZ(0))

