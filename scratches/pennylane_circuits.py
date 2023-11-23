from typing import Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import matchgate_parameter_sets as mps


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    nif_device = NonInteractingFermionicDevice(wires=kwargs.get("wires", 2))
    qubit_device = qml.device('default.qubit', wires=kwargs.get("wires", 2), shots=kwargs.get("shots", 1))
    qubit_device.operations.add(MatchgateOperator)
    return nif_device, qubit_device


def single_matchgate_circuit(params):
    h_params = mps.MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    op = MatchgateOperator(h_params, wires=[0, 1])
    qml.apply(op)
    return qml.probs(wires=0)


if __name__ == '__main__':
    nif_device, qubit_device = devices_init()
    print(f"{qubit_device.state = }")
    print(f"{nif_device.state = }")

    input_params = np.random.rand(6)
    # input_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)

    qubit_probs = qubit_qnode(input_params)
    print(f"{qubit_probs = }")
    nif_probs = nif_qnode(input_params)
    print(f"{nif_probs = }")
    check = np.allclose(nif_probs, qubit_probs)
    print(f"{check = }")










