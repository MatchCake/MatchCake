from typing import Tuple

import pennylane as qml

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import utils
from .test_single_matchgate_circuit import single_matchgate_circuit

_majorana_getters = {}


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    wires = kwargs.pop("wires", 2)
    qubit_device = qml.device('default.qubit', wires=wires, shots=kwargs.get("shots", None))
    qubit_device.operations.add(MatchgateOperator)
    majorana_getter = _majorana_getters.get(
        qubit_device.num_wires,
        utils.majorana.MajoranaGetter(qubit_device.num_wires, maxsize=kwargs.pop("maxsize", 1024)),
    )
    _majorana_getters[qubit_device.num_wires] = majorana_getter
    nif_device = NonInteractingFermionicDevice(
        wires=wires,
        prob_strategy=kwargs.pop("prob_strategy", "lookup_table"),
        majorana_getter=majorana_getter,
    )
    return nif_device, qubit_device
