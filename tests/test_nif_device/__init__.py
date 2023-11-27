from typing import Tuple

import pennylane as qml

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import utils
from .test_single_matchgate_circuit import single_matchgate_circuit

_majorana_getters = {}


def init_nif_device(*args, **kwargs) -> NonInteractingFermionicDevice:
    wires = kwargs.pop("wires", 2)
    majorana_getter = _majorana_getters.get(
        wires,
        utils.majorana.MajoranaGetter(wires, maxsize=kwargs.pop("maxsize", 1024)),
    )
    _majorana_getters[wires] = majorana_getter
    nif_device = NonInteractingFermionicDevice(
        wires=wires,
        prob_strategy=kwargs.pop("prob_strategy", "lookup_table"),
        majorana_getter=kwargs.pop("majorana_getter", majorana_getter),
    )
    return nif_device


def init_qubit_device(*args, **kwargs) -> qml.Device:
    wires = kwargs.pop("wires", 2)
    qubit_device = qml.device(kwargs.pop("name", 'default.qubit'), wires=wires, shots=kwargs.get("shots", None))
    qubit_device.operations.add(MatchgateOperator)
    return qubit_device


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    qubit_device = init_qubit_device(*args, **kwargs)
    nif_device = init_nif_device(*args, **kwargs)
    return nif_device, qubit_device
