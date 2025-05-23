from typing import Tuple

import pennylane as qml

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake import utils
from .test_single_matchgate_circuit import single_matchgate_circuit

_majorana_getters = {}


def init_nif_device(*args, **kwargs) -> NonInteractingFermionicDevice:
    wires = kwargs.pop("wires", 2)
    if isinstance(wires, int):
        n_particles = wires
    else:
        n_particles = len(wires)
    majorana_getter = _majorana_getters.get(
        n_particles,
        utils.majorana.MajoranaGetter(n_particles, maxsize=kwargs.pop("maxsize", 1024)),
    )
    _majorana_getters[n_particles] = majorana_getter
    nif_device = NonInteractingFermionicDevice(
        wires=wires,
        prob_strategy=kwargs.pop("prob_strategy", "LookupTable"),
        majorana_getter=kwargs.pop("majorana_getter", majorana_getter),
        n_workers=kwargs.pop("n_workers", 0),
        contraction_strategy=kwargs.pop("contraction_strategy", None),
        **kwargs,
    )
    return nif_device


def init_qubit_device(*args, **kwargs) -> qml.devices.Device:
    wires = kwargs.pop("wires", 2)
    qubit_device = qml.device(
        kwargs.pop("name", "default.qubit"),
        wires=wires,
        shots=kwargs.get("shots", None),
    )
    return qubit_device


def devices_init(
    *args, **kwargs
) -> Tuple[NonInteractingFermionicDevice, qml.devices.Device]:
    qubit_device = init_qubit_device(*args, **kwargs)
    nif_device = init_nif_device(*args, **kwargs)
    return nif_device, qubit_device
