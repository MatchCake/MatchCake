from typing import Tuple

import numpy as np
import pennylane as qml
from pennylane.ops.qubit.observables import BasisStateProjector

from matchcake import MatchgateOperation, NonInteractingFermionicDevice, utils

_majorana_getters = {}


def single_matchgate_circuit(params, initial_state=np.array([0, 0]), **kwargs):
    qml.BasisState(initial_state, wires=[0, 1])
    MatchgateOperation(params, wires=[0, 1], **kwargs)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


def specific_matchgate_circuit(params_wires_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = set(sum([list(wires) for _, wires in params_wires_list], []))
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params, wires in params_wires_list:
        MatchgateOperation(params, wires=wires)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    elif out_op == "expval":
        projector: BasisStateProjector = qml.Projector(initial_state, wires=all_wires)
        return qml.expval(projector)
    elif out_op == "sample":
        return qml.sample(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


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


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.devices.Device]:
    qubit_device = init_qubit_device(*args, **kwargs)
    nif_device = init_nif_device(*args, **kwargs)
    return nif_device, qubit_device
