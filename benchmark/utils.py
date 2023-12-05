import pennylane as qml
import sys
import numpy as np
import os
try:
    import msim
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    import msim
from msim import NonInteractingFermionicDevice, MatchgateOperator, utils


MPL_RC_DEFAULT_PARAMS = {
    "font.size": 18,
    "legend.fontsize": 16,
    "lines.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "DejaVu Sans",
        "Arial",
        "Tahoma",
        "calibri",
    ],
}

MPL_RC_BIG_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 22,
    "legend.fontsize": 20,
}}

MPL_RC_SMALL_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 12,
    "legend.fontsize": 10,
}}


def get_device_memory_usage(device: qml.Device) -> int:
    """Get the memory usage of a device in bytes.

    Args:
        device: The device to get the memory usage of.

    Returns:
        The memory usage of the device in bytes.
    """
    if isinstance(device, NonInteractingFermionicDevice):
        return device.memory_usage
    else:
        return device.state.size * device.state.dtype.itemsize


def init_nif_device(*args, **kwargs) -> NonInteractingFermionicDevice:
    wires = kwargs.pop("wires", 2)
    nif_device = NonInteractingFermionicDevice(
        wires=wires,
        prob_strategy=kwargs.pop("prob_strategy", "lookup_table"),
        **kwargs,
    )
    return nif_device


def init_qubit_device(*args, **kwargs) -> qml.Device:
    wires = kwargs.pop("wires", 2)
    qubit_device = qml.device(kwargs.pop("name", 'default.qubit'), wires=wires, shots=kwargs.get("shots", None))
    qubit_device.operations.add(MatchgateOperator)
    return qubit_device

