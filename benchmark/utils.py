import pennylane as qml
import sys
import numpy as np
from msim import NonInteractingFermionicDevice

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


