from importlib import metadata

import numpy as np
import pennylane as qml
import pytest

from matchcake.devices import NonInteractingFermionicDevice, SwapAugmentedFermionicDevice
from matchcake.operations import fSWAP

from ..configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON, TEST_SEED, set_seed

ENTRY_POINT_GROUP = "pennylane.plugins"


class TestDeviceRegistration:
    @classmethod
    def setup_class(cls) -> None:
        set_seed(TEST_SEED)

    @staticmethod
    def _entry_points() -> dict[str, str]:
        entries = metadata.entry_points(group=ENTRY_POINT_GROUP)
        return {entry.name: entry.value for entry in entries}

    @pytest.mark.parametrize(
        "short_name,expected_class",
        [
            ("nif.qubit", NonInteractingFermionicDevice),
            ("nif.swap.qubit", SwapAugmentedFermionicDevice),
        ],
    )
    def test_entry_point_is_declared(self, short_name: str, expected_class: type) -> None:
        entry_points = self._entry_points()
        assert short_name in entry_points
        module_path, _, attribute = entry_points[short_name].partition(":")
        assert attribute == expected_class.__name__
        assert expected_class.__module__ == module_path

    @pytest.mark.parametrize(
        "short_name,expected_class",
        [
            ("nif.qubit", NonInteractingFermionicDevice),
            ("nif.swap.qubit", SwapAugmentedFermionicDevice),
        ],
    )
    def test_device_resolves_by_name(self, short_name: str, expected_class: type) -> None:
        device = qml.device(short_name, wires=2)
        assert isinstance(device, expected_class)

    @pytest.mark.parametrize("short_name", ["nif.qubit", "nif.swap.qubit"])
    def test_registered_name_matches_class_attribute(self, short_name: str) -> None:
        device = qml.device(short_name, wires=2)
        assert device.name == short_name

    def test_resolved_device_executes_circuit(self) -> None:
        device = qml.device("nif.qubit", wires=2)

        @qml.qnode(device)
        def circuit() -> qml.measurements.ExpectationMP:
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            fSWAP(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        np.testing.assert_allclose(
            float(result),
            1.0,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
