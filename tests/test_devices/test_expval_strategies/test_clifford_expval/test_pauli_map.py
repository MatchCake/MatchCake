import numpy as np
import pytest
from pennylane.pauli import string_to_pauli_word

from matchcake.devices.expval_strategies.clifford_expval._pauli_map import (
    _MAJORANA_COEFFS_MAP,
    _MAJORANA_INDICES_LAMBDAS,
)


class TestPauliMap:
    @pytest.mark.parametrize(
        "pauli_map, pauli",
        [
            (_MAJORANA_COEFFS_MAP, "XX"),
            (_MAJORANA_COEFFS_MAP, "XXI"),
            (_MAJORANA_INDICES_LAMBDAS, "XX"),
            (_MAJORANA_INDICES_LAMBDAS, "XXI"),
            (_MAJORANA_INDICES_LAMBDAS, "XXI"),
            (_MAJORANA_INDICES_LAMBDAS, string_to_pauli_word("XXII")),
        ]
        + [(_MAJORANA_COEFFS_MAP, p) for p in _MAJORANA_COEFFS_MAP.keys()]
        + [(_MAJORANA_INDICES_LAMBDAS, p) for p in _MAJORANA_INDICES_LAMBDAS.keys()],
    )
    def test_find_item_in_map(self, pauli_map, pauli):
        assert pauli_map.find_item(pauli)

    def test_find_item_not_in_map(self):
        with pytest.raises(KeyError):
            _MAJORANA_COEFFS_MAP.find_item("XIX")

    def test_contains(self):
        assert "XX" in _MAJORANA_COEFFS_MAP
        assert "XX" in _MAJORANA_INDICES_LAMBDAS
        assert "XXI" in _MAJORANA_INDICES_LAMBDAS
        assert "XIX" not in _MAJORANA_COEFFS_MAP
        assert "XIX" not in _MAJORANA_INDICES_LAMBDAS

    def test_getitem(self):
        np.testing.assert_almost_equal(_MAJORANA_COEFFS_MAP["XX"], -1j)
        assert _MAJORANA_INDICES_LAMBDAS["XX"](0) == (1, 2)
        assert _MAJORANA_INDICES_LAMBDAS["XXI"](0) == (1, 2)
