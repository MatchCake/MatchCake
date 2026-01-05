import numpy as np
import pytest
import torch
from pennylane.ops.qubit import QubitUnitary
from pennylane.pauli import string_to_pauli_word

from matchcake.devices.expval_strategies.clifford_expval._pauli_map import (
    _MAJORANA_COEFFS_MAP,
    _MAJORANA_INDICES_LAMBDAS,
)
from matchcake.utils.majorana import majorana_to_pauli
from matchcake.utils.torch_utils import to_tensor


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


class Test_MAJORANA_INDICES_LAMBDAS:
    @pytest.mark.parametrize("k, pauli_string", [(k, ps) for k in range(6) for ps in _MAJORANA_INDICES_LAMBDAS.keys()])
    def test_transformation(self, k, pauli_string):
        pauli_word = QubitUnitary(string_to_pauli_word(pauli_string).matrix(), wires=[k, k + 1])

        majorana_indices = _MAJORANA_INDICES_LAMBDAS[pauli_string](k)
        majorana_coeff = _MAJORANA_COEFFS_MAP[pauli_string]
        pauli_from_majorana = majorana_to_pauli(majorana_indices[0]) @ majorana_to_pauli(majorana_indices[1])
        pauli_from_majorana_coeff = pauli_from_majorana * majorana_coeff

        all_wires = list(range(max(pauli_word.wires + pauli_from_majorana_coeff.wires) + 1))
        pauli_word_tensor = to_tensor(pauli_word.matrix(all_wires), dtype=torch.complex128)
        pauli_from_majorana_tensor = to_tensor(pauli_from_majorana_coeff.matrix(all_wires), dtype=torch.complex128)
        torch.testing.assert_close(pauli_word_tensor, pauli_from_majorana_tensor)
