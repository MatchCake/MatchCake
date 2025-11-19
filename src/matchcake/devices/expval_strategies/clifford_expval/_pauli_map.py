from collections import OrderedDict
from typing import Any

from pennylane.pauli import PauliWord, pauli_word_to_string


class _PauliMap(OrderedDict):
    def find_item(self, pauli: Any):
        if not isinstance(pauli, str):
            pauli_str = pauli_word_to_string(pauli)
        else:
            pauli_str = pauli
        for key, value in self.items():
            if key in pauli_str:
                return key, value
        raise KeyError(f"Pauli {pauli} not found in map")

    def __getitem__(self, item: Any):
        if super().__contains__(item):
            return super().__getitem__(item)
        return self.find_item(item)[1]

    def __contains__(self, item: Any):
        if super().__contains__(item):
            return True
        try:
            self.find_item(item)
            return True
        except KeyError:
            return False


_MAJORANA_INDICES_LAMBDAS = _PauliMap(
    {
        "XX": (lambda k: (2 * k + 1, 2 * k + 2)),
        "YY": (lambda k: (2 * k, 2 * k + 3)),
        "YX": (lambda k: (2 * k, 2 * k + 2)),
        "XY": (lambda k: (2 * k + 1, 2 * k + 3)),
        # "XX": (lambda k: (2 * k + 1, 2 * k + 2)),
        # "YY": (lambda k: (2 * k, 2 * k + 3)),
        # "YX": (lambda k: (2 * k, 2 * k + 2)),
        # "XY": (lambda k: (2 * k + 1, 2 * k + 3)),
        # TODO: Need to debug for ZI and IZ before adding them.
        # "ZI": (lambda k: (2 * k, 2 * k + 1)),
        # "IZ": (lambda k: (2 * k + 2, 2 * k + 3)),
    }
)

_MAJORANA_COEFFS_MAP = _PauliMap(
    {
        "XX": -1j,
        "YY": 1j,
        "YX": -1j,
        "XY": -1j,
        # TODO: Need to debug for ZI and IZ before adding them.
        # "ZI": -1j,
        # "IZ": -1j,
    }
)
