from pennylane.pauli import PauliWord, pauli_word_to_string


class _PauliMap(dict):
    def find_item(self, pauli):
        if isinstance(pauli, PauliWord):
            pauli_str = pauli_word_to_string(pauli)
        else:
            pauli_str = pauli
        for key, value in self.items():
            if key in pauli_str:
                return key, value
        raise KeyError(f'Pauli {pauli} not found in map')

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        return self.find_item(item)[1]


_MAJORANA_INDICES_LAMBDAS = _PauliMap({
    "ZI": (lambda k: (2 * k, 2 * k + 1)),
    "XX": (lambda k: (2 * k + 1, 2 * k + 2)),
    "YY": (lambda k: (2 * k, 2 * k + 3)),
    "YX": (lambda k: (2 * k, 2 * k + 2)),
    "XY": (lambda k: (2 * k + 1, 2 * k + 3)),
    "IZ": (lambda k: (2 * k + 2, 2 * k + 3)),
})

_MAJORANA_COEFFS_MAP = _PauliMap({
    "ZI": -1j,
    "XX": 1j,
    "YY": 1j,
    "YX": 1j,
    "XY": -1j,
    "IZ": -1j,
})
