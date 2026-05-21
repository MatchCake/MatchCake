from typing import Sequence, Tuple

import numpy as np


class JordanWigner:
    """Jordan-Wigner transformation for an n-qubit system.

    Convention (0-based):
        c_{2k}   = Z_0 Z_1 ... Z_{k-1}  X_k
        c_{2k+1} = Z_0 Z_1 ... Z_{k-1}  Y_k

    Single-qubit dictionary:
        X_k  ->  c_{2k}              (phase +1)
        Y_k  ->  c_{2k+1}            (phase +1)
        Z_k  ->  -i c_{2k} c_{2k+1} (phase -i)
        I_k  ->  (nothing)

    JW strings: X_k and Y_k each carry Z_0...Z_{k-1}; each Z_j = -i c_{2j} c_{2j+1}.
    Indices are sorted via insertion sort; duplicate pairs are cancelled via c^2 = 1.
    """

    @staticmethod
    def majorana_x_index(qubit: int) -> int:
        """Index of c_{2k} (the X-type Majorana on qubit k)."""
        return 2 * qubit

    @staticmethod
    def majorana_y_index(qubit: int) -> int:
        """Index of c_{2k+1} (the Y-type Majorana on qubit k)."""
        return 2 * qubit + 1

    @staticmethod
    def qubit_of_majorana(mu: int) -> int:
        """Qubit index for Majorana operator c_mu."""
        return mu // 2

    @staticmethod
    def is_x_type(mu: int) -> bool:
        """True if c_mu is the X-type Majorana (even index)."""
        return mu % 2 == 0

    @staticmethod
    def _sort_and_cancel(indices: list[int]) -> Tuple[np.ndarray, complex]:
        """Sort a Majorana index list via insertion sort, tracking anticommutation signs, and cancel duplicates.

        Each transposition contributes a factor of -1 to the phase.
        Adjacent equal indices are cancelled immediately using c_mu^2 = 1.
        """
        arr = list(indices)
        sign = 1

        i = 0
        while i < len(arr):
            j = i
            while j > 0 and arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                sign *= -1
                j -= 1
            if j > 0 and arr[j - 1] == arr[j]:
                arr.pop(j)
                arr.pop(j - 1)
                i = max(0, i - 1)
            else:
                i += 1

        return np.array(arr, dtype=int), complex(sign)

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def pauli_to_majorana(
        self,
        pauli_string: str,
        wires: Sequence[int],
    ) -> Tuple[np.ndarray, complex]:
        """Return (sorted_indices, phase) such that P = phase * c_{mu_1} ... c_{mu_m}.

        :param pauli_string: One character per wire in ``wires`` order; must be in {'I', 'X', 'Y', 'Z'}.
        :param wires: Wire labels in qubit order; len(wires) == len(pauli_string).
        :return: (sorted_indices, phase). ``phase`` is complex.
        """
        assert len(pauli_string) == len(wires), (
            f"len(pauli_string)={len(pauli_string)} != len(wires)={len(wires)}"
        )
        wire_to_qubit = {w: k for k, w in enumerate(wires)}

        indices: list[int] = []
        phase: complex = 1.0 + 0j

        for char, wire in zip(pauli_string, wires):
            k = wire_to_qubit[wire]
            if char == "I":
                continue
            elif char == "X":
                for j in range(k):
                    indices.extend([2 * j, 2 * j + 1])
                    phase *= -1j
                indices.append(2 * k)
            elif char == "Y":
                for j in range(k):
                    indices.extend([2 * j, 2 * j + 1])
                    phase *= -1j
                indices.append(2 * k + 1)
            elif char == "Z":
                indices.extend([2 * k, 2 * k + 1])
                phase *= -1j
            else:
                raise ValueError(f"Unknown Pauli character: {char!r}")

        if not indices:
            return np.array([], dtype=int), phase

        sorted_indices, perm_sign = self._sort_and_cancel(indices)
        phase *= perm_sign
        return sorted_indices, phase
