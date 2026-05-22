import numpy as np
import pytest
from pennylane.pauli import PauliWord

from matchcake.utils.jordan_wigner import JordanWigner


class TestJordanWigner:
    def test_majorana_x_index(self):
        for k in range(5):
            assert JordanWigner.majorana_x_index(k) == 2 * k

    def test_majorana_y_index(self):
        for k in range(5):
            assert JordanWigner.majorana_y_index(k) == 2 * k + 1

    def test_qubit_of_majorana(self):
        for mu in range(10):
            assert JordanWigner.qubit_of_majorana(mu) == mu // 2

    def test_is_x_type(self):
        for mu in range(10):
            assert JordanWigner.is_x_type(mu) == (mu % 2 == 0)

    def test_invalid_pauli_character_raises(self):
        jw = JordanWigner(2)
        with pytest.raises(ValueError, match="Unknown Pauli character"):
            jw.pauli_to_majorana("AX", [0, 1])

    @pytest.mark.parametrize("n_qubits,pauli_str,wires,exp_len", [
        (1, "I", [0], 0),
        (1, "X", [0], 1),
        (1, "Y", [0], 1),
        (1, "Z", [0], 2),
        (2, "XX", [0, 1], 2),
        (2, "ZZ", [0, 1], 4),
    ])
    def test_pauli_to_majorana_index_count(self, n_qubits, pauli_str, wires, exp_len):
        jw = JordanWigner(n_qubits)
        indices, phase = jw.pauli_to_majorana(pauli_str, wires)
        assert len(indices) == exp_len

    def test_identity_returns_empty_and_unit_phase(self):
        jw = JordanWigner(3)
        indices, phase = jw.pauli_to_majorana("III", [0, 1, 2])
        np.testing.assert_array_equal(indices, [])
        assert abs(phase - 1.0) < 1e-12

    def test_sort_and_cancel_removes_duplicates(self):
        arr, sign = JordanWigner._sort_and_cancel([1, 1])
        np.testing.assert_array_equal(arr, [])
        assert sign == 1

    def test_sort_and_cancel_sorts_with_sign(self):
        arr, sign = JordanWigner._sort_and_cancel([1, 0])
        np.testing.assert_array_equal(arr, [0, 1])
        assert sign == -1

    def test_y_jw_string_on_qubit_k_gt_0(self):
        """Y on qubit 1 triggers the for-j loop body (lines 102-103 in jordan_wigner.py)."""
        jw = JordanWigner(2)
        indices, phase = jw.pauli_to_majorana("IY", [0, 1])
        # IY on [0,1]: Y is on wire 1 (k=1), loop runs once (j=0),
        # extending [0,1] and multiplying phase by -1j, then appending 3.
        # After sort_and_cancel([0,1,3]) = ([0,1,3], 1), final phase = -1j.
        np.testing.assert_array_equal(indices, [0, 1, 3])
        assert abs(phase - (-1j)) < 1e-12

    def test_wrong_length_raises(self):
        jw = JordanWigner(2)
        with pytest.raises(AssertionError):
            jw.pauli_to_majorana("XYZ", [0, 1])

    def test_string_without_wires_raises(self):
        jw = JordanWigner(1)
        with pytest.raises(ValueError, match="wires must be provided"):
            jw.pauli_to_majorana("X")

    def test_pauli_word_matches_string(self):
        jw = JordanWigner(2)
        wires = [0, 1]
        for pauli_str in ("IX", "IY", "IZ", "XI", "YI", "ZI", "XX", "YY", "XY", "YX", "ZZ"):
            indices_str, phase_str = jw.pauli_to_majorana(pauli_str, wires)
            word = PauliWord({w: c for w, c in zip(wires, pauli_str) if c != "I"})
            indices_word, phase_word = jw.pauli_to_majorana(word, wires)
            np.testing.assert_array_equal(indices_str, indices_word)
            np.testing.assert_almost_equal(phase_str, phase_word)

    def test_pauli_word_default_wires(self):
        jw = JordanWigner(2)
        word = PauliWord({0: "X", 1: "Y"})
        indices_word, phase_word = jw.pauli_to_majorana(word)
        indices_str, phase_str = jw.pauli_to_majorana("XY", [0, 1])
        np.testing.assert_array_equal(indices_word, indices_str)
        np.testing.assert_almost_equal(phase_word, phase_str)

    def test_pauli_word_missing_wire_treated_as_identity(self):
        jw = JordanWigner(2)
        word = PauliWord({1: "X"})
        indices_word, phase_word = jw.pauli_to_majorana(word, [0, 1])
        indices_str, phase_str = jw.pauli_to_majorana("IX", [0, 1])
        np.testing.assert_array_equal(indices_word, indices_str)
        np.testing.assert_almost_equal(phase_word, phase_str)
