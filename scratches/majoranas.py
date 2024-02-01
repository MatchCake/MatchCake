from matchcake import utils
from matchcake import mps
import sympy as sp
import pennylane as qml

# c_0 = utils.get_majorana(3, 2)
# c_0_pauli_str = utils.get_majorana_pauli_string(3, 2, join_char='')
# print(c_0_pauli_str)
# c_0_p = qml.pauli.string_to_pauli_word(c_0_pauli_str).matrix()
# print(c_0_p)
# c_0_paulis = qml.pauli.pauli_decompose(c_0)
#
# print(c_0_paulis)


c0, c1 = utils.get_majorana_pauli_string(0, 2, join_char=''), utils.get_majorana_pauli_string(1, 2, join_char='')
# print(c0, c1)
c0 = qml.pauli.string_to_pauli_word(c0)
c1 = qml.pauli.string_to_pauli_word(c1)
# print(c0, c1)
c0 = utils.get_majorana(0, 2)
c1 = utils.get_majorana(1, 2)
# print(c_0, c_1)

c0_c1 = c0 @ c1
# print(c0_c1)
# print(qml.pauli.pauli_decompose(c0_c1))


p = mps.MatchgateHamiltonianCoefficientsParams(
    *mps.MatchgateHamiltonianCoefficientsParams.to_sympy(),
    backend='sympy'
)
h = p.compute_hamiltonian()

# pretty print h

print(sp.pretty(h, wrap_line=False))


