from msim import utils, MatchgateStandardHamiltonianParams, MatchgateHamiltonianCoefficientsParams, \
    MatchgateStandardParams
from msim import matchgate_parameter_sets as mps
import numpy as np
import pennylane as qml

params = MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0)
std_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)
hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(params.to_matrix())
elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
std_h_params_ = MatchgateStandardHamiltonianParams.from_numpy(params_arr)

hamiltonian_coefficients_matrix = params.to_matrix()
n_particles = 2
coeffs, obs = [], []
for mu in range(2 * n_particles):
    for nu in range(2 * n_particles):
        c_mu = qml.pauli.string_to_pauli_word(utils.get_majorana_pauli_string(mu, n_particles, join_char=''))
        c_nu = qml.pauli.string_to_pauli_word(utils.get_majorana_pauli_string(nu, n_particles, join_char=''))
        coeffs.append(hamiltonian_coefficients_matrix[mu, nu])
        obs.append(c_mu @ c_nu)

qml.Hamiltonian(coeffs, obs).matrix()

print(f"{params = }")
print(f"{std_h_params = }")
print(f"{std_h_params_ = }")
print(f"{std_h_params == std_h_params_ = }")
print(f"{hamiltonian = }")

# mps.MatchgatePolarParams.RAISE_ERROR_IF_INVALID_PARAMS = False
# from_params = mps.MatchgatePolarParams(r0=1, r1=0, theta0=0, theta1=0, theta2=np.pi/2, theta3=0, theta4=np.pi / 2)
# print(f"{from_params = }")
# to_params = mps.MatchgateStandardParams.parse_from_params(from_params)
# print(f"{to_params = }")
# _from_params = mps.MatchgatePolarParams.parse_from_params(to_params)
# print(f"{_from_params = }")
# print(f"{from_params == _from_params = }")
#
#
# from_params = mps.MatchgateStandardParams(a=1, x=1, y=1, d=-1)
# print(f"{from_params = }")
# to_params = mps.MatchgatePolarParams.parse_from_params(from_params)
# print(f"{to_params = }")
# reconstructed_params = mps.MatchgateStandardParams.parse_from_params(to_params)
# print(f"{reconstructed_params = }")
# print(f"{from_params == reconstructed_params = }")




