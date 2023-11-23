from msim import utils, MatchgateStandardHamiltonianParams, MatchgateHamiltonianCoefficientsParams, \
    MatchgateStandardParams
from msim import matchgate_parameter_sets as mps
import numpy as np
import pennylane as qml
from pennylane import utils as qml_utils

# params = MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0)
# std_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)
# hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(params.to_matrix())
# elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
# params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
# std_h_params_ = MatchgateStandardHamiltonianParams.from_numpy(params_arr)
# std_h_params_ = MatchgateStandardHamiltonianParams(
#     h0=hamiltonian[0, 0],
#     h1=hamiltonian[0, 1],
#     h2=hamiltonian[0, 2],
#     h3=hamiltonian[1, 1],
#     h4=hamiltonian[1, 2],
#     h5=hamiltonian[2, 2],
#
# )
# mu, nu, n_particles = 0, 1, 2
# c_mu_matrix = utils.get_majorana(mu, n_particles)
# c_nu_matrix = utils.get_majorana(nu, n_particles)
# c_mu_nu_matrix = c_mu_matrix @ c_nu_matrix
# print(f"c_{mu} = {c_mu_matrix}, \nc_{nu} = {c_nu_matrix}, \nc_{mu}c_{nu} = {c_mu_nu_matrix}")
#
# hamiltonian_coefficients_matrix = params.to_matrix()
# n_particles = 2
# coeffs, obs = [], []
# for mu in range(2 * n_particles):
#     for nu in range(2 * n_particles):
#         if mu == nu:
#             continue
#         c_mu_str = utils.get_majorana_pauli_string(mu, n_particles, join_char='')
#         c_nu_str = utils.get_majorana_pauli_string(nu, n_particles, join_char='')
#         c_mu = qml.pauli.string_to_pauli_word(c_mu_str)
#         c_nu = qml.pauli.string_to_pauli_word(c_nu_str)
#         c_mu_matrix = utils.get_majorana(mu, n_particles)
#         c_nu_matrix = utils.get_majorana(nu, n_particles)
#         c_mu_nu_matrix = c_mu_matrix @ c_nu_matrix
#         coeffs.append(hamiltonian_coefficients_matrix[mu, nu])
#         obs.append(c_mu @ c_nu)
        # print(f"c_{mu} = {c_mu_str}, c_{nu} = {c_nu_str}")
        # print(f"c_{mu} = {c_mu_matrix}, \nc_{nu} = {c_nu_matrix}, \nc_{mu}c_{nu} = {c_mu_nu_matrix}")
        # print(f"c_{mu} = {c_mu.matrix()}")
        # print(f"c_{nu} = {c_nu.matrix()}")
        # print(f"c_{mu}c_{nu} = {(c_mu @ c_nu).matrix()}")

# H = qml.Hamiltonian(coeffs, obs)
# Hmat = H.sparse_matrix().toarray()
#
# print(f"{params = }")
# print(f"{std_h_params = }")
# print(f"{std_h_params_ = }")
# print(f"{std_h_params == std_h_params_ = }")
# print(f"{hamiltonian = }")
# print(f"{Hmat = }")

# mps.MatchgatePolarParams.RAISE_ERROR_IF_INVALID_PARAMS = False
# from_params = mps.MatchgatePolarParams(r0=1, r1=0, theta0=0, theta1=0, theta2=np.pi/2, theta3=0, theta4=np.pi / 2)
# print(f"{from_params = }")
# to_params = mps.MatchgateStandardParams.parse_from_params(from_params)
# print(f"{to_params = }")
# _from_params = mps.MatchgatePolarParams.parse_from_params(to_params)
# print(f"{_from_params = }")
# print(f"{from_params == _from_params = }")
mps.MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True
from_params = mps.MatchgatePolarParams(
    r0=0, r1=0, theta0=np.pi, theta1=np.pi/2, theta2=np.pi/3, theta3=np.pi/4, theta4=np.pi/5
)

to_params = mps.MatchgateStandardParams.parse_from_params(from_params)
reconstructed_params = mps.MatchgatePolarParams.parse_from_params(to_params)
reconstructed_to_params = mps.MatchgateStandardParams.parse_from_params(reconstructed_params)
print(f"{from_params = }")
print(f"{reconstructed_params = }")
print(f"{from_params == reconstructed_params = }")
print(f"{to_params = }")
print(f"{reconstructed_to_params = }")
print(f"{reconstructed_to_params == to_params = }")





