from msim import utils, MatchgateStandardHamiltonianParams, MatchgateHamiltonianCoefficientsParams, \
    MatchgateStandardParams
from msim import matchgate_parameter_sets as mps
import numpy as np

params = MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0)
std_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)
hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(std_h_params.to_matrix())
elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
std_h_params_ = MatchgateStandardHamiltonianParams.from_numpy(params_arr)

print(f"{params = }")
print(f"{std_h_params = }")
print(f"{std_h_params_ = }")
print(f"{hamiltonian = }")







