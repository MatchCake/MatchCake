import msim
from msim import matchgate_parameter_sets as mps
import numpy as np
import pennylane as qml
import faulthandler

faulthandler.enable()

b = 3
params = np.random.rand(b, mps.MatchgatePolarParams.N_PARAMS)
# params = np.random.rand(mps.MatchgatePolarParams.N_PARAMS)
# r0, r1, theta0, theta1, theta2, theta3 = tuple(params)
# matchgate_params = mps.MatchgatePolarParams(
#     r0=r0,
#     r1=r1,
#     theta0=theta0,
#     theta1=theta1,
#     theta2=theta2,
#     theta3=theta3,
# )
# matchgate_params = mps.MatchgatePolarParams(params)
# matchgate_params = mps.MatchgatePolarParams(r0=74, r1=[0, 1, 2])
matchgate_params = mps.MatchgatePolarParams(
    r0=np.random.rand(),
    r1=np.random.rand(),
    theta0=np.random.rand(),
    theta1=np.random.rand(),
    theta2=np.random.rand(),
    theta3=np.random.rand(),
)
print(matchgate_params)
matchgate = msim.Matchgate(matchgate_params)
print(matchgate)

print(matchgate.hamiltonian_coefficients_params.to_matrix())

gate_det = np.linalg.det(matchgate.gate_data)
hamiltonian_form_det = np.linalg.det(qml.math.expm(1j * matchgate.hamiltonian_matrix))
hamiltonian_trace = qml.math.trace(matchgate.hamiltonian_matrix, axis1=-2, axis2=-1)
exp_trace = qml.math.exp(1j * hamiltonian_trace)
np.testing.assert_almost_equal(hamiltonian_form_det, gate_det)
np.testing.assert_almost_equal(gate_det, exp_trace)

mg = msim.Matchgate(mps.MatchgateStandardParams(a=[1, 1], w=1, z=1, d=1))
print(mg.single_transition_particle_matrix)
np.testing.assert_allclose(mg.single_transition_particle_matrix, qml.math.stack([np.eye(4) for _ in range(2)]))
