import msim
from msim import matchgate_parameter_sets as mps
import numpy as np
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
matchgate_params = mps.MatchgatePolarParams(params)
# matchgate_params = mps.MatchgatePolarParams(r0=74, r1=[0, 1, 2])
print(matchgate_params)
m = msim.Matchgate(matchgate_params)
print(m)
