import numpy as np

from .. import utils
from . import transfer_functions
from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import (
    MatchgateHamiltonianCoefficientsParams,
)
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams

Identity = MatchgateStandardParams(
    a=1,
    b=0,
    c=0,
    d=1,
    w=1,
    x=0,
    y=0,
    z=1,
)
fSWAP = MatchgateStandardParams(
    a=utils.PAULI_Z[0, 0],
    b=utils.PAULI_Z[0, 1],
    c=utils.PAULI_Z[1, 0],
    d=utils.PAULI_Z[1, 1],
    w=utils.PAULI_X[0, 0],
    x=utils.PAULI_X[0, 1],
    y=utils.PAULI_X[1, 0],
    z=utils.PAULI_X[1, 1],
)
HellParams = MatchgatePolarParams(
    r0=0.5,
    r1=0.1,
    theta0=np.pi,
    theta1=np.pi / 2,
    theta2=np.pi / 3,
    theta3=np.pi / 4,
    theta4=np.pi / 5,
)
_inv_sqrt_2 = 1 / np.sqrt(2)
fHH = MatchgateStandardParams(
    a=_inv_sqrt_2,
    b=_inv_sqrt_2,
    c=_inv_sqrt_2,
    d=-_inv_sqrt_2,
    w=_inv_sqrt_2,
    x=_inv_sqrt_2,
    y=_inv_sqrt_2,
    z=-_inv_sqrt_2,
)
ZI = MatchgateStandardParams(
    a=1,
    b=0,
    c=0,
    d=-1,
    w=1,
    x=0,
    y=0,
    z=-1,
)
IZ = MatchgateStandardParams(
    a=1,
    b=0,
    c=0,
    d=-1,
    w=-1,
    x=0,
    y=0,
    z=1,
)
