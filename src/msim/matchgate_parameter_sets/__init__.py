from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import MatchgateHamiltonianCoefficientsParams
from .matchgate_params import MatchgateParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams
from .matchgate_polar_params import MatchgatePolarParams
from . import transfer_functions
from .. import utils

Identity = MatchgateStandardParams(
    a=1, b=0, c=0, d=1,
    w=1, x=0, y=0, z=1,
)
fSWAP = MatchgateStandardParams(
    a=utils.PAULI_Z[0, 0], b=utils.PAULI_Z[0, 1], c=utils.PAULI_Z[1, 0], d=utils.PAULI_Z[1, 1],
    w=utils.PAULI_X[0, 0], x=utils.PAULI_X[0, 1], y=utils.PAULI_X[1, 0], z=utils.PAULI_X[1, 1]
)
