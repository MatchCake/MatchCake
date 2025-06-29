from .. import matchgate_parameter_sets
from .angle_embedding import MAngleEmbedding, MAngleEmbeddings
from .fermionic_controlled_z import FermionicControlledZ, fCZ
from .fermionic_hadamard import FermionicHadamard, fH
from .fermionic_rotations import (
    FermionicRotation,
    FermionicRotationXX,
    FermionicRotationYY,
    FermionicRotationZZ,
    fRXX,
    fRYY,
    fRZZ,
)
from .fermionic_superposition import FermionicSuperposition
from .fermionic_swap import FermionicSWAP, fSWAP, fswap_chain, fswap_chain_gen
from .matchgate_operation import MatchgateOperation
from .rxx import Rxx
from .rzz import Rzz
from .single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmAngleEmbedding,
    SptmFermionicSuperposition,
    SptmFHH,
    SptmfRxRx,
    SptmFSwap,
    SptmFSwapRzRz,
    SptmIdentity,
    SptmRyRy,
    SptmRzRz,
)


class ZI(MatchgateOperation):
    num_wires = 2
    num_params = 0

    def __init__(self, wires):
        super().__init__(matchgate_parameter_sets.ZI, wires=wires)


class IZ(MatchgateOperation):
    num_wires = 2
    num_params = 0

    def __init__(self, wires):
        super().__init__(matchgate_parameter_sets.IZ, wires=wires)
