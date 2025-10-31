from .. import matchgate_parameter_sets
from .angle_embedding import MAngleEmbedding, MAngleEmbeddings
from .comp_hh import CompHH
from .comp_rotations import (
    CompRotation,
    CompRxRx,
    CompRyRy,
    CompRzRz,
)
from .fermionic_controlled_z import FermionicControlledZ, fCZ
from .fermionic_superposition import FermionicSuperposition
from .fermionic_swap import CompZX, fSWAP, fswap_chain, fswap_chain_gen
from .matchgate_operation import MatchgateOperation
from .rxx import Rxx
from .rzz import Rzz
from .single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmAngleEmbedding,
    SptmCompHH,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
    SptmFermionicSuperposition,
    SptmFSwapCompRzRz,
    SptmIdentity,
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
