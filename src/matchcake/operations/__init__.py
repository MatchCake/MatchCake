

from .matchgate_operation import (
    MatchgateOperation
)


from .fermionic_rotations import (
    FermionicRotation,
    FermionicRotationXX,
    FermionicRotationYY,
    FermionicRotationZZ,
    fRXX,
    fRYY,
    fRZZ,
)

from .angle_embedding import (
    MAngleEmbedding,
    MAngleEmbeddings,
)

from .fermionic_controlled_not import (
    FermionicCNOT,
    fCNOT,
)

from .fermionic_hadamard import (
    FermionicHadamard,
    fH,
)

from .fermionic_controlled_z import (
    FermionicControlledZ,
    fCZ,
)

from .fermionic_swap import (
    FermionicSWAP,
    fSWAP,
)

from .fermionic_superposition import (
    FermionicSuperposition,
)

from .. import matchgate_parameter_sets


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
