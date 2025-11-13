from pennylane.wires import Wires

from .single_particle_transition_matrices.single_particle_transition_matrix import SingleParticleTransitionMatrixOperation
from .single_particle_transition_matrices.sptm_comp_hh import SptmCompHH
from .matchgate_operation import MatchgateOperation
from ..utils.constants import CLIFFORD_H


class CompHH(MatchgateOperation):
    r"""
    Represents a matchgate composition of two hadamard gates

    .. math::
        U = M(H, H)

    where :math:`M` is a matchgate, :math:`H` is the hadamard gate.
    It's also called the fermionic hadamard.
    """
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __new__(cls, wires=None, id=None, **kwargs):
        return cls.from_sub_matrices(
            CLIFFORD_H, CLIFFORD_H,
            wires=wires,
            id=id,
            **kwargs
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name

    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SptmCompHH(wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)
