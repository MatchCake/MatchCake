from typing import Optional

import torch
from pennylane.wires import Wires

from .. import matchgate_parameter_sets as mgp
from ..utils.constants import CLIFFORD_H
from .matchgate_operation import MatchgateOperation
from .single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from .single_particle_transition_matrices.sptm_comp_hh import SptmCompHH


class CompHH(MatchgateOperation):
    r"""
    Represents a matchgate composition of two hadamard gates

    .. math::
        U = M(H, H)

    where :math:`M` is a matchgate, :math:`H` is the hadamard gate.
    It's also called the fermionic hadamard.
    """

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(
        self,
        wires=None,
        id=None,
        default_dtype: torch.dtype = torch.complex128,
        default_device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(
            mgp.MatchgateStandardParams.from_sub_matrices(CLIFFORD_H, CLIFFORD_H),
            wires=wires,
            id=id,
            default_dtype=default_dtype,
            default_device=default_device,
            **kwargs,
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name

    #
    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SptmCompHH(wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)
