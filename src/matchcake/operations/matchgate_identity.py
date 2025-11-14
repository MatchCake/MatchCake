from typing import Optional

import torch

from .matchgate_operation import MatchgateOperation
from ..matchgate_parameter_sets.matchgate_standard_params import MatchgateStandardParams


class MatchgateIdentity(MatchgateOperation):
    def __init__(
            self,
            wires=None,
            dtype: torch.dtype = torch.complex128,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(
            MatchgateStandardParams(a=1, w=1, z=1, d=1),
            wires=wires,
            dtype=dtype,
            device=device,
            **kwargs
        )
