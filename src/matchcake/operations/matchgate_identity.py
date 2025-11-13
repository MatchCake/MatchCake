from typing import Optional

import torch

from .matchgate_operation import MatchgateOperation


class MatchgateIdentity(MatchgateOperation):
    def __new__(
            cls,
            wires=None,
            dtype: torch.dtype = torch.complex128,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        return cls.from_std_params(
            a=1, w=1, z=1, d=1,
            dtype=dtype,
            device=device,
            **kwargs
        )
