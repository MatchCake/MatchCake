from typing import Optional

import torch


class MatchgateParams:
    def matrix(self, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError()
