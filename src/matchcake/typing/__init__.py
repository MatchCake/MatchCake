from typing import Union
from pennylane.typing import TensorLike as PennylaneTensorLike
import torch


TensorLike = Union[PennylaneTensorLike, torch.Tensor]