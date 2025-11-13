from typing import Union
from numbers import Number
from pennylane.typing import TensorLike as PennylaneTensorLike
import torch


TensorLike = Union[PennylaneTensorLike, torch.Tensor, Number]