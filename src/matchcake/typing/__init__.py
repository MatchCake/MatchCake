from numbers import Number
from typing import Union

import torch
from pennylane.typing import TensorLike as PennylaneTensorLike

TensorLike = Union[PennylaneTensorLike, torch.Tensor, Number]
