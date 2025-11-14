import numbers
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.exceptions import ComplexWarning

DTYPES_TO_TORCH_TYPES = {
    # Natives
    float: torch.float64,
    int: torch.long,
    complex: torch.complex128,
    bool: torch.bool,
    # Numpy
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.long,
    np.bool: torch.bool,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def to_tensor(
    x: Any, dtype: Optional[torch.dtype] = torch.float64, device: Optional[torch.device] = None
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    dtype = DTYPES_TO_TORCH_TYPES.get(dtype, dtype)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, numbers.Number):
            return torch.tensor(x, dtype=dtype, device=device)
        elif isinstance(x, dict):
            return {k: to_tensor(v, dtype=dtype, device=device) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)([to_tensor(v, dtype=dtype, device=device) for v in x])
        elif not isinstance(x, torch.Tensor):
            try:
                return torch.tensor(x, dtype=dtype, device=device)
            except Exception as e:
                raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")


def to_cuda(x: Any, dtype=torch.float64):
    return to_tensor(x, dtype=dtype, device=torch.device("cuda"))


def to_cpu(x: Any, dtype=torch.float64):
    return to_tensor(x, dtype=dtype, device=torch.device("cpu"))


def to_numpy(x: Any, dtype=np.float64):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ComplexWarning)
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=dtype)
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(dtype)
        elif isinstance(x, numbers.Number):
            return x
        elif isinstance(x, dict):
            return {k: to_numpy(v, dtype=dtype) for k, v in x.items()}
        elif not isinstance(x, torch.Tensor):
            try:
                return np.asarray(x, dtype=dtype)
            except Exception as e:
                raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")


def detach(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)([detach(v) for v in x])
    return x


def torch_wrap_circular_bounds(tensor, lower_bound: float = 0.0, upper_bound: float = 1.0):
    return (tensor - lower_bound) % (upper_bound - lower_bound) + lower_bound
