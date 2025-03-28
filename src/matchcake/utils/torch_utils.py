import numbers
from typing import Any

import numpy as np
import torch


def to_tensor(x: Any, dtype=torch.float64):
    if dtype is float:
        dtype = torch.float64
    elif dtype is int:
        dtype = torch.int64
    elif dtype is complex:
        dtype = torch.complex128
    elif dtype is bool:
        dtype = torch.bool

    if isinstance(x, np.ndarray):
        if dtype is None:
            return torch.from_numpy(x)
        return torch.from_numpy(x).type(dtype)
    elif isinstance(x, torch.Tensor):
        if dtype is None:
            return x
        return x.type(dtype)
    elif isinstance(x, numbers.Number):
        return torch.tensor(x, dtype=dtype)
    elif isinstance(x, dict):
        return {k: to_tensor(v, dtype=dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)([to_tensor(v, dtype=dtype) for v in x])
    elif not isinstance(x, torch.Tensor):
        try:
            return torch.tensor(x, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")


def to_cuda(x: Any, dtype=torch.float64):
    return to_tensor(x, dtype=dtype).cuda()


def to_cpu(x: Any, dtype=torch.float64):
    return to_tensor(x, dtype=dtype).cpu()


def to_numpy(x: Any, dtype=np.float64):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
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
