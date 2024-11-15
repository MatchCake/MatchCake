import numbers
from typing import Any

import numpy as np
import torch


def to_tensor(x: Any, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    elif isinstance(x, torch.Tensor):
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


def to_cuda(x: Any, dtype=torch.float32):
    return to_tensor(x, dtype=dtype).cuda()


def to_cpu(x: Any, dtype=torch.float32):
    return to_tensor(x, dtype=dtype).cpu()


def to_numpy(x: Any, dtype=np.float32):
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
