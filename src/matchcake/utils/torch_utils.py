import numbers
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

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


def get_torch_dtype(dtype: Any, default: torch.dtype) -> torch.dtype:
    if dtype is None:
        return default
    return DTYPES_TO_TORCH_TYPES.get(dtype, dtype)  # type: ignore[arg-type, return-value]


COMPLEX_TO_REAL_DTYPE = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
REAL_TO_COMPLEX_DTYPE = {real: complex_ for complex_, real in COMPLEX_TO_REAL_DTYPE.items()}


def _torch_dtype_of(tensor: Any) -> torch.dtype:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    return torch.as_tensor(np.asarray(tensor)).dtype


def infer_real_dtype(tensor: Any, default: torch.dtype = torch.float64) -> torch.dtype:
    """
    Infer the real-valued ``torch`` dtype matching the precision of ``tensor``.

    Complex inputs map to their real counterpart (e.g. ``complex128 -> float64``),
    floating inputs keep their precision, and non-floating inputs fall back to
    ``default``.

    :param tensor: Tensor (torch, numpy, ...) whose precision is inspected.
    :param default: Dtype returned for non-floating inputs.
    :return: A real ``torch.dtype``.
    """
    dtype = _torch_dtype_of(tensor)
    if dtype in COMPLEX_TO_REAL_DTYPE:
        return COMPLEX_TO_REAL_DTYPE[dtype]
    if dtype.is_floating_point:
        return dtype
    return default


def infer_complex_dtype(tensor: Any, default: torch.dtype = torch.complex128) -> torch.dtype:
    """
    Infer the complex ``torch`` dtype matching the precision of ``tensor``.

    Complex inputs keep their precision, floating inputs map to their complex
    counterpart (e.g. ``float32 -> complex64``), and other inputs fall back to
    ``default``.

    :param tensor: Tensor (torch, numpy, ...) whose precision is inspected.
    :param default: Dtype returned for non-floating/non-complex inputs.
    :return: A complex ``torch.dtype``.
    """
    dtype = _torch_dtype_of(tensor)
    if dtype in COMPLEX_TO_REAL_DTYPE:
        return dtype
    if dtype in REAL_TO_COMPLEX_DTYPE:
        return REAL_TO_COMPLEX_DTYPE[dtype]
    return default


def torch_dtype_name(dtype: torch.dtype) -> str:
    """Return the bare name of a ``torch`` dtype, e.g. ``torch.float32 -> "float32"``."""
    return str(dtype).rsplit(".", 1)[-1]


def to_tensor(
    x: Any, dtype: Optional[torch.dtype] = torch.float64, device: Optional[torch.device] = None
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    dtype = DTYPES_TO_TORCH_TYPES.get(dtype, dtype)  # type: ignore[call-overload]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, numbers.Number):
            return torch.tensor(x, dtype=dtype, device=device)
        elif isinstance(x, dict):
            return cast(Dict[str, torch.Tensor], {k: to_tensor(v, dtype=dtype, device=device) for k, v in x.items()})
        elif isinstance(x, (list, tuple)):
            return type(x)([to_tensor(v, dtype=dtype, device=device) for v in x])
        elif not isinstance(x, torch.Tensor):
            try:
                return torch.tensor(x, dtype=dtype, device=device)
            except Exception as e:
                raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")  # pragma: no cover


def to_cuda(x: Any, dtype=torch.float64):  # pragma: no cover
    return to_tensor(x, dtype=dtype, device=torch.device("cuda"))  # pragma: no cover


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
    raise ValueError(f"Unsupported type {type(x)}")  # pragma: no cover


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
