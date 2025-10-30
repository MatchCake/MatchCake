from typing import Literal, Optional

import numpy as np
import pennylane as qml
import torch
import torch_pfaffian
import tqdm
from pennylane.typing import TensorLike

from . import torch_utils
from .math import convert_and_cast_like


def pfaffian_by_det(
    __matrix: TensorLike,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
    epsilon: float = 1e-12,
) -> TensorLike:
    shape = qml.math.shape(__matrix)
    p_bar = p_bar or tqdm.tqdm(total=1, disable=not show_progress)
    p_bar.set_description(f"[det] Computing determinant of {shape} matrix")

    # Quick return if possible
    if shape[-2] % 2 == 1:
        p_bar.set_description("Odd-sized matrix")
        p_bar.close()
        matrix = qml.math.cast(__matrix, dtype=complex)
        zero_like = convert_and_cast_like(0, matrix)
        pfaffian_val = qml.math.convert_like(np.ones(shape[:-2], dtype=complex), matrix)
        return pfaffian_val * zero_like * matrix[..., 0, 0]

    backend = qml.math.get_interface(__matrix)
    if backend in ["autograd", "numpy"]:
        det = qml.math.linalg.det(__matrix)
        pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    else:
        det = qml.math.det(__matrix)
        pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian_by_det_cuda(
    __matrix: TensorLike,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
    epsilon: float = 1e-12,
) -> TensorLike:
    shape = qml.math.shape(__matrix)
    p_bar = p_bar or tqdm.tqdm(total=1, disable=not show_progress)
    p_bar.set_description(f"[cuda_det] Computing determinant of {shape} matrix")
    backend = qml.math.get_interface(__matrix)
    cuda_det = torch.det(torch_utils.to_cuda(__matrix))
    if backend in ["torch"]:
        if __matrix.device.type == "cuda":
            det = cuda_det
        else:
            det = torch_utils.to_cpu(cuda_det)
    else:
        det = torch_utils.to_cpu(cuda_det)
    det = convert_and_cast_like(det, __matrix)
    pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian(
    __matrix: TensorLike,
    method: Literal["det", "cuda_det", "PfaffianFDBPf"] = "det",
    epsilon: float = 1e-12,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
) -> TensorLike:
    """
    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T).

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: TensorLike
    :param method: Method to use to compute the pfaffian.
    :type method: Literal["det", "cuda_det", "PfaffianFDBPf"]
    :param epsilon: Tolerance for the determinant method
    :type epsilon: float
    :param p_bar: Progress bar
    :type p_bar: Optional[tqdm.tqdm]
    :param show_progress: Whether to show progress bar. If no progress bar is provided, a new one is created
    :type show_progress: bool
    :return: Pfaffian of the matrix
    :rtype: TensorLike
    """
    shape = qml.math.shape(__matrix)
    assert shape[-2] == shape[-1] > 0, "Matrix must be square"

    if method == "det":
        return pfaffian_by_det(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "cuda_det":
        return pfaffian_by_det_cuda(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "PfaffianFDBPf":
        pf = torch_pfaffian.get_pfaffian_function(method)(torch_utils.to_tensor(__matrix, dtype=torch.complex128))
        return convert_and_cast_like(pf, __matrix)
    raise ValueError(f"Invalid method. Got {method}, must be 'det', 'cuda_det', or 'PfaffianFDBPf'.")
