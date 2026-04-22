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
    epsilon: float = 1e-32,
) -> TensorLike:
    """
    Compute the Pfaffian of a skew-symmetric matrix using determinant-based methods.

    This function calculates the Pfaffian value of a skew-symmetric matrix (or a batch
    of such matrices) using the determinant of the matrix. If the matrix is of odd
    dimensions, the result is zero because skew-symmetric matrices of odd size
    always have a zero Pfaffian. The computation is optimized for different backends
    like "autograd", "numpy", and others.

    :param __matrix: Input skew-symmetric matrix or batch of skew-symmetric matrices.
    :param p_bar: Optional progress bar instance for updating progress.
    :param show_progress: Whether to display progress during computation. Defaults
        to False.
    :param epsilon: Minimum value to clip the determinant during the computation of
        the Pfaffian to avoid numerical instabilities. Defaults to 1e-32.
    :return: The computed Pfaffian value(s) of the input skew-symmetric matrix or batch.
    :rtype: TensorLike
    """
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
        pf = qml.math.sqrt(qml.math.clip(qml.math.abs(det), min=epsilon))
    else:
        det = qml.math.det(__matrix)
        pf = qml.math.sqrt(qml.math.clamp(qml.math.abs(det), min=epsilon))
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian_by_det_cuda(
    __matrix: TensorLike,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
    epsilon: float = 1e-32,
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

    det_backend = qml.math.get_interface(det)
    if det_backend in ["autograd", "numpy"]:
        pf = qml.math.sqrt(qml.math.clip(qml.math.abs(det), min=epsilon))
    else:
        pf = qml.math.sqrt(qml.math.clamp(qml.math.abs(det), min=epsilon))

    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian(
    __matrix: TensorLike,
    method: Literal["det", "cuda_det", "PfaffianFDBPf"] = "det",
    epsilon: float = 1e-32,
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
        torch_pfaffian.PfaffianStrategy.EPSILON = epsilon
        pf = torch_pfaffian.get_pfaffian_function(method)(torch_utils.to_tensor(__matrix, dtype=torch.complex128))
        return convert_and_cast_like(pf, __matrix)
    raise ValueError(f"Invalid method. Got {method}, must be 'det', 'cuda_det', or 'PfaffianFDBPf'.")
