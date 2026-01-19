from typing import Any, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pennylane as qml

from ..constants import (
    _CIRCUIT_MATMUL_DIRECTION,
    _FOP_MATMUL_DIRECTION,
    MatmulDirectionType,
)
from ..typing import TensorLike


def shape(tensor):
    """Get the shape of a tensor.

    :param tensor: The tensor.
    :type tensor: Any
    :return: The shape of the tensor.
    :rtype: tuple
    """
    _shape = []
    if isinstance(tensor, (list, tuple)) and len(tensor) > 0:
        _shape.append(len(tensor))
        _shape.extend(qml.math.shape(tensor[0]))
    else:
        _shape.extend(qml.math.shape(tensor))
    return tuple(_shape)


def convert_and_cast_like(tensor1, tensor2):
    r"""
    Convert and cast the tensor1 to the same type as tensor2.

    :param tensor1: Tensor to convert and cast.
    :type tensor1: Any
    :param tensor2: Tensor to use as a reference.
    :type tensor2: Any

    :return: Converted and casted tensor1.
    """
    import warnings

    new_tensor1 = qml.math.convert_like(tensor1, tensor2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if not qml.math.any(qml.math.iscomplex(new_tensor1)):
            new_tensor1 = qml.math.real(new_tensor1)
        if (
            "complex" in qml.math.get_dtype_name(new_tensor1).lower()
            and not "complex" in qml.math.get_dtype_name(tensor2).lower()
        ):
            new_tensor1 = qml.math.real(new_tensor1)
        try:
            new_tensor1 = qml.math.cast_like(new_tensor1, tensor2)
        except TypeError:
            new_tensor1 = qml.math.cast_like(new_tensor1, tensor2)
    return new_tensor1


def convert_like_and_cast_to(tensor, like, dtype=None):
    r"""
    Convert and cast the tensor to the same type as the tensor like.

    :param tensor: Tensor to convert and cast.
    :type tensor: Any
    :param like: Tensor to use as a reference.
    :type like: Any
    :param dtype: Data type to cast the tensor.
    :type dtype: Any

    :return: Converted and casted tensor.
    """
    new_tensor = qml.math.convert_like(tensor, like)
    if dtype is not None:
        new_tensor = qml.math.cast(new_tensor, dtype)
    return new_tensor


def eye_block_matrix(matrix: TensorLike, n: int, index: int):
    r"""

    Take a matrix and insert it into a bigger eye matrix like this:

    .. math::

        \begin{pmatrix}
            I & 0 & 0 \\
            0 & M & 0 \\
            0 & 0 & I
        \end{pmatrix}

    where :math:`I` is the identity matrix and :math:`M` is the input matrix.

    :param matrix:
    :param n:
    :param index:
    :return:
    """
    eye = qml.math.eye(n - qml.math.shape(matrix)[0], like=matrix)
    return qml.math.block_diag([eye[:index, :index], matrix, eye[index:, index:]])


def convert_tensors_to_same_type_and_cast_to(
    tensors: List[TensorLike],
    cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ),
    dtype=None,
) -> List[TensorLike]:
    r"""
    Convert the tensors to the same type using the given priorities.

    :param tensors: Tensors to convert and cast.
    :type tensors: List[TensorLike]
    :param cast_priorities: Priorities of the casting. Higher the index is, higher the priority.
    :type cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]]

    :return: Converted and casted tensors.
    :rtype: List[TensorLike]
    """
    if len(tensors) == 0:
        return []

    tensors_priorities = [cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors]
    highest_priority = max(tensors_priorities)
    if all(priority == highest_priority for priority in tensors_priorities):
        return tensors
    like = tensors[tensors_priorities.index(highest_priority)]
    return [convert_like_and_cast_to(tensor, like, dtype) for tensor in tensors]


def convert_tensors_to_same_type(
    tensors: List[TensorLike],
    cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ),
) -> List[TensorLike]:
    r"""
    Convert the tensors to the same type using the given priorities.

    :param tensors: Tensors to convert and cast.
    :type tensors: List[TensorLike]
    :param cast_priorities: Priorities of the casting. Higher the index is, higher the priority.
    :type cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]]

    :return: Converted and casted tensors.
    :rtype: List[TensorLike]
    """
    if len(tensors) == 0:
        return []

    tensors_priorities = [cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors]
    highest_priority = max(tensors_priorities)
    if all(priority == highest_priority for priority in tensors_priorities):
        return tensors
    like = tensors[tensors_priorities.index(highest_priority)]
    return [qml.math.convert_like(tensor, like) for tensor in tensors]


def convert_and_cast_tensor_from_tensors(
    tensor: TensorLike,
    tensors: List[TensorLike],
    cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ),
) -> TensorLike:
    r"""
    Convert and cast the tensor to the same type as the tensors using the given priorities.

    :param tensor: Tensor to convert and cast.
    :type tensor: TensorLike
    :param tensors: Tensors to use as a reference.
    :type tensors: List[TensorLike]
    :param cast_priorities: Priorities of the casting. Higher the index is, higher the priority.
    :type cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]]

    :return: Converted and casted tensor.
    :rtype: TensorLike
    """
    if len(tensors) == 0:
        return tensor

    tensor_priority = cast_priorities.index(qml.math.get_interface(tensor))
    tensors_priorities = [cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors]
    highest_priority = max(tensors_priorities)
    if tensor_priority == highest_priority:
        return tensor
    # like is the first tensor with the highest priority
    like = tensors[tensors_priorities.index(highest_priority)]
    return convert_and_cast_like(tensor, like)


def exp_taylor_series(x: Any, terms: int = 18) -> Any:
    r"""
    Compute the matrix exponential using the Taylor series.

    :param x: input of the exponential.
    :type x: Any
    :param terms: Number of terms in the Taylor series.
    :type terms: int

    :return: The exponential of the input.
    :rtype: Any
    """
    results = [1]
    for i in range(1, terms + 1):
        results.append(results[-1] * x / i)
    return sum(results)


def random_index(
    probs,
    n: Optional[int] = None,
    axis=-1,
    normalize_probs: bool = True,
    eps: float = 1e-12,
):
    _n = n or 1
    axis = np.mod(axis, probs.ndim)
    if normalize_probs:
        probs = probs / (probs.sum(axis=axis, keepdims=True) + eps)

    shape_wo_axis = list(probs.shape)
    shape_wo_axis.pop(axis)
    shape_wo_axis = [_n] + shape_wo_axis
    r = np.expand_dims(np.random.rand(*shape_wo_axis), axis=1 + axis)
    indexes = (probs.cumsum(axis=axis) > r).argmax(axis=1 + axis)
    if n is None:
        return indexes[0]
    return indexes


def convert_2d_to_1d_indexes(indexes: Iterable[Tuple[int, int]], n_rows: Optional[int] = None) -> np.ndarray:
    indexes = np.asarray(indexes)
    if n_rows is None:
        n_rows = np.max(indexes[:, 0]) + 1
    new_indexes = indexes[:, 0] * n_rows + indexes[:, 1]
    return new_indexes


def convert_1d_to_2d_indexes(indexes: Iterable[int], n_rows: Optional[int] = None) -> np.ndarray:
    indexes = np.asarray(indexes)
    if n_rows is None:
        n_rows = int(np.sqrt(len(indexes)))
    new_indexes = np.stack([indexes // n_rows, indexes % n_rows], axis=-1)
    return new_indexes


def matmul(left: Any, right: Any, operator: Literal["einsum", "matmul", "@"] = "@"):
    r"""
    Perform a matrix multiplication of two matrices.

    :param left: Left matrix.
    :type left: Any
    :param right: Right matrix.
    :type right: Any
    :param operator: Operator to use for the matrix multiplication.
        "einsum" for einsum, "matmul" for matmul, "@" for __matmul__.
    :type operator: Literal["einsum", "matmul", "@"]

    :return: Result of the matrix multiplication.
    :rtype: Any
    """
    if operator == "matmul":
        return qml.math.matmul(left, right)
    if operator == "@":
        return left @ right
    return qml.math.einsum("...ij,...jk->...ik", left, right)


def circuit_matmul(
    first_matrix: Any,
    second_matrix: Any,
    direction: MatmulDirectionType = _CIRCUIT_MATMUL_DIRECTION,
    operator: Literal["einsum", "matmul", "@"] = "@",
) -> Any:
    r"""
    Perform a matrix multiplication of two matrices with the given direction.

    :param first_matrix: First matrix.
    :type first_matrix: Any
    :param second_matrix: Second matrix.
    :type second_matrix: Any
    :param direction: Direction of the matrix multiplication. "rl" for right to left and "lr" for left to right.
        That means the result will be first_matrix @ second_matrix if direction is "rl" and second_matrix @ first_matrix
    :type direction: Literal["rl", "lr"]
    :param operator: Operator to use for the matrix multiplication.
        "einsum" for einsum, "matmul" for matmul, "@" for __matmul__.
    :type operator: Literal["einsum", "matmul", "@"]

    :return: Result of the matrix multiplication.
    :rtype: Any
    """
    left, right = MatmulDirectionType.place_ops(direction, first_matrix, second_matrix)
    return matmul(left, right, operator)


def fermionic_operator_matmul(
    first_matrix: Any,
    second_matrix: Any,
    direction: MatmulDirectionType = _FOP_MATMUL_DIRECTION,
    operator: Literal["einsum", "matmul", "@"] = "@",
):
    r"""
    Perform a matrix multiplication of two fermionic operator matrices with the given direction.

    :param first_matrix: First fermionic operator matrix.
    :type first_matrix: Any
    :param second_matrix: Second fermionic operator matrix.
    :type second_matrix: Any
    :param direction: Direction of the matrix multiplication. "rl" for right to left and "lr" for left to right.
        That means the result will be first_matrix @ second_matrix if direction is "rl" and second_matrix @ first_matrix
        if direction is "lr".
    :type direction: Literal["rl", "lr"]
    :param operator: Operator to use for the matrix multiplication.
        "einsum" for einsum, "matmul" for matmul, "@" for __matmul__.
    :type operator: Literal["einsum", "matmul", "@"]

    :return: Result of the matrix multiplication.
    :rtype: Any
    """
    left, right = MatmulDirectionType.place_ops(direction, first_matrix, second_matrix)
    return matmul(left, right, operator)


def dagger(tensor: Any) -> Any:
    r"""
    Compute the conjugate transpose of the tensor.

    :param tensor: Input tensor.
    :type tensor: Any

    :return: Conjugate transpose of the tensor.
    :rtype: Any
    """
    return qml.math.conj(qml.math.einsum("...ij->...ji", tensor))


def det(tensor: Any) -> Any:
    r"""
    Compute the determinant of the tensor.

    :param tensor: Input tensor.
    :type tensor: Any

    :return: Determinant of the tensor.
    :rtype: Any
    """
    backend = qml.math.get_interface(tensor)
    if backend in ["autograd", "numpy"]:
        return qml.math.linalg.det(tensor)
    return qml.math.det(tensor)


def svd(tensor: Any) -> Tuple[Any, Any, Any]:
    r"""
    Compute the singular value decomposition of the tensor.

    :param tensor: Input tensor.
    :type tensor: Any

    :return: Singular value decomposition of the tensor.
    :rtype: Tuple[Any, Any, Any]
    """
    backend = qml.math.get_interface(tensor)
    if backend in ["autograd", "numpy", "torch"]:
        return qml.math.linalg.svd(tensor)
    return qml.math.svd(tensor)


def orthonormalize(tensor: Any, check_if_normalize: bool = True, raises_error: bool = False) -> Any:
    r"""
    Orthonormalize the tensor.

    ..math::
        U, S, V = SVD(tensor)
        return U @ V

    :param tensor: Input tensor.
    :type tensor: Any
    :param check_if_normalize: Whether to check if the tensor is already orthonormalized.
    :type check_if_normalize: bool

    :return: Orthonormalized tensor.
    :rtype: Any
    """
    try:
        if check_if_normalize:
            if check_is_unitary(tensor):
                return tensor
        u, s, v = svd(tensor)
        # test if the tensor is already orthonormalized with the eigenvalues
        if qml.math.allclose(s**1, 1):
            return tensor
        return matmul(u, v, "einsum")
    except Exception as e:
        if raises_error:
            raise e
        return tensor


def eye_like(tensor: Any):
    eye = qml.math.zeros_like(tensor)
    tensor_shape = qml.math.shape(tensor)
    eye[..., qml.math.arange(tensor_shape[-2]), qml.math.arange(tensor_shape[-1])] = 1
    return eye


def check_is_unitary(tensor: Any):
    return qml.math.allclose(matmul(tensor, dagger(tensor)), eye_like(tensor))
