from typing import Tuple, List, Literal, Any
import scipy
import pennylane as qml
from ..templates.tensor_like import TensorLike
try:
    import torch
except ImportError:
    torch = None


def cast_to_complex(__inputs):
    r"""

    Cast the inputs to complex numbers.

    :param __inputs: Inputs to cast
    :return: Inputs casted to complex numbers
    """
    return type(__inputs)(qml.math.asarray(__inputs).astype(complex))


def _torch_adjoint(A, E, f):
    import torch
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def _torch_logm_scipy(A):
    import torch
    if A.ndim == 2:
        return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)
    return torch.stack([torch.from_numpy(scipy.linalg.logm(A_.cpu(), disp=False)[0]) for A_ in A.cpu()]).to(A.device)


class TorchLogm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        import torch
        assert A.ndim in (2, 3) and A.size(-2) == A.size(-1)  # Square matrix, maybe batched
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return _torch_logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        if A.ndim == 2:
            return _torch_adjoint(A, G, _torch_logm_scipy)
        return torch.stack([_torch_adjoint(A_, G_, _torch_logm_scipy) for A_, G_ in zip(A, G)])


torch_logm = TorchLogm.apply


@qml.math.multi_dispatch()
def logm(tensor, like=None):
    """Compute the matrix exponential of an array :math:`\\ln{X}`.

    ..note::
        This function is not differentiable with Autograd, as it
        relies on the scipy implementation.
    """
    if like == "torch":
        return torch_logm(tensor)
    if like == "jax":
        from jax.scipy.linalg import logm as jax_logm
        
        return jax_logm(tensor)
    if like == "tensorflow":
        import tensorflow as tf
        
        return tf.linalg.logm(tensor)
    from scipy.linalg import logm as scipy_logm
    
    as_arr = qml.math.array(tensor, dtype=complex)
    tensor_shape = qml.math.shape(as_arr)
    batched_tensor = qml.math.reshape(as_arr, (-1, *tensor_shape[-2:]))
    stacked_tensor = qml.math.stack([scipy_logm(m) for m in batched_tensor], axis=0)
    return qml.math.reshape(stacked_tensor, tensor_shape)


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

    :return: Converted and casted tensor.
    """
    import warnings
    import numpy as np
    interface1, interface2 = qml.math.get_interface(tensor1), qml.math.get_interface(tensor2)
    new_tensor1 = tensor1
    if interface1 != interface2:
        new_tensor1 = qml.math.convert_like(tensor1, tensor2)
    dtype1, dtype2 = qml.math.get_dtype_name(new_tensor1), qml.math.get_dtype_name(tensor2)
    if dtype1 != dtype2:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.ComplexWarning)
            new_tensor1 = qml.math.cast_like(new_tensor1, tensor2)
    return new_tensor1


def astensor(tensor, like=None, **kwargs):
    """Convert input to a tensor.

    :param tensor: Input tensor.
    :type tensor: Any
    :param like: The desired tensor framework to use.
    :type like: str, optional
    :param kwargs: Additional keyword arguments that are passed to the tensor framework.
    :return: The tensor.
    :rtype: Any
    """
    from ..templates.tensor_like import TensorLike
    if isinstance(tensor, TensorLike):
        return tensor
    return qml.math.array(tensor, like=like, **kwargs)


def eye_block_matrix(matrix: TensorLike, n: int, index: int):
    """

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


def get_like_tensors_of_highest_priority(
        tensors: List[TensorLike],
        cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
                "numpy", "autograd", "jax", "tf", "torch"
        )
) -> TensorLike:
    r"""
    Convert and cast the tensors to the same type using the given priorities.

    :param tensors: Tensors to convert and cast.
    :type tensors: List[TensorLike]
    :param cast_priorities: Priorities of the casting. Higher the index is, higher the priority.
    :type cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]]

    :return: Converted and casted tensors.
    :rtype: List[TensorLike]
    """
    if len(tensors) == 0:
        return None
    tensors_priorities = [cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors]
    highest_priority = max(tensors_priorities)
    like = tensors[tensors_priorities.index(highest_priority)]
    return like


def convert_and_cast_tensors_to_same_type(
        tensors: List[TensorLike],
        cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
                "numpy", "autograd", "jax", "tf", "torch"
        )
) -> List[TensorLike]:
    r"""
    Convert and cast the tensors to the same type using the given priorities.

    :param tensors: Tensors to convert and cast.
    :type tensors: List[TensorLike]
    :param cast_priorities: Priorities of the casting. Higher the index is, higher the priority.
    :type cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]]

    :return: Converted and casted tensors.
    :rtype: List[TensorLike]
    """
    if len(tensors) == 0:
        return []

    tensors_priorities = [
        cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors
    ]
    highest_priority = max(tensors_priorities)
    if all(priority == highest_priority for priority in tensors_priorities):
        return tensors
    like = tensors[tensors_priorities.index(highest_priority)]
    return [convert_and_cast_like(tensor, like) for tensor in tensors]


def convert_and_cast_tensor_from_tensors(
        tensor: TensorLike,
        tensors: List[TensorLike],
        cast_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = (
                "numpy", "autograd", "jax", "tf", "torch"
        )
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
    tensors_priorities = [
        cast_priorities.index(qml.math.get_interface(tensor)) for tensor in tensors
    ]
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


def exp_euler(x: TensorLike) -> TensorLike:
    r"""
    Compute the matrix exponential using the Euler formula.

    :math:`e^{ix} = \cos(x) + i \sin(x)`

    :param x: input of the exponential.
    :type x: TensorLike

    :return: The exponential of the input.
    :rtype: TensorLike
    """
    return qml.math.cos(x) + 1j * qml.math.sin(x)
