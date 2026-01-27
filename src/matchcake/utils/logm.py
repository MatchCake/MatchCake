from typing import Callable

import pennylane as qml
import torch
from scipy.linalg import logm as scipy_logm

from ..typing import TensorLike


class TorchLogm(torch.autograd.Function):
    """
    Computes the matrix logarithm of input tensors and supports the computation of gradients
    through the operation.

    This class implements a custom torch.autograd.Function to compute the
    matrix logarithm for square matrices or batches of square matrices. The
    forward and backward passes are explicitly defined, utilizing functions
    provided by SciPy for numerical stability. This implementation supports real
    and complex data types.

    See: https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _torch_logm_scipy(tensor: torch.Tensor):
        if tensor.ndim == 2:
            return torch.from_numpy(scipy_logm(tensor.cpu(), disp=False)[0]).to(tensor.device)
        return torch.stack([TorchLogm._torch_logm_scipy(mat) for mat in tensor]).to(tensor.device)

    @staticmethod
    def _torch_adjoint(inputs: torch.Tensor, grads: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]):
        A_H = inputs.mH.to(grads.dtype)
        n = inputs.size(-2)
        shape = [*inputs.shape[:-2], 2 * n, 2 * n]
        M = torch.zeros(*shape, dtype=grads.dtype, device=grads.device)
        M[..., :n, :n] = A_H
        M[..., n:, n:] = A_H
        M[..., :n, n:] = grads
        return f(M)[..., :n, n:].to(inputs.dtype)

    @staticmethod
    def forward(ctx, tensor):
        assert tensor.ndim in (2, 3) and tensor.size(-2) == tensor.size(-1)  # Square matrix, maybe batched
        assert tensor.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(tensor)
        return TorchLogm._torch_logm_scipy(tensor)

    @staticmethod
    def backward(ctx, grads):
        (inputs,) = ctx.saved_tensors  # pragma: no cover
        return TorchLogm._torch_adjoint(inputs, grads, TorchLogm._torch_logm_scipy)  # pragma: no cover


torch_logm = TorchLogm.apply


@qml.math.multi_dispatch(argnum=0, tensor_list=0)
def logm(tensor: TensorLike, like=None):
    """Compute the matrix exponential of an array :math:`\\ln{X}`.

    .. note::
        This function is not differentiable with Autograd, as it
        relies on the scipy implementation.
    """
    if like == "torch":
        return torch_logm(tensor)
    if like in ["jax", "tensorflow"]:  # pragma: no cover
        return qml.math.logm(tensor)  # pragma: no cover

    as_arr = qml.math.array(tensor, dtype=complex)
    tensor_shape = qml.math.shape(as_arr)
    batched_tensor = qml.math.reshape(as_arr, (-1, *tensor_shape[-2:]))
    stacked_tensor = qml.math.stack([scipy_logm(m) for m in batched_tensor], axis=0)
    return qml.math.reshape(stacked_tensor, tensor_shape)
