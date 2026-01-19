import pennylane as qml
import scipy
import torch

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
    """

    @staticmethod
    def _torch_logm_scipy(tensor: TensorLike):
        if tensor.ndim == 2:
            return torch.from_numpy(scipy.linalg.logm(tensor.cpu(), disp=False)[0]).to(tensor.device)
        return torch.stack([torch.from_numpy(scipy.linalg.logm(A_.cpu(), disp=False)[0]) for A_ in tensor.cpu()]).to(
            tensor.device
        )

    @staticmethod
    def _torch_adjoint(tensor0, tensor1, f):
        A_H = tensor0.T.conj().to(tensor1.dtype)
        n = tensor0.size(0)
        M = torch.zeros(2 * n, 2 * n, dtype=tensor1.dtype, device=tensor1.device)
        M[:n, :n] = A_H
        M[n:, n:] = A_H
        M[:n, n:] = tensor1
        return f(M)[:n, n:].to(tensor0.dtype)

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
        (inputs,) = ctx.saved_tensors
        if inputs.ndim == 2:
            return TorchLogm._torch_adjoint(inputs, grads, TorchLogm._torch_logm_scipy)
        return torch.stack(
            [TorchLogm._torch_adjoint(A_, G_, TorchLogm._torch_logm_scipy) for A_, G_ in zip(inputs, grads)]
        )


torch_logm = TorchLogm.apply


@qml.math.multi_dispatch()
def logm(tensor: TensorLike, like=None):
    """Compute the matrix exponential of an array :math:`\\ln{X}`.

    .. note::
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
