from typing import Any
import torch


class Pfaffian(torch.autograd.Function):
    EPSILON = 1e-10

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor):
        det = torch.det(matrix)
        pf = torch.sqrt(torch.abs(det) + Pfaffian.EPSILON)
        ctx.save_for_backward(matrix, pf)
        return pf

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
        r"""

        ..math:
            \frac{\partial \text{pf}(A)}{\partial x_i} = \frac{1}{2} \text{pf}(A) tr(A^{-1} \frac{\partial A}{\partial x_i})

        :param ctx: Context
        :param grad_output: Gradient of the output
        :return: Gradient of the input
        """
        matrix, pf = ctx.saved_tensors
        matrix_clone = matrix.detach().clone()
        pf_clone = pf.detach().clone()
        grad_outputs_clone = grad_output.clone()

        am1 = torch.pinverse(matrix_clone)
        am1_papx = am1 * grad_outputs_clone.view(-1, 1, 1)
        trace = torch.einsum('...ii->...', am1_papx)
        dpf_dx = 0.5 * pf_clone * trace
        return dpf_dx


