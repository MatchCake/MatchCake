from typing import Any, List, Optional, Sequence, Tuple

import torch


class ReversibleColumnProduct(torch.autograd.Function):
    r"""
    Product of a matrix with a sequence of orthogonal blocks acting on a few of its columns, with a
    backward pass that stores no intermediate matrix.

    Each step multiplies the columns :math:`I` of the matrix by a stack of blocks,

    .. math::
        R_{:, I} \leftarrow R_{:, I} \, G,

    which is the product of :math:`R` with :math:`G` padded with the identity. Plain autograd keeps
    what it needs to differentiate every step and copies the whole gradient of :math:`R` at each one,
    so its backward pass costs :math:`O(n^2)` per step in time and memory. Here the blocks are
    orthogonal, so the backward pass undoes the steps instead: starting from the final matrix and its
    gradient :math:`\bar{R}`, for each step in reverse order,

    .. math::
        R_{:, I} \leftarrow R_{:, I} \, G^{-1}, \qquad
        \bar{G} = R_{:, I}^\top \, \bar{R}_{:, I}, \qquad
        \bar{R}_{:, I} \leftarrow \bar{R}_{:, I} \, G^\top.

    Only the columns of the step are read and written, so the backward pass costs :math:`O(n)` per
    step, and two matrices are held whatever the number of steps. An orthogonal map does not amplify
    rounding errors, so undoing the steps is stable. The inverse of each block is used, which is its
    transpose when it is orthogonal, so that the gradients are exact for any invertible block and the
    second derivative is exact in every direction, not only along the orthogonal group. The blocks
    must be close to orthogonal for the matrix to be recovered accurately: the caller checks it.
    """

    @staticmethod
    def multiply(columns: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """Multiply gathered columns by their blocks.

        :param columns: Gathered columns, shape (..., rows, n_blocks, size).
        :type columns: torch.Tensor
        :param blocks: Blocks, shape (n_blocks, size, size) or (batch, n_blocks, size, size).
        :type blocks: torch.Tensor
        :return: The products, shape (..., rows, n_blocks, size).
        :rtype: torch.Tensor
        """
        # (..., n_blocks, rows, size) @ (..., n_blocks, size, size), with matmul so that vmap can batch it
        return (columns.transpose(-3, -2) @ blocks).transpose(-3, -2)

    @staticmethod
    def expanded(matrix: torch.Tensor, blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return a copy of the matrix with the batch dimension its product with the blocks has.

        :param matrix: The matrix, shape (rows, columns) or (batch, rows, columns).
        :type matrix: torch.Tensor
        :param blocks: The stacks of blocks, shape (n_blocks, size, size) or (batch, n_blocks, size, size).
        :type blocks: Sequence[torch.Tensor]
        :return: A new tensor, shape (rows, columns) or (batch, rows, columns).
        :rtype: torch.Tensor
        """
        batch_size = max([stack.shape[0] for stack in blocks if stack.ndim == 4], default=0)
        if batch_size > 0 and (matrix.ndim == 2 or (matrix.shape[0] == 1 and batch_size > 1)):
            return matrix.expand(batch_size, *matrix.shape[-2:]).clone()
        return matrix.clone()

    @staticmethod
    def forward(ctx: Any, indices: Sequence[torch.Tensor], matrix: torch.Tensor, *blocks: torch.Tensor) -> torch.Tensor:
        """
        :param ctx: The autograd context.
        :param indices: For each step, the columns each block acts on, shape (n_blocks, size). The
            blocks of one step act on disjoint columns.
        :type indices: Sequence[torch.Tensor]
        :param matrix: The matrix, shape (rows, columns) or (batch, rows, columns). It is not modified.
        :type matrix: torch.Tensor
        :param blocks: For each step, the orthogonal blocks, shape (n_blocks, size, size) or
            (batch, n_blocks, size, size).
        :type blocks: torch.Tensor
        :return: The product, shape (rows, columns) or (batch, rows, columns).
        :rtype: torch.Tensor
        """
        result = ReversibleColumnProduct.expanded(matrix, blocks)
        for step_indices, step_blocks in zip(indices, blocks):
            columns = result[..., :, step_indices]
            result[..., :, step_indices] = ReversibleColumnProduct.multiply(columns, step_blocks)
        ctx.indices = indices
        ctx.save_for_backward(matrix, *blocks, result)
        ctx.save_for_forward(matrix, *blocks)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        :param ctx: The autograd context.
        :param grad_output: The gradient of the product, shape (rows, columns) or (batch, rows, columns).
        :type grad_output: torch.Tensor
        :return: None for the indices, then the gradient of the matrix and of each stack of blocks,
            None for the ones that need no gradient.
        :rtype: Tuple[Optional[torch.Tensor], ...]
        """
        matrix, *blocks, result = ctx.saved_tensors
        state = result.clone()
        gradient = grad_output.clone()
        block_gradients: List[Optional[torch.Tensor]] = [None] * len(blocks)
        for step in reversed(range(len(blocks))):
            step_indices, step_blocks = ctx.indices[step], blocks[step]
            # The inverse, not the transpose, undoes the step: the two agree for an orthogonal block, but only
            # the inverse makes the recovered columns independent of the block, which the second derivative needs.
            columns = ReversibleColumnProduct.multiply(state[..., :, step_indices], torch.linalg.inv(step_blocks))
            gradient_columns = gradient[..., :, step_indices]
            if ctx.needs_input_grad[2 + step]:
                # (..., n_blocks, size, rows) @ (..., n_blocks, rows, size)
                block_gradient = columns.permute(*range(columns.ndim - 3), -2, -1, -3) @ gradient_columns.transpose(
                    -3, -2
                )
                if step_blocks.ndim == 3 and columns.ndim == 4:
                    block_gradient = block_gradient.sum(dim=0)
                block_gradients[step] = block_gradient
            gradient[..., :, step_indices] = ReversibleColumnProduct.multiply(gradient_columns, step_blocks.mT)
            state[..., :, step_indices] = columns
        matrix_gradient = None
        if ctx.needs_input_grad[1]:
            matrix_gradient = gradient
            if matrix.ndim == 2 and gradient.ndim == 3:
                matrix_gradient = gradient.sum(dim=0)
            elif matrix.ndim == 3 and matrix.shape[0] != gradient.shape[0]:
                matrix_gradient = gradient.sum(dim=0, keepdim=True)
        return (None, matrix_gradient, *block_gradients)

    @staticmethod
    def jvp(ctx: Any, _: Any, matrix_tangent: torch.Tensor, *block_tangents: torch.Tensor) -> torch.Tensor:
        """
        :param ctx: The autograd context.
        :param _: The tangent of the indices, which have none.
        :param matrix_tangent: The tangent of the matrix.
        :type matrix_tangent: torch.Tensor
        :param block_tangents: The tangent of each stack of blocks.
        :type block_tangents: torch.Tensor
        :return: The tangent of the product, shape (rows, columns) or (batch, rows, columns).
        :rtype: torch.Tensor
        """
        matrix, *blocks = ctx.saved_tensors
        state = ReversibleColumnProduct.expanded(matrix, blocks)
        # autograd hands a tangent of zeros to an input that has none
        tangent = matrix_tangent.expand_as(state).clone()
        for step_indices, step_blocks, step_tangent in zip(ctx.indices, blocks, block_tangents):
            columns = state[..., :, step_indices]
            tangent[..., :, step_indices] = ReversibleColumnProduct.multiply(
                tangent[..., :, step_indices], step_blocks
            ) + ReversibleColumnProduct.multiply(columns, step_tangent)
            state[..., :, step_indices] = ReversibleColumnProduct.multiply(columns, step_blocks)
        return tangent
