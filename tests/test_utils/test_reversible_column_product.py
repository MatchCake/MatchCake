from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from matchcake.utils.reversible_column_product import ReversibleColumnProduct

from ..configs import ATOL_MATRIX_COMPARISON, GRADCHECK_ATOL, GRADCHECK_EPS, RTOL_MATRIX_COMPARISON, TEST_SEED

N_COLUMNS = 12
STEPS = {
    "one_block": [[(4, 4)]],
    "two_steps_on_the_same_columns": [[(0, 4)], [(0, 4)]],
    "overlapping_steps": [[(0, 4), (6, 4)], [(2, 4), (8, 4)], [(0, 4)]],
    "different_sizes": [[(0, 2), (4, 2)], [(2, 6)], [(8, 4)]],
}


class TestReversibleColumnProduct:
    @staticmethod
    def _orthogonal(generator: torch.Generator, *shape: int) -> torch.Tensor:
        """Random orthogonal matrices, shape (*shape, size, size) with ``shape[-1] == size``."""
        matrices = torch.randn(*shape, shape[-1], dtype=torch.float64, generator=generator)
        return torch.linalg.qr(matrices).Q

    @classmethod
    def _make_inputs(
        cls, steps_name: str, matrix_batch: Optional[int], block_batch: Optional[int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        generator = torch.Generator().manual_seed(TEST_SEED)
        indices, blocks = [], []
        for step in STEPS[steps_name]:
            by_size: dict = {}
            for start, size in step:
                by_size.setdefault(size, []).append(start)
            for size, starts in by_size.items():
                indices.append(torch.tensor(starts)[:, None] + torch.arange(size)[None, :])
                shape = (len(starts), size) if block_batch is None else (block_batch, len(starts), size)
                blocks.append(cls._orthogonal(generator, *shape))
        matrix_shape = (N_COLUMNS, N_COLUMNS) if matrix_batch is None else (matrix_batch, N_COLUMNS, N_COLUMNS)
        matrix = torch.randn(*matrix_shape, dtype=torch.float64, generator=generator)
        return indices, matrix, blocks

    @staticmethod
    def _dense_product(indices: List[torch.Tensor], matrix: torch.Tensor, blocks: List[torch.Tensor]) -> torch.Tensor:
        """The same product with every block padded with the identity, by plain autograd."""
        result = matrix
        for step_indices, step_blocks in zip(indices, blocks):
            for position in range(step_indices.shape[0]):
                block = step_blocks[..., position, :, :]
                padded = (
                    torch.eye(N_COLUMNS, dtype=matrix.dtype).expand(*block.shape[:-2], N_COLUMNS, N_COLUMNS).clone()
                )
                columns = step_indices[position]
                padded[..., columns[:, None], columns[None, :]] = block
                result = result @ padded
        return result

    @pytest.mark.parametrize("steps_name", list(STEPS))
    @pytest.mark.parametrize("matrix_batch, block_batch", [(None, None), (3, None), (None, 3), (3, 3), (1, 3)])
    def test_forward_matches_the_padded_product(self, steps_name, matrix_batch, block_batch):
        indices, matrix, blocks = self._make_inputs(steps_name, matrix_batch, block_batch)
        before = matrix.clone()
        result = ReversibleColumnProduct.apply(indices, matrix, *blocks)
        torch.testing.assert_close(
            result,
            self._dense_product(indices, matrix, blocks),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        torch.testing.assert_close(matrix, before, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("steps_name", list(STEPS))
    @pytest.mark.parametrize("matrix_batch, block_batch", [(None, None), (3, None), (None, 3), (3, 3), (1, 3)])
    def test_gradients_match_plain_autograd(self, steps_name, matrix_batch, block_batch):
        """``gradcheck`` cannot be used: a perturbed block is not orthogonal, so the steps cannot be undone."""
        indices, matrix, blocks = self._make_inputs(steps_name, matrix_batch, block_batch)
        generator = torch.Generator().manual_seed(TEST_SEED + 1)
        gradients = []
        for product in (ReversibleColumnProduct.apply, self._dense_product):
            inputs = [matrix.clone().requires_grad_()] + [block.clone().requires_grad_() for block in blocks]
            result = (
                product(indices, inputs[0], *inputs[1:])
                if product is not self._dense_product
                else product(indices, inputs[0], inputs[1:])
            )
            weights = torch.randn(result.shape, dtype=torch.float64, generator=generator.manual_seed(TEST_SEED + 1))
            gradients.append(torch.autograd.grad((result * weights).sum(), inputs))
        for ours, reference in zip(*gradients):
            torch.testing.assert_close(ours, reference, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_inputs_that_need_no_gradient_get_none(self):
        indices, matrix, blocks = self._make_inputs("overlapping_steps", None, None)
        blocks[0].requires_grad_()
        result = ReversibleColumnProduct.apply(indices, matrix, *blocks)
        result.sum().backward()
        assert blocks[0].grad is not None
        assert all(block.grad is None for block in blocks[1:])
        assert matrix.grad is None

    @pytest.mark.parametrize("matrix_batch, block_batch", [(None, None), (3, None), (None, 3), (3, 3)])
    def test_gradients_through_parameters_of_orthogonal_blocks(self, matrix_batch, block_batch):
        """Rotations stay orthogonal under a perturbation of their angle, so ``gradcheck`` applies."""
        indices, matrix, _ = self._make_inputs("overlapping_steps", matrix_batch, None)
        n_blocks = [step_indices.shape[0] for step_indices in indices]
        shape = (sum(n_blocks),) if block_batch is None else (block_batch, sum(n_blocks))
        angles = torch.rand(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(TEST_SEED))

        def product(matrix_input: torch.Tensor, angles_input: torch.Tensor) -> torch.Tensor:
            cos, sin, zero, one = (
                torch.cos(angles_input),
                torch.sin(angles_input),
                torch.zeros_like(angles_input),
                torch.ones_like(angles_input),
            )
            rotations = torch.stack(
                [cos, zero, -sin, zero, zero, one, zero, zero, sin, zero, cos, zero, zero, zero, zero, one], dim=-1
            ).reshape(*angles_input.shape, 4, 4)
            blocks = torch.split(rotations, n_blocks, dim=-3)
            return ReversibleColumnProduct.apply(indices, matrix_input, *blocks)

        inputs = (matrix.requires_grad_(), angles.requires_grad_())
        assert torch.autograd.gradcheck(
            product, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, check_forward_ad=True, fast_mode=True
        )
        if matrix_batch is None and block_batch is None:
            # the second derivative is checked without batch only: it takes minutes with one
            assert torch.autograd.gradgradcheck(product, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL)

    def test_backward_holds_no_matrix_per_step(self):
        """The graph has one node for the whole product, however many steps it has."""
        indices, matrix, blocks = self._make_inputs("overlapping_steps", None, None)
        blocks = [block.requires_grad_() for block in blocks]
        result = ReversibleColumnProduct.apply(indices, matrix, *blocks)
        assert type(result.grad_fn).__name__ == "ReversibleColumnProductBackward"
        assert len(result.grad_fn.next_functions) == len(blocks) + 1

    def test_float32_gradients_stay_close_after_many_steps(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        n_steps = 400
        starts = torch.randint(0, N_COLUMNS - 3, (n_steps,), generator=generator)
        indices = [(start + torch.arange(4))[None, :] for start in starts]
        blocks64 = [self._orthogonal(generator, 1, 4) for _ in range(n_steps)]
        matrix64 = torch.eye(N_COLUMNS, dtype=torch.float64)
        gradients = []
        for dtype in (torch.float32, torch.float64):
            blocks = [block.to(dtype).requires_grad_() for block in blocks64]
            result = ReversibleColumnProduct.apply(indices, matrix64.to(dtype), *blocks)
            weights = torch.linspace(0.0, 1.0, N_COLUMNS**2, dtype=dtype).reshape(N_COLUMNS, N_COLUMNS)
            gradients.append(torch.stack(torch.autograd.grad((result * weights).sum(), blocks)).double())
        np.testing.assert_allclose(
            gradients[0].numpy(), gradients[1].numpy(), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("with_matrix_tangent, with_block_tangents", [(True, False), (False, True)])
    def test_forward_mode_with_tangents_on_some_inputs_only(self, with_matrix_tangent, with_block_tangents):
        indices, matrix, blocks = self._make_inputs("overlapping_steps", None, None)
        generator = torch.Generator().manual_seed(TEST_SEED + 2)
        matrix_tangent = torch.randn(matrix.shape, dtype=torch.float64, generator=generator)
        block_tangents = [torch.randn(block.shape, dtype=torch.float64, generator=generator) for block in blocks]
        with torch.autograd.forward_ad.dual_level():
            dual_matrix = torch.autograd.forward_ad.make_dual(matrix, matrix_tangent) if with_matrix_tangent else matrix
            dual_blocks = [
                torch.autograd.forward_ad.make_dual(block, tangent) if with_block_tangents else block
                for block, tangent in zip(blocks, block_tangents)
            ]
            result = ReversibleColumnProduct.apply(indices, dual_matrix, *dual_blocks)
            tangent = torch.autograd.forward_ad.unpack_dual(result).tangent
        step = 1e-6
        shifted_matrix = matrix + step * matrix_tangent if with_matrix_tangent else matrix
        shifted_blocks = [
            block + step * block_tangent if with_block_tangents else block
            for block, block_tangent in zip(blocks, block_tangents)
        ]
        difference = self._dense_product(indices, shifted_matrix, shifted_blocks) - self._dense_product(
            indices, matrix, blocks
        )
        torch.testing.assert_close(tangent, difference / step, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_second_derivative_with_respect_to_the_entries_of_a_block_matches_plain_autograd(self):
        """Away from the orthogonal group, only the inverse of the block undoes its step."""
        indices, matrix, blocks = self._make_inputs("two_steps_on_the_same_columns", None, None)
        weights = torch.randn(matrix.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(TEST_SEED))

        def reversible_loss(block: torch.Tensor) -> torch.Tensor:
            return (ReversibleColumnProduct.apply(indices, matrix, block, blocks[1]) @ weights).pow(2).sum()

        def dense_loss(block: torch.Tensor) -> torch.Tensor:
            return (self._dense_product(indices, matrix, [block, blocks[1]]) @ weights).pow(2).sum()

        torch.testing.assert_close(
            torch.autograd.functional.hessian(reversible_loss, blocks[0]),
            torch.autograd.functional.hessian(dense_loss, blocks[0]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_gradients_of_slightly_non_orthogonal_blocks_match_plain_autograd(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        n_steps = 300
        starts = torch.randint(0, N_COLUMNS - 3, (n_steps,), generator=generator)
        indices = [(start + torch.arange(4))[None, :] for start in starts]
        blocks = [(1.0 + 3e-5) * self._orthogonal(generator, 1, 4) for _ in range(n_steps)]
        matrix = torch.eye(N_COLUMNS, dtype=torch.float64)
        weights = torch.linspace(0.0, 1.0, N_COLUMNS**2, dtype=torch.float64).reshape(N_COLUMNS, N_COLUMNS)
        gradients = []
        for reversible in (True, False):
            inputs = [block.clone().requires_grad_() for block in blocks]
            if reversible:
                result = ReversibleColumnProduct.apply(indices, matrix, *inputs)
            else:
                result = self._dense_product(indices, matrix, inputs)
            gradients.append(torch.stack(torch.autograd.grad((result * weights).sum(), inputs)))
        torch.testing.assert_close(gradients[0], gradients[1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("matrix_batch, block_batch", [(None, None), (3, None), (None, 3), (3, 3)])
    def test_batched_gradient_outputs_are_supported(self, matrix_batch, block_batch):
        """``torch.autograd.functional.jacobian(vectorize=True)`` asks for all the rows of a Jacobian at once."""
        indices, matrix, blocks = self._make_inputs("overlapping_steps", matrix_batch, block_batch)
        inputs = [matrix.requires_grad_()] + [block.requires_grad_() for block in blocks]
        result = ReversibleColumnProduct.apply(indices, inputs[0], *inputs[1:])
        generator = torch.Generator().manual_seed(TEST_SEED + 3)
        grad_outputs = torch.randn(4, *result.shape, dtype=torch.float64, generator=generator)
        batched = torch.autograd.grad(
            result, inputs, grad_outputs=grad_outputs, is_grads_batched=True, retain_graph=True
        )
        for index in range(grad_outputs.shape[0]):
            single = torch.autograd.grad(result, inputs, grad_outputs=grad_outputs[index], retain_graph=True)
            for ours, reference in zip(batched, single):
                torch.testing.assert_close(
                    ours[index], reference, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
                )
