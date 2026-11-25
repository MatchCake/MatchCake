import os

import numpy as np
import pytest
import torch

from matchcake.ml.kernels.gram_matrix import GramMatrix


class TestGramMatrix:
    @pytest.mark.parametrize("shape, initial_value", [((4, 5), 0.0), ((5, 5), 1.0), ((5, 4), -1.0)])
    def test_init(self, shape, initial_value):
        gram = GramMatrix(shape, initial_value=initial_value, requires_grad=False)
        assert gram.shape == shape
        assert gram._tensor is None
        assert isinstance(gram._memmap, np.memmap)
        for i, j in np.ndindex(gram.shape):
            torch.testing.assert_close(gram[i, j], torch.tensor(initial_value))

    @pytest.mark.parametrize("shape, initial_value", [((4, 5), 0.0), ((5, 5), 1.0), ((5, 4), -1.0)])
    def test_init_requires_grad(self, shape, initial_value):
        gram = GramMatrix(shape, initial_value=initial_value, requires_grad=True)
        assert gram.shape == shape
        assert isinstance(gram._tensor, torch.Tensor)
        assert gram._memmap is None
        for i, j in np.ndindex(gram.shape):
            torch.testing.assert_close(gram[i, j], torch.tensor(initial_value))

    @pytest.mark.parametrize(
        "shape, requires_grad", [(shape, rg) for shape in [(4, 5), (5, 5), (5, 4)] for rg in [True, False]]
    )
    def test_setitem(self, shape, requires_grad):
        gram = GramMatrix(shape, initial_value=0.0, requires_grad=requires_grad)
        value = 1.0
        for i, j in np.ndindex(gram.shape):
            gram[i, j] = value
            torch.testing.assert_close(gram[i, j], torch.tensor(value))

    @pytest.mark.parametrize(
        "shape, requires_grad", [(shape, rg) for shape in [(4, 5), (5, 5), (5, 4)] for rg in [True, False]]
    )
    def test_apply_(self, shape, requires_grad):
        gram = GramMatrix(shape, initial_value=0.0, requires_grad=requires_grad)
        value = -1.0
        gram.apply_(lambda ids: value, symmetrize=True)
        for i, j in np.ndindex(gram.shape):
            if i == j:
                torch.testing.assert_close(gram[i, j], torch.tensor(1.0))
            else:
                torch.testing.assert_close(gram[i, j], torch.tensor(value))

    @pytest.mark.parametrize(
        "shape, requires_grad", [(shape, rg) for shape in [(4, 5), (5, 5), (5, 4), (1, 1)] for rg in [True, False]]
    )
    def test_apply_without_symmetrize_evaluates_every_entry(self, shape, requires_grad):
        gram = GramMatrix(shape, initial_value=0.0, requires_grad=requires_grad)
        gram.apply_(lambda ids: 10.0 * ids[:, 0] + ids[:, 1] + 0.5, batch_size=3, symmetrize=False)
        expected = torch.tensor(
            [[10.0 * i + j + 0.5 for j in range(shape[1])] for i in range(shape[0])], dtype=torch.float32
        )
        torch.testing.assert_close(gram.to_tensor(), expected)

    @pytest.mark.parametrize(
        "shape, requires_grad", [(shape, rg) for shape in [(4, 5), (5, 5), (5, 4)] for rg in [True, False]]
    )
    def test_symmetrize_(self, shape, requires_grad):
        gram = GramMatrix(shape, initial_value=0.0, requires_grad=requires_grad)
        for indices in gram.indices_batch_generator():
            gram[indices[:, 0], indices[:, 1]] = np.random.rand(len(indices))
        torch.testing.assert_close(torch.diagonal(gram.to_tensor()), torch.zeros_like(torch.diagonal(gram.to_tensor())))
        triu_indices = np.triu_indices(n=shape[0], m=shape[1], k=1)
        tril_indices = np.tril_indices(n=shape[0], m=shape[1], k=-1)
        if shape[0] < shape[1]:
            torch.testing.assert_close(gram[tril_indices], torch.zeros_like(gram[tril_indices]))
        else:
            torch.testing.assert_close(gram[triu_indices], torch.zeros_like(gram[triu_indices]))
        gram.symmetrize_()
        torch.testing.assert_close(torch.diagonal(gram.to_tensor()), torch.ones_like(torch.diagonal(gram.to_tensor())))
        if shape[0] < shape[1]:
            torch.testing.assert_close(gram[tril_indices], gram[tril_indices[1], tril_indices[0]])
        else:
            torch.testing.assert_close(gram[triu_indices], gram[triu_indices[1], triu_indices[0]])

    @pytest.mark.parametrize(
        "dtype, requires_grad", [(dtype, rg) for dtype in [torch.float32, torch.float64] for rg in [True, False]]
    )
    def test_dtype(self, dtype, requires_grad):
        gram = GramMatrix((3, 4), requires_grad=requires_grad, dtype=dtype)
        value = 1.0 + 1e-12
        gram.apply_(lambda ids: np.full(len(ids), value), symmetrize=False)
        assert gram.dtype == dtype
        assert gram.to_tensor().dtype == dtype
        assert gram[0, 1].dtype == dtype
        is_resolved = bool((gram.to_tensor() != 1.0).all())
        assert is_resolved == (dtype == torch.float64)

    def test_init_closes_its_file_descriptor(self):
        if not os.path.isdir("/proc/self/fd"):
            pytest.skip("No /proc filesystem to count the open file descriptors.")
        gram_matrices = [GramMatrix((2, 2)) for _ in range(3)]
        n_open = len(os.listdir("/proc/self/fd"))
        gram_matrices.extend(GramMatrix((2, 2)) for _ in range(40))
        # A live memory map keeps one descriptor; the one opened to create the file must be closed.
        assert len(os.listdir("/proc/self/fd")) <= n_open + 40
        del gram_matrices

    def test_default_dtype(self):
        assert GramMatrix((2, 2)).dtype == torch.float32

    def test_to_tensor(self):
        gram = GramMatrix((4, 5), initial_value=0.0, requires_grad=False)
        gram.apply_(lambda ids: np.random.rand(len(ids)), symmetrize=False)
        tensor = gram.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == gram.shape
        assert tensor.requires_grad == gram.requires_grad

    def test_del(self):
        gram = GramMatrix((4, 5), initial_value=0.0, requires_grad=False)
        assert gram._filepath.exists()
        del gram
        assert True

    def test_indices_batch_generator_square(self):
        gram = GramMatrix((10, 10))
        expected_indices = np.stack(np.tril_indices(n=gram.shape[0], m=gram.shape[1], k=-1), axis=1)
        indices_from_batches = np.concatenate(list(gram.indices_batch_generator()), axis=0)
        assert indices_from_batches.shape == expected_indices.shape
        indices_from_batches_unique = np.unique(indices_from_batches, axis=0)
        assert indices_from_batches_unique.shape == indices_from_batches.shape
        np.testing.assert_equal(indices_from_batches_unique, np.unique(expected_indices, axis=0))

    def test_indices_batch_generator_rec_tril(self):
        gram = GramMatrix((12, 10))
        expected_indices = np.stack(np.tril_indices(n=gram.shape[0], m=gram.shape[1], k=-1), axis=1)
        indices_from_batches = np.concatenate(list(gram.indices_batch_generator()), axis=0)
        assert indices_from_batches.shape == expected_indices.shape
        indices_from_batches_unique = np.unique(indices_from_batches, axis=0)
        assert indices_from_batches_unique.shape == indices_from_batches.shape
        np.testing.assert_equal(indices_from_batches_unique, np.unique(expected_indices, axis=0))

    def test_indices_batch_generator_rec_triu(self):
        gram = GramMatrix((10, 12))
        expected_indices = np.stack(np.triu_indices(n=gram.shape[0], m=gram.shape[1], k=1), axis=1)
        indices_from_batches = np.concatenate(list(gram.indices_batch_generator()), axis=0)
        assert indices_from_batches.shape == expected_indices.shape
        indices_from_batches_unique = np.unique(indices_from_batches, axis=0)
        assert indices_from_batches_unique.shape == indices_from_batches.shape
        np.testing.assert_equal(indices_from_batches_unique, np.unique(expected_indices, axis=0))

    @pytest.mark.parametrize("shape", [(10, 10), (12, 10), (10, 12), (1, 1), (1, 3)])
    def test_indices_batch_generator_not_triangular(self, shape):
        gram = GramMatrix(shape)
        expected_indices = np.stack([ids.ravel() for ids in np.indices(shape)], axis=1)
        batches = list(gram.indices_batch_generator(batch_size=7, triangular=False))
        assert all(len(batch) <= 7 for batch in batches)
        np.testing.assert_equal(np.concatenate(batches, axis=0), expected_indices)
