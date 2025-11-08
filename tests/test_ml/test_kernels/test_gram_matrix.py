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
        "shape, requires_grad", [(shape, rg) for shape in [(4, 5), (5, 5), (5, 4)] for rg in [True, False]]
    )
    def test_symmetrize_(self, shape, requires_grad):
        gram = GramMatrix(shape, initial_value=0.0, requires_grad=requires_grad)
        gram.apply_(lambda ids: np.random.rand(len(ids)), symmetrize=False)
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
