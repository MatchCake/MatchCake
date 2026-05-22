import numbers

import numpy as np
import pytest
import torch

from matchcake.utils.torch_utils import detach, to_numpy, to_tensor, torch_wrap_circular_bounds

from ..configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON, TEST_SEED, set_seed


class TestToTensor:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_from_numpy(self):
        x = np.array([1.0, 2.0])
        result = to_tensor(x, dtype=torch.float64)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_from_tensor(self):
        x = torch.tensor([1.0, 2.0])
        result = to_tensor(x, dtype=torch.float64)
        assert isinstance(result, torch.Tensor)

    def test_from_number(self):
        result = to_tensor(3.14, dtype=torch.float64)
        assert isinstance(result, torch.Tensor)

    def test_from_dict(self):
        x = {"a": np.array([1.0]), "b": 2.0}
        result = to_tensor(x, dtype=torch.float64)
        assert isinstance(result, dict)
        assert isinstance(result["a"], torch.Tensor)
        assert isinstance(result["b"], torch.Tensor)

    def test_from_list(self):
        x = [1.0, 2.0]
        result = to_tensor(x, dtype=torch.float64)
        assert isinstance(result, list)

    def test_from_tuple(self):
        x = (1.0, 2.0)
        result = to_tensor(x, dtype=torch.float64)
        assert isinstance(result, tuple)

    def test_from_fallback_object(self):
        class _ConvertibleObject:
            def __float__(self):
                return 42.0

        obj = _ConvertibleObject()
        result = to_tensor(obj, dtype=torch.float64)
        assert isinstance(result, torch.Tensor)

    def test_unsupported_type_raises(self):
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            to_tensor(object(), dtype=torch.float64)


class TestToNumpy:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_from_numpy(self):
        x = np.array([1.0, 2.0])
        result = to_numpy(x, dtype=np.float64)
        np.testing.assert_allclose(result, x, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_from_tensor(self):
        x = torch.tensor([1.0, 2.0])
        result = to_numpy(x, dtype=np.float64)
        assert isinstance(result, np.ndarray)

    def test_from_number(self):
        result = to_numpy(3.14)
        assert isinstance(result, numbers.Number)

    def test_from_dict(self):
        x = {"a": np.array([1.0]), "b": torch.tensor([2.0])}
        result = to_numpy(x, dtype=np.float64)
        assert isinstance(result, dict)
        assert isinstance(result["a"], np.ndarray)

    def test_from_fallback(self):
        x = [1.0, 2.0]
        result = to_numpy(x, dtype=np.float64)
        assert isinstance(result, np.ndarray)

    def test_unconvertible_type_raises(self):
        class _BadArray:
            def __array__(self, *args, **kwargs):
                raise RuntimeError("Cannot convert to numpy array.")

        with pytest.raises(ValueError):
            to_numpy(_BadArray(), dtype=np.float64)


class TestDetach:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_tensor(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = detach(x)
        assert not result.requires_grad

    def test_dict(self):
        x = {"a": torch.tensor([1.0], requires_grad=True)}
        result = detach(x)
        assert isinstance(result, dict)
        assert not result["a"].requires_grad

    def test_list(self):
        x = [torch.tensor([1.0], requires_grad=True)]
        result = detach(x)
        assert isinstance(result, list)
        assert not result[0].requires_grad

    def test_tuple(self):
        x = (torch.tensor([1.0], requires_grad=True),)
        result = detach(x)
        assert isinstance(result, tuple)
        assert not result[0].requires_grad

    def test_non_tensor_passthrough(self):
        result = detach(3.14)
        np.testing.assert_allclose(result, 3.14, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)


class TestTorchWrapCircularBounds:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "value, lower, upper, expected",
        [
            (0.5, 0.0, 1.0, 0.5),
            (1.5, 0.0, 1.0, 0.5),
            (-0.5, 0.0, 1.0, 0.5),
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
        ],
    )
    def test_wraps_correctly(self, value, lower, upper, expected):
        tensor = torch.tensor(value, dtype=torch.float64)
        result = torch_wrap_circular_bounds(tensor, lower_bound=lower, upper_bound=upper)
        torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
