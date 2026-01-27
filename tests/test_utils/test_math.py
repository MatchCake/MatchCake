import numpy as np
import pytest
import torch

from matchcake import utils
from matchcake.utils.math import (
    check_is_unitary,
    circuit_matmul,
    convert_1d_to_2d_indexes,
    convert_2d_to_1d_indexes,
    convert_tensors_to_same_type,
    convert_tensors_to_same_type_and_cast_to,
    det,
    eye_like,
    fermionic_operator_matmul,
    matmul,
    svd,
)

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)
HIGH_VALUE = 2 * np.pi
LOW_PRECISION_TERMS = 10
MEDIUM_PRECISION_TERMS = 18
HIGH_PRECISION_TERMS = 26


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(LOW_PRECISION_TERMS, LOW_PRECISION_TERMS + N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_exp_taylor_series_approx(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(MEDIUM_PRECISION_TERMS, MEDIUM_PRECISION_TERMS + N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_exp_taylor_series_medium_precision(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(
        out,
        target,
        atol=(ATOL_SCALAR_COMPARISON + ATOL_APPROX_COMPARISON) / 2,
        rtol=(RTOL_SCALAR_COMPARISON + RTOL_APPROX_COMPARISON) / 2,
    )


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(HIGH_PRECISION_TERMS, HIGH_PRECISION_TERMS + N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_exp_taylor_series_high_precision(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)


@pytest.mark.parametrize(
    "probs_shape",
    [
        probs_shape
        for probs_shape in [
            (10,),
            (
                32,
                10,
            ),
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_random_index(probs_shape):
    n = 1000 * probs_shape[-1]
    probs = np.random.uniform(0, 1, probs_shape)
    probs_normed = probs / probs.sum(axis=-1, keepdims=True)
    indexes = utils.math.random_index(probs_normed, n=n)
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=probs.shape[-1]), axis=0, arr=indexes).T
    probs_estimate = counts / indexes.shape[0]
    probs_estimate_normed = probs_estimate / probs_estimate.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        probs_estimate_normed,
        probs_normed,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "probs_shape",
    [probs_shape for probs_shape in [(4, 3, 2), (5, 4, 3)] for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_random_index_3d_none(probs_shape):
    n = 128 * np.prod(probs_shape)

    probs = np.random.uniform(0, 1, probs_shape)
    probs_normed = probs / probs.sum(axis=-1, keepdims=True)
    indexes = np.stack([utils.math.random_index(probs_normed, n=None) for _ in range(n)], axis=0)
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=probs.shape[-1]),
        axis=0,
        arr=indexes.reshape(indexes.shape[0], -1),
    ).T.reshape(probs.shape)
    probs_estimate = counts / indexes.shape[0]
    probs_estimate_normed = probs_estimate / probs_estimate.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        probs_estimate_normed,
        probs_normed,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


class TestMath:
    def test_convert_and_cast_like(self):
        source = np.random.uniform(-10, 10, (4, 3)).astype(np.float32)
        target = np.random.uniform(-10, 10, (4, 3)).astype(np.float64)
        out = utils.math.convert_and_cast_like(source, target)
        assert out.shape == target.shape
        assert out.dtype == target.dtype
        np.testing.assert_allclose(
            out,
            np.broadcast_to(source, target.shape).astype(np.float64),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_convert_like_and_cast_to(self):
        source = np.random.uniform(-10, 10, (4, 3)).astype(np.float32)
        target = torch.from_numpy(source).to(dtype=torch.float64)
        out = utils.math.convert_like_and_cast_to(source, target, torch.complex32)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.complex32

    def test_convert_tensors_to_same_type_and_cast_to(self):
        tensor_a = np.random.rand(4, 3).astype(np.float32)
        tensor_b = torch.randn((4, 3), dtype=torch.float64)
        out_a, out_b = convert_tensors_to_same_type_and_cast_to(
            [tensor_a, tensor_b],
            dtype=torch.complex64,
        )
        assert isinstance(out_a, torch.Tensor)
        assert isinstance(out_b, torch.Tensor)
        assert out_a.dtype == torch.complex64
        assert out_b.dtype == torch.complex64

    def test_convert_tensors_to_same_type_and_cast_to_empty(self):
        tensor_list = []
        out_list = convert_tensors_to_same_type_and_cast_to(
            tensor_list,
            dtype=torch.complex64,
        )
        assert out_list == []

    def test_convert_tensors_to_same_type(self):
        tensor_a = np.random.rand(4, 3).astype(np.float32)
        tensor_b = torch.randn((4, 3), dtype=torch.float64)
        out_a, out_b = convert_tensors_to_same_type([tensor_a, tensor_b])
        assert isinstance(out_a, torch.Tensor)
        assert isinstance(out_b, torch.Tensor)

    def test_convert_tensors_to_same_type_empty(self):
        tensor_list = []
        out_list = convert_tensors_to_same_type(tensor_list)
        assert out_list == []

    def test_convert_and_cast_tensor_from_tensors(self):
        source = np.random.rand(4, 3).astype(np.float32)
        tensor_a = np.random.rand(4, 3).astype(np.float32)
        tensor_b = torch.randn((4, 3), dtype=torch.complex64)
        out = utils.math.convert_and_cast_tensor_from_tensors(source, [tensor_a, tensor_b])
        assert isinstance(out, torch.Tensor)

    def test_convert_and_cast_tensor_from_tensors_empty(self):
        source = np.random.rand(4, 3).astype(np.float32)
        tensor_list = []
        out = utils.math.convert_and_cast_tensor_from_tensors(source, tensor_list)
        assert isinstance(out, np.ndarray)
        assert out.dtype == source.dtype

    @pytest.mark.parametrize(
        "indexes, n_rows, expected",
        [
            # Test case: Single 2D index
            ([(0, 0)], 3, [0]),
            # Test case: Multiple 2D indexes with default n_rows inferred
            ([(0, 0), (1, 1), (2, 2)], None, [0, 4, 8]),
            # Test case: Multiple 2D indexes with custom n_rows
            ([(1, 0), (2, 1), (3, 2)], 4, [4, 9, 14]),
            # Test case: Edge case with no indexes
            ([], None, []),
            # Test case: Edge case where input is shaped oddly
            ([(0, 1), (1, 0)], 2, [1, 2]),
        ],
    )
    def test_convert_2d_to_1d_indexes(self, indexes, n_rows, expected):
        """
        Test that convert_2d_to_1d_indexes correctly converts 2D indexes to 1D.
        """
        result = convert_2d_to_1d_indexes(indexes, n_rows)
        np.testing.assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "indexes, n_rows, error_type",
        [
            # Test case: Invalid input (non-iterable indexes)
            ("invalid_input", 3, TypeError),
            # Test case: Invalid n_rows value (non-integer)
            ([(0, 0)], "invalid_n_rows", TypeError),
            # Test case: 2D index with invalid dimensions
            ([(0, 0, 0)], None, ValueError),
        ],
    )
    def test_convert_2d_to_1d_indexes_errors(self, indexes, n_rows, error_type):
        """
        Test that convert_2d_to_1d_indexes raises errors for invalid inputs.
        """
        with pytest.raises(error_type):
            convert_2d_to_1d_indexes(indexes, n_rows)

    @pytest.mark.parametrize(
        "indexes, n_rows, expected",
        [
            # Test case: Single 1D index
            ([0], 3, [(0, 0)]),
            # Test case: Multiple 1D indexes with default n_rows inferred
            ([0, 4, 8], 3, [(0, 0), (1, 1), (2, 2)]),
            # Test case: Multiple 1D indexes with custom n_rows
            ([4, 9, 14], 4, [(1, 0), (2, 1), (3, 2)]),
            # Test case: Edge case with no indexes
            ([], 3, []),
            # Test case: Edge case where input is shaped oddly
            ([1, 2], 2, [(0, 1), (1, 0)]),
        ],
    )
    def test_convert_1d_to_2d_indexes(self, indexes, n_rows, expected):
        result = convert_1d_to_2d_indexes(indexes, n_rows)
        np.testing.assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "indexes, n_rows, error_type",
        [
            # Test case: Invalid input (non-iterable indexes)
            ("invalid_input", 3, TypeError),
            # Test case: Invalid n_rows value (non-integer)
            (
                [
                    0,
                ],
                "invalid_n_rows",
                TypeError,
            ),
            # Test case: 2D index with invalid dimensions
            ([(0, 0, 0)], None, ValueError),
        ],
    )
    def test_convert_1d_to_2d_indexes_errors(self, indexes, n_rows, error_type):
        """
        Test that convert_1d_to_2d_indexes raises errors for invalid inputs.
        """
        with pytest.raises(error_type):
            convert_1d_to_2d_indexes(indexes, n_rows)

    @pytest.mark.parametrize("operator", ["einsum", "matmul", "@"])
    def test_matmul(self, operator):
        a = np.random.rand(4, 3).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)
        out = matmul(a, b, operator)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(
            out,
            expected,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_circuit_matmul(self):
        a = np.random.rand(4, 4).astype(np.float32)
        b = np.random.rand(4, 4).astype(np.float32)
        out = circuit_matmul(a, b)
        expected = np.matmul(b, a)
        np.testing.assert_allclose(
            out,
            expected,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_fermionic_operator_matmul(self):
        a = np.random.rand(4, 4).astype(np.float32)
        b = np.random.rand(4, 4).astype(np.float32)
        out = fermionic_operator_matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(
            out,
            expected,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_eye_like(self):
        source = np.random.rand(3, 4, 4).astype(np.float32)
        out = eye_like(source)
        expected = np.stack([np.eye(4), np.eye(4), np.eye(4)], axis=0)
        np.testing.assert_allclose(
            out,
            expected,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_check_is_unitary(self):
        matrix = np.array([[1, 0], [0, 1]], dtype=np.complex64)
        assert check_is_unitary(matrix)

        non_unitary_matrix = np.array([[1, 1], [0, 1]], dtype=np.complex64)
        assert not check_is_unitary(non_unitary_matrix)

    def test_det(self):
        x = np.linspace(-1, 1, 6**2).reshape(6, 6)
        expected = np.linalg.det(x)
        out = det(x)
        np.testing.assert_allclose(
            out,
            expected,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

        x_torch = torch.from_numpy(x).float()
        expected_torch = torch.det(x_torch)
        out_torch = det(x_torch)
        torch.testing.assert_close(out_torch, expected_torch)

    def test_svd(self):
        x = np.linspace(-1, 1, 6**2).reshape(6, 6)
        expected_u, expected_s, expected_vh = np.linalg.svd(x)
        out_u, out_s, out_vh = svd(x)
        np.testing.assert_allclose(
            out_u,
            expected_u,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        np.testing.assert_allclose(
            out_s,
            expected_s,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        np.testing.assert_allclose(
            out_vh,
            expected_vh,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
