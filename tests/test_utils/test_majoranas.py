import numpy as np
import pytest

from matchcake import utils
from matchcake.utils import get_majorana_pauli_string

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize("k,n", [(k, n) for n in range(2, 10) for k in range(n)])
def test_majorana_output_shape(k: int, n: int):
    """
    Test that the function output a matrix of shape 2^n x 2^n.
    """
    out_shape = utils.get_majorana(k, n).shape
    target_shape = (2**n, 2**n)
    assert out_shape == target_shape, (
        f"The output matrix is not of the correct shape with k={k} and n={n}. "
        f"Got {out_shape} instead of {target_shape}"
    )


@pytest.mark.parametrize(
    "mu,nu,n",
    [(mu, nu, n) for n in range(1, 5) for mu in range(2 * n) for nu in range(2 * n)],
)
def test_majoranas_anti_commutation(mu: int, nu: int, n: int):
    c_mu = utils.get_majorana(mu, n)
    c_nu = utils.get_majorana(nu, n)
    anti_commutator = c_mu @ c_nu + c_nu @ c_mu
    target = 2 * np.eye(2**n) * int(mu == nu)
    np.testing.assert_allclose(
        anti_commutator,
        target,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg=f"The Majorana matrices do not anticommute "
        f"with mu={mu}, nu={nu} and n={n}. "
        f"Got {anti_commutator} instead of "
        f"{target}",
    )


@pytest.mark.parametrize("i,n", [(i, n) for n in range(1, 5) for i in range(2 * n)])
def test_majoranas_identity(i: int, n: int):
    c_mu = utils.get_majorana(i, n)
    c_nu = utils.get_majorana(i, n)
    product = c_mu @ c_nu
    target = np.eye(2**n)
    np.testing.assert_allclose(
        product,
        target,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "i,c_i",
    [
        (0, np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])),
        (1, np.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]])),
        (2, np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])),
        (3, np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])),
    ],
)
def test_get_majorana(i, c_i):
    _c_i = utils.get_majorana(i, len(c_i) // 2)
    np.testing.assert_allclose(
        _c_i,
        c_i,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "i,j,c_ij",
    [
        (
            0,
            1,
            np.array([[1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, -1j]]),
        ),
        (0, 2, np.array([[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])),
        (
            0,
            3,
            np.array([[0, 0, 0, 1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]]),
        ),
        (1, 2, np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]])),
        (1, 3, np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])),
        (
            2,
            3,
            np.array([[1j, 0, 0, 0], [0, -1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]]),
        ),
    ],
)
def test_get_majorana_product(i, j, c_ij):
    c_i = utils.get_majorana(i, len(c_ij) // 2)
    c_j = utils.get_majorana(j, len(c_ij) // 2)
    _c_ij = c_i @ c_j
    np.testing.assert_allclose(
        _c_ij,
        c_ij,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


class TestMajorana:
    @pytest.mark.parametrize(
        "i, n, join_char, expected",
        [
            (0, 3, "⊗", "X⊗I⊗I"),
            (1, 3, "⊗", "Y⊗I⊗I"),
            (2, 3, "⊗", "Z⊗X⊗I"),
            (3, 3, "⊗", "Z⊗Y⊗I"),
            (4, 3, "⊗", "Z⊗Z⊗X"),
            (5, 3, "⊗", "Z⊗Z⊗Y"),
        ],
    )
    def test_get_majorana_pauli_string(self, i, n, join_char, expected):
        result = get_majorana_pauli_string(i, n, join_char)
        assert result == expected, f"Expected {expected} but got {result} for i={i}, n={n}, join_char='{join_char}'"
