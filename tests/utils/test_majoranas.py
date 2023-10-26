import pytest
import numpy as np
from msim import utils


@pytest.mark.parametrize("k,n", [(k, n) for n in range(2, 10) for k in range(n)])
def test_majorana_output_shape(k: int, n: int):
    """
    Test that the function output a matrix of shape 2^n x 2^n.
    """
    out_shape = utils.get_majorana(k, n).shape
    target_shape = (2 ** n, 2 ** n)
    assert out_shape == target_shape, (f"The output matrix is not of the correct shape with k={k} and n={n}. "
                                       f"Got {out_shape} instead of {target_shape}")


@pytest.mark.parametrize("mu,nu,n", [(mu, nu, n) for n in range(1, 5) for mu in range(2*n) for nu in range(2*n)])
def test_majoranas_anti_commutation(mu: int, nu: int, n: int):
    c_mu = utils.get_majorana(mu, n)
    c_nu = utils.get_majorana(nu, n)
    anti_commutator = c_mu @ c_nu + c_nu @ c_mu
    target = 2 * np.eye(2 ** n) * int(mu == nu)
    assert np.allclose(anti_commutator, target), (f"The Majorana matrices do not anticommute "
                                                  f"with mu={mu}, nu={nu} and n={n}. "
                                                  f"Got {anti_commutator} instead of "
                                                  f"{target}")
