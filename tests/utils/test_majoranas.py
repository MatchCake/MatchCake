import pytest
import numpy as np
from msim import utils



r"""
Test the majorana functions in order to check that they return the correct matrices.

- Make that the matrices are of the correct shape
- Make sure that :math:`\{c_\mu,c_\nu/} = 2\delta_{\mu\nu}I`

"""


@pytest.mark.parametrize("k,n", [(k, n) for n in range(3, 4) for k in range(n)])
def test_majorana_mu_output_shape(k: int, n: int):
    """
    Test that the function output a matrix of shape n^2 x n^2.
    
    :param k:
    :param n:
    :return:
    """
    out_shape = utils.get_majorana_mu(k, n).shape
    target_shape = (2 ** n, 2 ** n)
    assert out_shape == target_shape, (f"The output matrix is not of the correct shape with k={k} and n={n}. "
                                       f"Got {out_shape} instead of {target_shape}")
    


