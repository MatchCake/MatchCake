import numpy as np
import pytest

from matchcake.matchgate_parameter_sets import transfer_functions
from matchcake.matchgate_parameter_sets.transfer_functions import (
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
)
from ...configs import (
    TEST_SEED,
    set_seed,
)

MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_hamiltonian_coefficients_to_standard(
    __from_params: MatchgateHamiltonianCoefficientsParams,
    __to_params: MatchgateStandardParams,
):
    to_params = transfer_functions.hamiltonian_coefficients_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."
