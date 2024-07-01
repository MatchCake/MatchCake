import numpy as np
import pytest

from matchcake.matchgate_parameter_sets import transfer_functions
from matchcake.matchgate_parameter_sets.transfer_functions import (
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
)
from matchcake.utils import (
    PAULI_X,
    PAULI_Z,
)
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
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
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
        (
                MatchgatePolarParams(r0=1, r1=0, theta0=0, theta1=0, theta2=np.pi / 2, theta3=0, theta4=np.pi / 2),
                MatchgateStandardParams(
                    a=PAULI_Z[0, 0], b=PAULI_Z[0, 1], c=PAULI_Z[1, 0], d=PAULI_Z[1, 1],
                    w=PAULI_X[0, 0], x=PAULI_X[0, 1], y=PAULI_X[1, 0], z=PAULI_X[1, 1]
                ),  # fSWAP
        ),
        (
                MatchgatePolarParams(
                    r0=1, r1=1, theta0=0.5j * np.pi, theta1=0.5j * np.pi,
                    theta2=0.5j * np.pi, theta3=0.5j * np.pi, theta4=0.5j * np.pi
                ),
                MatchgateStandardParams(a=0.2079, b=0, c=0, d=0.2079, w=0.2079, x=0, y=0, z=0.2079),
        ),
    ],
)
def test_polar_to_standard(__from_params: MatchgatePolarParams, __to_params: MatchgateStandardParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_polar_to_standard_back_and_forth(__from_params: MatchgatePolarParams):
    _from_params = __from_params.__copy__()
    to_params_list = []
    for _ in range(2):
        to_params = transfer_functions.polar_to_standard(_from_params)
        _from_params = transfer_functions.standard_to_polar(to_params)
        to_params_list.append(to_params)
    assert all([to_params_list[0] == to_params for to_params in to_params_list])


def test_polar_to_standard_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(
        batch_size, MatchgatePolarParams.N_PARAMS,
        device="cpu",
        requires_grad=True
    )
    __from_params = MatchgatePolarParams(rn_tensor)
    assert isinstance(__from_params.to_tensor(), torch.Tensor)
    assert __from_params.to_tensor().requires_grad
    assert __from_params.requires_grad

    to_params = transfer_functions.polar_to_standard(__from_params)
    assert isinstance(to_params.to_tensor(), torch.Tensor)
    assert to_params.to_tensor().requires_grad
    assert to_params.requires_grad
