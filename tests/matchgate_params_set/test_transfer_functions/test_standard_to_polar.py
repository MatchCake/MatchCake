from typing import Literal

import numpy as np
import pennylane as qml
import pytest

from matchcake.matchgate_parameter_sets import transfer_functions
from matchcake.matchgate_parameter_sets.transfer_functions import (
    MatchgateComposedHamiltonianParams, MatchgateHamiltonianCoefficientsParams,
    MatchgatePolarParams, MatchgateStandardParams)
from matchcake.utils import PAULI_X, PAULI_Z

from ...configs import N_RANDOM_TESTS_PER_CASE, TEST_SEED, set_seed

MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
        (
            MatchgateStandardParams(
                a=PAULI_Z[0, 0],
                b=PAULI_Z[0, 1],
                c=PAULI_Z[1, 0],
                d=PAULI_Z[1, 1],
                w=PAULI_X[0, 0],
                x=PAULI_X[0, 1],
                y=PAULI_X[1, 0],
                z=PAULI_X[1, 1],
            ),  # fSWAP
            MatchgatePolarParams(
                r0=1,
                r1=0,
                theta0=0,
                theta1=0,
                theta2=np.pi / 2,
                theta3=0,
                theta4=np.pi / 2,
            ),
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=0, x=1, y=1, z=1),
            MatchgatePolarParams(
                r0=0,
                r1=0,
                theta0=0,
                theta1=0,
                theta2=np.pi / 2,
                theta3=0,
                theta4=np.pi / 2,
            ),
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=1, x=1, y=1, z=1),
            MatchgatePolarParams(r0=0, r1=1, theta0=0, theta1=0, theta2=0, theta3=0, theta4=0),
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=0.5, x=1, y=1, z=1),
            MatchgatePolarParams(
                r0=0,
                r1=0.5,
                theta0=0,
                theta1=0,
                theta2=0,
                theta3=-0.14387037j,
                theta4=-0.69314718j,
            ),
        ),
    ],
)
def test_standard_to_polar(__from_params: MatchgateStandardParams, __to_params: MatchgatePolarParams):
    to_params = transfer_functions.standard_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params",
    [MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_standard_to_polar_back_and_forth(__from_params: MatchgatePolarParams):
    __from_params = MatchgateStandardParams.parse_from_any(__from_params)
    to_params = transfer_functions.standard_to_polar(__from_params)
    _from_params = transfer_functions.polar_to_standard(to_params)
    assert _from_params == __from_params


def test_standard_to_polar_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, MatchgateStandardParams.N_PARAMS, device="cpu", requires_grad=True)
    __from_params = MatchgateStandardParams(rn_tensor)
    assert isinstance(__from_params.to_tensor(), torch.Tensor)
    assert __from_params.to_tensor().requires_grad
    assert __from_params.requires_grad

    to_params = transfer_functions.standard_to_polar(__from_params)
    assert isinstance(to_params.to_tensor(), torch.Tensor)
    assert to_params.to_tensor().requires_grad
    assert to_params.requires_grad


@pytest.mark.parametrize(
    "params,interface",
    [
        (MatchgateStandardParams.random(), interface)
        for interface in ["numpy", "torch"]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_standard_to_polar_interface(params, interface: Literal["numpy", "torch"]):
    if interface == "torch":
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed.")
    from_params = params.to_interface(interface)
    to_params = transfer_functions.standard_to_polar(from_params)
    to_params_interface = qml.math.get_interface(to_params.to_vector())
    assert to_params_interface == interface, f"Interface {to_params_interface} is not equal to {interface}."
