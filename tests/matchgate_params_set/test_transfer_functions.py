import pytest
import numpy as np
from msim.matchgate_parameter_sets import transfer_functions
from msim.matchgate_parameter_sets.transfer_functions import (
    _transfer_funcs_by_type,
    params_to,
    MatchgateParams,
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from msim.utils import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I,
)
from ..configs import N_RANDOM_TESTS_PER_CASE
MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

np.random.seed(42)


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (_from_cls, _from_cls.random(), _to_cls)
        for _from_cls, _to_cls_dict in _transfer_funcs_by_type.items()
        for _to_cls in _to_cls_dict.keys()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_transfer_functions_back_and_forth(__from_cls, __from_params, __to_cls):
    to_params = _transfer_funcs_by_type[__from_cls][__to_cls](__from_params)
    _from_params = _transfer_funcs_by_type[__to_cls][__from_cls](to_params)
    assert _from_params == __from_params


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (_from_cls, _from_cls.random(), _to_cls)
        for _from_cls, _to_cls_dict in _transfer_funcs_by_type.items()
        for _to_cls in _to_cls_dict.keys()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_params_to_back_and_forth(__from_cls, __from_params, __to_cls):
    to_params = params_to(__from_params, __to_cls)
    _from_params = params_to(to_params, __from_cls)
    assert _from_params == __from_params


@pytest.mark.parametrize(
    "__from_params",
    [
        __cls.random()
        for __cls in _transfer_funcs_by_type.keys()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_identity_params_to(__from_params: MatchgateParams):
    to_params = params_to(__from_params, type(__from_params))
    assert to_params == __from_params, f"Transfer function from {type(__from_params)} to {type(__from_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_cls",
    [
        (_from_cls.random(), _to_cls)
        for _from_cls, _to_cls_dict in _transfer_funcs_by_type.items()
        for _to_cls in [MatchgateComposedHamiltonianParams] + (
            [MatchgatePolarParams] if not MatchgatePolarParams.ALLOW_COMPLEX_PARAMS else []
    )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_params_to_is_real(__from_params: MatchgateParams, __to_cls):
    to_params = params_to(__from_params, __to_cls)
    assert np.all(np.isreal(to_params.to_numpy())), (
        f"Transfer function from {type(__from_params)} to {__to_cls} failed. Must be real."
    )


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_standard_hamiltonian_to_standard(
        __from_params: MatchgateStandardHamiltonianParams,
        __to_params: MatchgateStandardParams
):
    to_params = transfer_functions.standard_hamiltonian_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


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
    ],
)
def test_polar_to_standard(__from_params: MatchgatePolarParams, __to_params: MatchgateStandardParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


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
        __to_params: MatchgateStandardParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_composed_hamiltonian_to_standard(
        __from_params: MatchgateComposedHamiltonianParams,
        __to_params: MatchgateStandardParams
):
    to_params = transfer_functions.composed_hamiltonian_to_standard(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0)
        ),
        (
            MatchgateStandardParams(
                a=PAULI_Z[0, 0], b=PAULI_Z[0, 1], c=PAULI_Z[1, 0], d=PAULI_Z[1, 1],
                w=PAULI_X[0, 0], x=PAULI_X[0, 1], y=PAULI_X[1, 0], z=PAULI_X[1, 1]
            ),  # fSWAP
            MatchgatePolarParams(r0=1, r1=0, theta0=0, theta1=0, theta2=np.pi / 2, theta3=0, theta4=np.pi / 2)
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=0, x=1, y=1, z=1),
            MatchgatePolarParams(r0=0, r1=0, theta0=0, theta1=0, theta2=np.pi / 2, theta3=0, theta4=np.pi / 2)
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=1, x=1, y=1, z=1),
            MatchgatePolarParams(r0=0, r1=1, theta0=0, theta1=0, theta2=0, theta3=0, theta4=0)
        ),
        (
            MatchgateStandardParams(a=0, b=1, c=1, d=1, w=0.5, x=1, y=1, z=1),
            MatchgatePolarParams(r0=0, r1=0.5, theta0=0, theta1=0, theta2=0, theta3=-0.14387037j, theta4=-0.69314718j)
        ),
    ],
)
def test_standard_to_polar(__from_params: MatchgateStandardParams, __to_params: MatchgatePolarParams):
    to_params = transfer_functions.standard_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_standard_to_polar_back_and_forth(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    assert _from_params == __from_params


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_composed_hamiltonian_to_polar(
        __from_params: MatchgateComposedHamiltonianParams,
        __to_params: MatchgatePolarParams
):
    to_params = transfer_functions.composed_hamiltonian_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_hamiltonian_coefficients_to_polar(
        __from_params: MatchgateHamiltonianCoefficientsParams,
        __to_params: MatchgatePolarParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0)
        ),
    ],
)
def test_standard_hamiltonian_to_polar(
        __from_params: MatchgateStandardHamiltonianParams,
        __to_params: MatchgatePolarParams
):
    to_params = transfer_functions.standard_hamiltonian_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_standard_to_standard_hamiltonian(
        __from_params: MatchgateStandardParams,
        __to_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.standard_to_standard_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_polar_to_standard_hamiltonian(
        __from_params: MatchgatePolarParams,
        __to_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.polar_to_standard_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_hamiltonian_coefficients_to_standard_hamiltonian(
        __from_params: MatchgateHamiltonianCoefficientsParams,
        __to_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_standard_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_composed_hamiltonian_to_standard_hamiltonian(
        __from_params: MatchgateComposedHamiltonianParams,
        __to_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.composed_hamiltonian_to_standard_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_polar_to_hamiltonian_coefficients(
        __from_params: MatchgatePolarParams,
        __to_params: MatchgateHamiltonianCoefficientsParams
):
    to_params = transfer_functions.polar_to_hamiltonian_coefficients(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=2.0, h1=2.0, h2=0.0, h3=2.0, h4=0.0, h5=0.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=2.0, m_y=2.0, m_z=2.0),
            MatchgateHamiltonianCoefficientsParams(h0=3.0, h1=3.0, h2=-1.0, h3=3.0, h4=-1.0, h5=-1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=2.0, n_y=2.0, n_z=2.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=3.0, h1=3.0, h2=1.0, h3=3.0, h4=1.0, h5=1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=-1.0, h3=1.0, h4=-1.0, h5=-1.0),
        ),
    ],
)
def test_composed_hamiltonian_to_hamiltonian_coefficients(
        __from_params: MatchgateComposedHamiltonianParams,
        __to_params: MatchgateHamiltonianCoefficientsParams
):
    to_params = transfer_functions.composed_hamiltonian_to_hamiltonian_coefficients(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_standard_to_hamiltonian_coefficients(
        __from_params: MatchgateStandardParams,
        __to_params: MatchgateHamiltonianCoefficientsParams
):
    to_params = transfer_functions.standard_to_hamiltonian_coefficients(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_standard_hamiltonian_to_hamiltonian_coefficients(
        __from_params: MatchgateStandardHamiltonianParams,
        __to_params: MatchgateHamiltonianCoefficientsParams
):
    to_params = transfer_functions.standard_hamiltonian_to_hamiltonian_coefficients(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_standard_to_composed_hamiltonian(
        __from_params: MatchgateStandardParams,
        __to_params: MatchgateComposedHamiltonianParams
):
    to_params = transfer_functions.standard_to_composed_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_polar_to_composed_hamiltonian(
        __from_params: MatchgatePolarParams,
        __to_params: MatchgateComposedHamiltonianParams
):
    to_params = transfer_functions.polar_to_composed_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
        (
            MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
        (
            MatchgateHamiltonianCoefficientsParams(h0=-1.0, h1=-1.0, h2=-1.0, h3=1.0, h4=1.0, h5=1.0),
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=-1.0, m_z=-1.0),
        ),
    ],
)
def test_hamiltonian_coefficients_to_composed_hamiltonian(
        __from_params: MatchgateHamiltonianCoefficientsParams,
        __to_params: MatchgateComposedHamiltonianParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_composed_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params,__to_params",

    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_standard_hamiltonian_to_composed_hamiltonian(
        __from_params: MatchgateStandardHamiltonianParams,
        __to_params: MatchgateComposedHamiltonianParams
):
    to_params = transfer_functions.standard_hamiltonian_to_composed_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


