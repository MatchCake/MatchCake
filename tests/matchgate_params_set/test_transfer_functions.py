import numpy as np
import pytest

from msim.matchgate_parameter_sets import transfer_functions, fSWAP
from msim.matchgate_parameter_sets.transfer_functions import (
    _transfer_funcs_by_type,
    _NODE_ORDER,
    infer_transfer_func,
    all_pairs_dijkstra_commutative_paths,
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
    PAULI_Z,
)
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED, ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON,
)

MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (_NODE_ORDER[_from_cls_idx], _NODE_ORDER[_from_cls_idx].random(), _NODE_ORDER[_to_cls_idx])
        for _from_cls_idx, path in all_pairs_dijkstra_commutative_paths.items()
        for _to_cls_idx in path
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_infer_transfer_func_back_and_forth(__from_cls, __from_params, __to_cls):
    forward = infer_transfer_func(__from_cls, __to_cls)
    backward = infer_transfer_func(__to_cls, __from_cls)
    to_params = forward(__from_params)
    _from_params = backward(to_params)
    assert _from_params == __from_params, (
        f"Transfer function from {__from_cls.get_short_name()} -> {__to_cls.get_short_name()} failed."
    )


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (_NODE_ORDER[_from_cls_idx], _NODE_ORDER[_from_cls_idx].random(), _NODE_ORDER[_to_cls_idx])
        for _from_cls_idx, path in all_pairs_dijkstra_commutative_paths.items()
        for _to_cls_idx in path
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_params_to_back_and_forth(__from_cls, __from_params, __to_cls):
    to_params = params_to(__from_params, __to_cls)
    _from_params = params_to(to_params, __from_cls)
    assert _from_params == __from_params, (
        f"Transfer function from {__from_cls.get_short_name()} -> {__to_cls.get_short_name()} failed."
    )


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
        for _to_cls in (
                               [MatchgatePolarParams] if not MatchgatePolarParams.ALLOW_COMPLEX_PARAMS else []
                       ) + (
                               [MatchgateComposedHamiltonianParams]
                               if not MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS
                               else []
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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
        (
                MatchgateStandardHamiltonianParams(u2=np.pi / 2, u3=-np.pi / 2, u4=-np.pi / 2, u5=np.pi / 2, u7=np.pi),
                MatchgateStandardParams(a=1, b=0, c=0, d=-1, w=0, x=1, y=1, z=0),
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


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_standard_to_polar_back_and_forth(__from_params: MatchgatePolarParams):
    __from_params = MatchgateStandardParams.parse_from_any(__from_params)
    to_params = transfer_functions.standard_to_polar(__from_params)
    _from_params = transfer_functions.polar_to_standard(to_params)
    assert _from_params == __from_params


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
                MatchgatePolarParams(
                    r0=0, r1=0.5, theta0=0, theta1=0, theta2=0, theta3=-0.14387037j, theta4=-0.69314718j
                    )
        ),
    ],
)
def test_standard_to_polar(__from_params: MatchgateStandardParams, __to_params: MatchgatePolarParams):
    to_params = transfer_functions.standard_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0, r1=0,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_polar_standard_back_and_forth_case1(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0, r1=1,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_polar_standard_back_and_forth_case2(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0, r1=np.random.uniform(*dist),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ]
)
def test_polar_standard_back_and_forth_case3(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1, r1=0,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_polar_standard_back_and_forth_case4(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1, r1=1,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_polar_standard_back_and_forth_case5(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1, r1=np.random.uniform(*dist),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ]
)
def test_polar_standard_back_and_forth_case6(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=np.random.uniform(*dist0), r1=np.random.uniform(*dist1),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist0 in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
        for dist1 in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ]
)
def test_polar_standard_back_and_forth_case7(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    assert to_params == _to_params


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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
        ),
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=-1, w=0, x=1, y=1, z=0),
                MatchgateStandardHamiltonianParams(u2=np.pi / 2, u3=-np.pi / 2, u4=-np.pi / 2, u5=np.pi / 2, u7=np.pi),
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
    "__from_params",
    [
        MatchgateStandardParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_standard_to_standard_hamiltonian_back_and_forth(
        __from_params: MatchgateStandardParams,
):
    to_params = transfer_functions.standard_to_standard_hamiltonian(__from_params)
    _from_params = transfer_functions.standard_hamiltonian_to_standard(to_params)
    np.testing.assert_allclose(
        _from_params.to_numpy(),
        __from_params.to_numpy(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    assert _from_params == __from_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgateComposedHamiltonianParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_standard_hamiltonian_to_standard_back_and_forth(
        __from_params: MatchgateStandardHamiltonianParams,
):
    __from_params = MatchgateStandardHamiltonianParams.parse_from_params(__from_params)
    to_params = transfer_functions.standard_hamiltonian_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_standard_hamiltonian(to_params)
    _to_params = transfer_functions.standard_hamiltonian_to_standard(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
        ),
        # (  TODO: has to be re-computed by hand
        #     MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        #     MatchgateStandardHamiltonianParams(h0=4j, h1=4j, h2=0, h3=-4, h4=4, h5=0, h6=4j, h7=-4j),
        # ),
        # ( TODO: has to be re-computed by hand
        #         MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=2.0, h2=3.0, h3=4.0, h4=5.0, h5=6.0),
        #         MatchgateStandardHamiltonianParams(
        #             h0=14j, h1=6+14j, h2=-10j, h3=2j-14, h4=14+2j, h5=10j, h6=-6+14j, h7=-14j
        #         ),
        # ),
    ],
)
def test_hamiltonian_coefficients_to_standard_hamiltonian(
        __from_params: MatchgateHamiltonianCoefficientsParams,
        __to_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_standard_hamiltonian(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgateHamiltonianCoefficientsParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_hamiltonian_coefficients_to_standard_hamiltonian_back_and_forth(
        __from_params: MatchgateHamiltonianCoefficientsParams
):
    to_params = transfer_functions.hamiltonian_coefficients_to_standard_hamiltonian(__from_params)
    _from_params = transfer_functions.standard_hamiltonian_to_hamiltonian_coefficients(to_params)
    assert _from_params == __from_params


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgateStandardHamiltonianParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_standard_hamiltonian_to_hamiltonian_coefficients_back_and_forth(
        __from_params: MatchgateStandardHamiltonianParams
):
    to_params = transfer_functions.standard_hamiltonian_to_hamiltonian_coefficients(__from_params)
    _from_params = transfer_functions.hamiltonian_coefficients_to_standard_hamiltonian(to_params)
    _to_params = transfer_functions.standard_hamiltonian_to_hamiltonian_coefficients(_from_params)
    assert to_params == _to_params


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
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
                MatchgateHamiltonianCoefficientsParams(h0=0.5, h1=0.5, h2=0.0, h3=0.5, h4=0.0, h5=0.0),
        ),
        (
                MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=2.0, m_y=2.0, m_z=2.0),
                MatchgateHamiltonianCoefficientsParams(
                    h0=3.0 / 4, h1=3.0 / 4, h2=-1.0 / 4, h3=3.0 / 4, h4=-1.0 / 4, h5=-1.0 / 4
                    ),
        ),
        (
                MatchgateComposedHamiltonianParams(n_x=2.0, n_y=2.0, n_z=2.0, m_x=1.0, m_y=1.0, m_z=1.0),
                MatchgateHamiltonianCoefficientsParams(
                    h0=3.0 / 4, h1=3.0 / 4, h2=1.0 / 4, h3=3.0 / 4, h4=1.0 / 4, h5=1.0 / 4
                    ),
        ),
        (
                MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgateHamiltonianCoefficientsParams(
                    h0=1.0 / 4, h1=1.0 / 4, h2=1.0 / 4, h3=1.0 / 4, h4=1.0 / 4, h5=1.0 / 4
                    ),
        ),
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=1.0, m_z=1.0),
                MatchgateHamiltonianCoefficientsParams(
                    h0=1.0 / 4, h1=1.0 / 4, h2=-1.0 / 4, h3=1.0 / 4, h4=-1.0 / 4, h5=-1.0 / 4
                    ),
        ),
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=1.0, m_z=1.0, epsilon=10),
                MatchgateHamiltonianCoefficientsParams(
                    h0=1.0 / 4, h1=1.0 / 4, h2=-1.0 / 4, h3=1.0 / 4, h4=-1.0 / 4, h5=-1.0 / 4, epsilon=10
                ),
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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
        # (TODO: has to be re-computed by hand
        #     MatchgateStandardHamiltonianParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0, h6=1.0, h7=1.0),
        #     MatchgateHamiltonianCoefficientsParams(h0=-0.5j, h1=0, h2=0, h3=-0.5j, h4=0, h5=0),
        # ),
        # (TODO: has to be re-computed by hand
        #     MatchgateStandardHamiltonianParams(h0=1.0, h1=2.0, h2=3.0, h3=4.0, h4=5.0, h5=6.0, h6=7.0, h7=8.0),
        #     MatchgateHamiltonianCoefficientsParams(h0=-1j, h1=6/8, h2=0, h3=-18j/8, h4=-0.5, h5=0.5j),
        # )
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
        # (TODO: has to be re-computed by hand
        #     MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        #     MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=0.0, m_y=0.0, m_z=0.0),
        # ),
        # (TODO: has to be re-computed by hand
        #     MatchgateHamiltonianCoefficientsParams(h0=-1.0, h1=-1.0, h2=-1.0, h3=1.0, h4=1.0, h5=1.0),
        #     MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=-1.0, m_z=-1.0),
        # ),
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
                MatchgateStandardHamiltonianParams(u0=0.0, u1=0.0, u2=0.0, u3=0.0, u4=0.0, u5=0.0, u6=0.0, u7=0.0),
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


@pytest.mark.parametrize(
    "__from_params,__to_params",
    [
        (
                MatchgateStandardParams(
                    a=[1, fSWAP.a, 0, 0],
                    b=[0, fSWAP.b, 1, 1],
                    c=[0, fSWAP.c, 1, 1],
                    d=[1, fSWAP.d, 1, 1],
                    w=[1, fSWAP.w, 1, 0.5],
                    x=[0, fSWAP.x, 1, 1],
                    y=[0, fSWAP.y, 1, 1],
                    z=[1, fSWAP.z, 1, 1],
                ),
                MatchgatePolarParams(
                    r0=[1, 1, 0, 0],
                    r1=[1, 0, 1, 0.5],
                    theta0=[0, 0, 0, 0],
                    theta1=[0, 0, 0, 0],
                    theta2=[0, np.pi / 2, 0, 0],
                    theta3=[0, 0, 0, -0.14387037j],
                    theta4=[0, np.pi / 2, 0, -0.69314718j],
                )
        ),
    ],
)
def test_standard_to_polar_in_batch(__from_params: MatchgateStandardParams, __to_params: MatchgatePolarParams):
    for from_vector, to_vector in zip(__from_params.to_numpy(), __to_params.to_numpy()):
        _v_from_params = MatchgateStandardParams(from_vector)
        _v_to_params = MatchgatePolarParams(to_vector)
        to_params = transfer_functions.standard_to_polar(_v_from_params)
        assert _v_to_params == to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."

    to_params = transfer_functions.standard_to_polar(__from_params)
    assert to_params == __to_params, f"Transfer function from {type(__from_params)} to {type(__to_params)} failed."
