import numpy as np
import pytest

from matchcake.matchgate_parameter_sets import transfer_functions
from matchcake.matchgate_parameter_sets.transfer_functions import (
    _NODE_ORDER, MatchgateComposedHamiltonianParams,
    MatchgateHamiltonianCoefficientsParams, MatchgatePolarParams,
    all_pairs_dijkstra_commutative_paths, infer_transfer_func, params_to)

from ...configs import (ATOL_APPROX_COMPARISON, ATOL_MATRIX_COMPARISON,
                        N_RANDOM_TESTS_PER_CASE, RTOL_APPROX_COMPARISON,
                        RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed)

MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (
            _NODE_ORDER[_from_cls_idx],
            _NODE_ORDER[_from_cls_idx].random(),
            _NODE_ORDER[_to_cls_idx],
        )
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
    np.testing.assert_allclose(
        _from_params.to_numpy(),
        __from_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg=f"Transfer function from {__from_cls.get_short_name()} -> {__to_cls.get_short_name()} failed.",
    )


@pytest.mark.parametrize(
    "__from_cls,__from_params,__to_cls",
    [
        (
            _NODE_ORDER[_from_cls_idx],
            _NODE_ORDER[_from_cls_idx].random(),
            _NODE_ORDER[_to_cls_idx],
        )
        for _from_cls_idx, path in all_pairs_dijkstra_commutative_paths.items()
        for _to_cls_idx in path
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_params_to_back_and_forth(__from_cls, __from_params, __to_cls):
    to_params = params_to(__from_params, __to_cls)
    _from_params = params_to(to_params, __from_cls)
    np.testing.assert_allclose(
        _from_params.to_numpy(),
        __from_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg=f"Transfer function from {__from_cls.get_short_name()} -> {__to_cls.get_short_name()} failed.",
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0,
            r1=0,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_polar_standard_back_and_forth_case1(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0,
            r1=1,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_polar_standard_back_and_forth_case2(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=0,
            r1=np.random.uniform(*dist),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ],
)
def test_polar_standard_back_and_forth_case3(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1,
            r1=0,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_polar_standard_back_and_forth_case4(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1,
            r1=1,
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_polar_standard_back_and_forth_case5(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=1,
            r1=np.random.uniform(*dist),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ],
)
def test_polar_standard_back_and_forth_case6(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "__from_params",
    [
        MatchgatePolarParams(
            r0=np.random.uniform(*dist0),
            r1=np.random.uniform(*dist1),
            theta0=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta1=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta2=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta3=np.random.uniform(-2 * np.pi, 2 * np.pi),
            theta4=np.random.uniform(-2 * np.pi, 2 * np.pi),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for dist0 in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
        for dist1 in [(1e-3, 1 - 1e-3), (-1e12, -1e-3), (1 + 1e-3, 1e12)]
    ],
)
def test_polar_standard_back_and_forth_case7(__from_params: MatchgatePolarParams):
    to_params = transfer_functions.polar_to_standard(__from_params)
    _from_params = transfer_functions.standard_to_polar(to_params)
    _to_params = transfer_functions.polar_to_standard(_from_params)
    np.testing.assert_allclose(
        to_params.to_numpy(),
        _to_params.to_numpy(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
