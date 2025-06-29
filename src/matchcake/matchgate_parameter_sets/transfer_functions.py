from typing import Callable, List, Type

import networkx as nx
import numpy as np
import pennylane as qml

from .. import utils
from ..utils import math
from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import (
    MatchgateHamiltonianCoefficientsParams,
)
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams


def polar_to_standard(params: MatchgatePolarParams, **kwargs) -> MatchgateStandardParams:
    r"""
    Convert from polar to standard parameterization. The conversion is given by

    .. math::
        \begin{align}
            a &= r_0 e^{i\theta_0} \\
            b &= (\sqrt{1 - r_0^2}) e^{i(\theta_2+\theta_4-(\theta_1+\pi))} \\
            c &= (\sqrt{1 - r_0^2}) e^{i\theta_1} \\
            d &= r_0 e^{i(\theta_2+\theta_4-\theta_0)} \\
            w &= r_1 e^{i\theta_2} \\
            x &= (\sqrt{1 - r_1^2}) e^{i(\theta_2+\theta_4-(\theta_3+\pi))} \\
            y &= (\sqrt{1 - r_1^2}) e^{i\theta_3} \\
            z &= r_1 e^{i\theta_4}
        \end{align}


    :param params: The polar parameters
    :type params: MatchgatePolarParams
    :param kwargs: Additional keyword arguments

    :keyword backend: The backend to use for the computation

    :return: The standard parameters
    :rtype: MatchgateStandardParams
    """
    backend = MatchgateParams.load_backend_lib(kwargs.pop("backend", qml.math))

    r0_tilde = MatchgatePolarParams.compute_r_tilde(params.r0, backend=backend)
    # r0_tilde = backend.ones_like(params.r0)
    r1_tilde = MatchgatePolarParams.compute_r_tilde(params.r1, backend=backend)
    # r1_tilde = backend.ones_like(params.r1)
    # return MatchgateStandardParams(
    #     a=params.r0 * backend.exp(1j * params.theta0),
    #     b=r0_tilde * backend.exp(1j * (params.theta2 + params.theta4 - (params.theta1 + np.pi))),
    #     c=r0_tilde * backend.exp(1j * params.theta1),
    #     d=params.r0 * backend.exp(1j * (params.theta2 + params.theta4 - params.theta0)),
    #     w=params.r1 * backend.exp(1j * params.theta2),
    #     x=r1_tilde * backend.exp(1j * (params.theta2 + params.theta4 - (params.theta3 + np.pi))),
    #     y=r1_tilde * backend.exp(1j * params.theta3),
    #     z=params.r1 * backend.exp(1j * params.theta4),
    # )
    return MatchgateStandardParams(
        a=params.r0 * math.exp_euler(params.theta0),
        b=r0_tilde * math.exp_euler(params.theta2 + params.theta4 - (params.theta1 + np.pi)),
        c=r0_tilde * math.exp_euler(params.theta1),
        d=params.r0 * math.exp_euler(params.theta2 + params.theta4 - params.theta0),
        w=params.r1 * math.exp_euler(params.theta2),
        x=r1_tilde * math.exp_euler(params.theta2 + params.theta4 - (params.theta3 + np.pi)),
        y=r1_tilde * math.exp_euler(params.theta3),
        z=params.r1 * math.exp_euler(params.theta4),
    )


def standard_to_standard_hamiltonian(params: MatchgateStandardParams, **kwargs) -> MatchgateStandardHamiltonianParams:
    gate = qml.math.cast(params.to_matrix(), dtype=complex)
    hamiltonian = -1j * math.logm(gate)
    return MatchgateStandardHamiltonianParams.from_matrix(hamiltonian, **kwargs)


def standard_hamiltonian_to_hamiltonian_coefficients(
    params: MatchgateStandardHamiltonianParams, **kwargs
) -> MatchgateHamiltonianCoefficientsParams:
    r"""
    Convert from standard hamiltonian to hamiltonian coefficients parameterization. The conversion is given by

    .. math::
        \begin{align}
            h_0 &= -\frac{i}{2} (\hat{h}_0 + \hat{h}_2), \\
            h_1 &= \frac{1}{4} (\hat{h}_6 - \hat{h}_1 + \hat{h}_4 - \hat{h}_3), \\
            h_2 &= \frac{i}{4} (\hat{h}_3 + \hat{h}_4 - \hat{h}_1 - \hat{h}_6), \\
            h_3 &= -\frac{i}{4} (\hat{h}_3 + \hat{h}_4 + \hat{h}_1 + \hat{h}_6), \\
            h_4 &= \frac{1}{4} (\hat{h}_1 - \hat{h}_6 + \hat{h}_4 - \hat{h}_3), \\
            h_5 &= \frac{i}{2} (\hat{h}_2 - \hat{h}_0).
        \end{align}


    :param params: The standard hamiltonian parameters
    :type params: MatchgateStandardHamiltonianParams
    :param kwargs: Additional keyword arguments

    :return: The hamiltonian coefficients parameters
    :rtype: MatchgateHamiltonianCoefficientsParams
    """
    epsilon = 0.5 * (params.u0 + params.u7)
    return MatchgateHamiltonianCoefficientsParams(
        h0=0.25 * (params.u0 + params.u2 - 2 * epsilon),
        h1=-0.125j * (params.u1 - params.u6 + params.u3 - params.u4),
        h2=0.125 * (params.u1 + params.u6 - params.u3 - params.u4),
        h3=0.125 * (params.u1 + params.u6 + params.u3 + params.u4),
        h4=-0.125j * (params.u3 - params.u4 - params.u1 + params.u6),
        h5=0.25 * (params.u0 - params.u2),
        epsilon=epsilon,
    )


def hamiltonian_coefficients_to_composed_hamiltonian(
    params: MatchgateHamiltonianCoefficientsParams, **kwargs
) -> MatchgateComposedHamiltonianParams:
    r"""
    Convert from hamiltonian coefficients to composed hamiltonian parameterization. The conversion is given by

    .. math::
        \begin{align}
            n_x &= \frac{1}{2} (h_2 + h_3), \\
            n_y &= \frac{1}{2} (h_1 + h_4), \\
            n_z &= \frac{1}{2} (h_0 + h_5), \\
            m_x &= \frac{1}{2} (h_3 - h_2), \\
            m_y &= \frac{1}{2} (h_1 - h_4), \\
            m_z &= \frac{1}{2} (h_0 - h_5).
        \end{align}


    :param params: The hamiltonian coefficients parameters
    :type params: MatchgateHamiltonianCoefficientsParams
    :param kwargs: Additional keyword arguments

    :return: The composed hamiltonian parameters
    :rtype: MatchgateComposedHamiltonianParams
    """
    return MatchgateComposedHamiltonianParams(
        n_x=2 * (params.h2 + params.h3),
        n_y=2 * (params.h1 + params.h4),
        n_z=2 * (params.h0 + params.h5),
        m_x=2 * (params.h3 - params.h2),
        m_y=2 * (params.h1 - params.h4),
        m_z=2 * (params.h0 - params.h5),
        epsilon=params.epsilon,
    )


def composed_hamiltonian_to_hamiltonian_coefficients(
    params: MatchgateComposedHamiltonianParams, **kwargs
) -> MatchgateHamiltonianCoefficientsParams:
    r"""
    Convert from composed hamiltonian to hamiltonian coefficients parameterization. The conversion is given by

    .. math::
        \begin{align}
            h_0 &= n_z + m_z \\
            h_1 &= n_y + m_y \\
            h_2 &= n_x - m_x \\
            h_3 &= n_x + m_x \\
            h_4 &= n_y - m_y \\
            h_5 &= n_z - m_z
        \end{align}

    :param params: The composed hamiltonian parameters
    :type params: MatchgateComposedHamiltonianParams
    :param kwargs: Additional keyword arguments

    :return: The hamiltonian coefficients parameters
    :rtype: MatchgateHamiltonianCoefficientsParams
    """
    return MatchgateHamiltonianCoefficientsParams(
        h0=0.25 * (params.n_z + params.m_z),
        h1=0.25 * (params.n_y + params.m_y),
        h2=0.25 * (params.n_x - params.m_x),
        h3=0.25 * (params.n_x + params.m_x),
        h4=0.25 * (params.n_y - params.m_y),
        h5=0.25 * (params.n_z - params.m_z),
        epsilon=params.epsilon,
    )


def hamiltonian_coefficients_to_standard_hamiltonian(
    params: MatchgateHamiltonianCoefficientsParams, **kwargs
) -> MatchgateStandardHamiltonianParams:
    r"""
    Convert from hamiltonian coefficients to standard hamiltonian parameterization. The conversion is given by

    .. math::
        \begin{align}
            \hat{h}_0 &= 2i (h_0 + h_5) + \epsilon \\
            \hat{h}_1 &= 2(h_4 - h_1) + 2i(h_2 + h_3) \\
            \hat{h}_2 &= 2i(h_0 - h_5) + \epsilon \\
            \hat{h}_3 &= 2i(h_3 - h_2) - 2(h_1 + h_4) \\
            \hat{h}_4 &= 2(h_1 + h_4) + 2i(h_3 - h_2) \\
            \hat{h}_5 &= 2i (h_5 - h_0) + \epsilon \\
            \hat{h}_6 &= 2(h_1 - h_4) + 2i(h_2 + h_3) \\
            \hat{h}_7 &= -2i (h_0 + h_5) + \epsilon
        \end{align}

    :param params: The hamiltonian coefficients parameters
    :type params: MatchgateHamiltonianCoefficientsParams
    :param kwargs: Additional keyword arguments

    :return: The standard hamiltonian parameters
    :rtype: MatchgateStandardHamiltonianParams
    """
    # backend = MatchgateParams.load_backend_lib(kwargs.pop("backend", pnp))
    # hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(params.to_matrix())
    # hamiltonian = params.compute_hamiltonian()
    # elements_indexes_as_array = backend.array(MatchgateStandardHamiltonianParams.ELEMENTS_INDEXES)
    # params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    # return MatchgateStandardHamiltonianParams.from_numpy(params_arr)
    return MatchgateStandardHamiltonianParams(
        u0=2 * (params.h0 + params.h5) + params.epsilon,
        u1=2 * (params.h3 + params.h2) + 2j * (params.h1 - params.h4),
        u2=2 * (params.h0 - params.h5) + params.epsilon,
        u3=2 * (params.h3 - params.h2) + 2j * (params.h1 + params.h4),
        u4=2 * (params.h3 - params.h2) - 2j * (params.h1 + params.h4),
        u5=2 * (params.h5 - params.h0) + params.epsilon,
        u6=2 * (params.h3 + params.h2) - 2j * (params.h1 - params.h4),
        u7=-2 * (params.h0 + params.h5) + params.epsilon,
    )


def standard_hamiltonian_to_standard(params: MatchgateStandardHamiltonianParams, **kwargs) -> MatchgateStandardParams:
    gate = utils.get_unitary_from_hermitian_matrix(params.to_matrix())
    return MatchgateStandardParams.from_matrix(gate, **kwargs)


def _compute_r0_from_standard(a, backend):
    return backend.sqrt(a * backend.conjugate(a))


def _compute_r1_from_standard(w, backend):
    return backend.sqrt(w * backend.conjugate(w))


def _compute_theta0_from_standard(a, r0, r0_is_zero, r0_is_one, zero_like, backend, eps=1e-12):
    theta0 = backend.where(r0_is_zero, zero_like, -1j * (backend.log(a + eps) - backend.log(r0 + eps)))
    theta0 = backend.where(r0_is_one, -1j * backend.log(a + eps), theta0)
    return theta0


def _compute_theta1_from_standard(c, r0_tilde, r0_is_zero, r0_is_one, zero_like, backend, eps=1e-12):
    theta1 = backend.where(
        r0_is_zero,
        -1j * backend.log(c + eps),
        -1j * (backend.log(c + eps) - backend.log(r0_tilde + eps)),
    )
    theta1 = backend.where(r0_is_one, 0, theta1)
    return theta1


def _compute_theta2_from_standard(a, b, c, d, w, r1, r1_is_zero, r1_is_one, r0_is_one, backend, eps=1e-12):
    theta2 = backend.where(
        r1_is_zero,
        -0.5j * backend.log(-b * c + eps),
        -1j * (backend.log(w + eps) - backend.log(r1 + eps)),
    )
    theta2 = backend.where(r1_is_one, -1j * backend.log(w + eps), theta2)
    theta2 = backend.where(r0_is_one & r1_is_zero, -0.5j * backend.log(d * a + eps), theta2)
    return theta2


def _compute_theta3_from_standard(y, r1_is_zero, r1_is_one, r1_tilde, backend, eps=1e-12):
    theta3 = backend.where(
        r1_is_zero,
        -1j * backend.log(y + eps),
        -1j * (backend.log(y + eps) - backend.log(r1_tilde + eps)),
    )
    theta3 = backend.where(r1_is_one, 0, theta3)
    return theta3


def _compute_theta4_from_standard(a, b, c, d, z, r1, r1_is_zero, r1_is_one, r0_is_one, backend, eps=1e-12):
    theta4 = backend.where(
        r1_is_zero,
        -0.5j * backend.log(-b * c + eps),
        -1j * (backend.log(z + eps) - backend.log(r1 + eps)),
    )
    theta4 = backend.where(r1_is_one, -1j * backend.log(z + eps), theta4)
    theta4 = backend.where(r0_is_one & r1_is_zero, -0.5j * backend.log(d * a + eps), theta4)
    return theta4


def standard_to_polar(params: MatchgateStandardParams, **kwargs) -> MatchgatePolarParams:
    backend = MatchgateParams.load_backend_lib(kwargs.pop("backend", qml.math))
    eps = kwargs.pop("eps", 1e-12)
    params_arr = params.to_vector()
    a, b, c, d, w, x, y, z = qml.math.cast(
        qml.math.transpose(qml.math.reshape(params_arr, (-1, params.N_PARAMS))),
        dtype=complex,
    )
    r0 = _compute_r0_from_standard(a, backend=backend)
    r1 = _compute_r1_from_standard(w, backend=backend)
    r0_tilde = MatchgatePolarParams.compute_r_tilde(r0, backend=backend)
    r1_tilde = MatchgatePolarParams.compute_r_tilde(r1, backend=backend)

    zero_like, one_like = utils.math.convert_and_cast_like(0, r0), utils.math.convert_and_cast_like(1, r0)
    r0_is_zero, r0_is_one = backend.isclose(r0, zero_like), backend.isclose(r0, one_like)
    r1_is_zero, r1_is_one = backend.isclose(r1, zero_like), backend.isclose(r1, one_like)

    theta0 = _compute_theta0_from_standard(a, r0, r0_is_zero, r0_is_one, zero_like, backend, eps=eps)
    theta1 = _compute_theta1_from_standard(c, r0_tilde, r0_is_zero, r0_is_one, zero_like, backend, eps=eps)
    theta2 = _compute_theta2_from_standard(a, b, c, d, w, r1, r1_is_zero, r1_is_one, r0_is_one, backend, eps=eps)
    theta3 = _compute_theta3_from_standard(y, r1_is_zero, r1_is_one, r1_tilde, backend, eps=eps)
    theta4 = _compute_theta4_from_standard(a, b, c, d, z, r1, r1_is_zero, r1_is_one, r0_is_one, backend, eps=eps)
    return MatchgatePolarParams(
        r0=r0,
        r1=r1,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        theta4=theta4,
        **kwargs,
    )


def identity_transfer(params: MatchgateParams, **kwargs) -> MatchgateParams:
    return params


_classes_transfer_path = [
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateStandardHamiltonianParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
]

_forward_transfer_path = [
    polar_to_standard,
    standard_to_standard_hamiltonian,
    standard_hamiltonian_to_hamiltonian_coefficients,
    hamiltonian_coefficients_to_composed_hamiltonian,
]

_backward_transfer_path = [
    composed_hamiltonian_to_hamiltonian_coefficients,
    hamiltonian_coefficients_to_standard_hamiltonian,
    standard_hamiltonian_to_standard,
    standard_to_polar,
]
_NODE_ORDER = [
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateStandardHamiltonianParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
]
_transfer_adj_matrix = np.asarray(
    [
        # (i->j)
        # polar, standard, standard_hamiltonian, hamiltonian_coefficients, composed_hamiltonian
        [1, 1, 0, 0, 0],  # polar
        [1, 1, 1, 0, 0],  # standard
        [0, 1, 1, 1, 0],  # standard_hamiltonian
        [0, 0, 1, 1, 1],  # hamiltonian_coefficients
        [0, 0, 0, 1, 1],  # composed_hamiltonian
    ]
)
_commutative_transfer_adj_matrix = np.asarray(
    [
        # (i->j)
        # polar, standard, standard_hamiltonian, hamiltonian_coefficients, composed_hamiltonian
        [1, 0, 0, 0, 0],  # polar
        [0, 1, 1, 0, 0],  # standard
        [0, 0, 1, 0, 0],  # standard_hamiltonian
        [0, 0, 1, 1, 1],  # hamiltonian_coefficients
        [0, 0, 0, 1, 1],  # composed_hamiltonian
    ]
)
all_pairs_dijkstra_paths = dict(
    nx.all_pairs_dijkstra_path(nx.from_numpy_array(_transfer_adj_matrix, create_using=nx.DiGraph))
)
all_pairs_dijkstra_commutative_paths = dict(
    nx.all_pairs_dijkstra_path(nx.from_numpy_array(_commutative_transfer_adj_matrix, create_using=nx.DiGraph))
)


def polar_to_standard_hamiltonian(params: MatchgatePolarParams, **kwargs) -> MatchgateStandardHamiltonianParams:
    params = polar_to_standard(params, **kwargs)
    return standard_to_standard_hamiltonian(params, **kwargs)


def polar_to_hamiltonian_coefficients(params: MatchgatePolarParams, **kwargs) -> MatchgateHamiltonianCoefficientsParams:
    params = polar_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)


def polar_to_composed_hamiltonian(params: MatchgatePolarParams, **kwargs) -> MatchgateComposedHamiltonianParams:
    params = polar_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_composed_hamiltonian(params, **kwargs)


def standard_to_hamiltonian_coefficients(
    params: MatchgateStandardParams, **kwargs
) -> MatchgateHamiltonianCoefficientsParams:
    params = standard_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)


def standard_to_composed_hamiltonian(params: MatchgateStandardParams, **kwargs) -> MatchgateComposedHamiltonianParams:
    params = standard_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_composed_hamiltonian(params, **kwargs)


def hamiltonian_coefficients_to_standard(
    params: MatchgateHamiltonianCoefficientsParams, **kwargs
) -> MatchgateStandardParams:
    params = hamiltonian_coefficients_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_standard(params, **kwargs)


def hamiltonian_coefficients_to_polar(params: MatchgateHamiltonianCoefficientsParams, **kwargs) -> MatchgatePolarParams:
    params = hamiltonian_coefficients_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def composed_hamiltonian_to_standard_hamiltonian(
    params: MatchgateComposedHamiltonianParams, **kwargs
) -> MatchgateStandardHamiltonianParams:
    params = composed_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_standard_hamiltonian(params, **kwargs)


def composed_hamiltonian_to_standard(params: MatchgateComposedHamiltonianParams, **kwargs) -> MatchgateStandardParams:
    params = composed_hamiltonian_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_standard(params, **kwargs)


def composed_hamiltonian_to_polar(params: MatchgateComposedHamiltonianParams, **kwargs) -> MatchgatePolarParams:
    params = composed_hamiltonian_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def standard_hamiltonian_to_polar(params: MatchgateStandardHamiltonianParams, **kwargs) -> MatchgatePolarParams:
    params = standard_hamiltonian_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def standard_hamiltonian_to_composed_hamiltonian(
    params: MatchgateStandardHamiltonianParams, **kwargs
) -> MatchgateComposedHamiltonianParams:
    params = standard_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_composed_hamiltonian(params, **kwargs)


_transfer_funcs_by_type = {
    # from              : to
    MatchgatePolarParams: {
        MatchgateStandardParams: polar_to_standard,
        MatchgateHamiltonianCoefficientsParams: polar_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: polar_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: polar_to_standard_hamiltonian,
    },
    MatchgateStandardParams: {
        MatchgatePolarParams: standard_to_polar,
        MatchgateHamiltonianCoefficientsParams: standard_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: standard_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: standard_to_standard_hamiltonian,
    },
    MatchgateHamiltonianCoefficientsParams: {
        MatchgatePolarParams: hamiltonian_coefficients_to_polar,
        MatchgateStandardParams: hamiltonian_coefficients_to_standard,
        MatchgateComposedHamiltonianParams: hamiltonian_coefficients_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: hamiltonian_coefficients_to_standard_hamiltonian,
    },
    MatchgateComposedHamiltonianParams: {
        MatchgatePolarParams: composed_hamiltonian_to_polar,
        MatchgateStandardParams: composed_hamiltonian_to_standard,
        MatchgateHamiltonianCoefficientsParams: composed_hamiltonian_to_hamiltonian_coefficients,
        MatchgateStandardHamiltonianParams: composed_hamiltonian_to_standard_hamiltonian,
    },
    MatchgateStandardHamiltonianParams: {
        MatchgatePolarParams: standard_hamiltonian_to_polar,
        MatchgateStandardParams: standard_hamiltonian_to_standard,
        MatchgateHamiltonianCoefficientsParams: standard_hamiltonian_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: standard_hamiltonian_to_composed_hamiltonian,
    },
}


def infer_transfer_func(from_cls: Type[MatchgateParams], to_cls: Type[MatchgateParams]) -> Callable:
    from_cls_idx = _NODE_ORDER.index(from_cls)
    to_cls_idx = _NODE_ORDER.index(to_cls)
    path = all_pairs_dijkstra_paths[from_cls_idx][to_cls_idx]

    def func(params: MatchgateParams, **kwargs) -> MatchgateParams:
        if len(path) == 1:
            return params
        for i, j in zip(path[:-1], path[1:]):
            params = _transfer_funcs_by_type[_NODE_ORDER[i]][_NODE_ORDER[j]](params, **kwargs)
        return params

    return func


def params_to(params, __cls: Type[MatchgateParams], **kwargs) -> MatchgateParams:
    if isinstance(params, __cls):
        return params
    if not isinstance(params, MatchgateParams):
        return __cls(*params, **kwargs)
    _from_cls = type(params)
    if _from_cls in _transfer_funcs_by_type and __cls in _transfer_funcs_by_type[_from_cls]:
        return _transfer_funcs_by_type[_from_cls][__cls](params, **kwargs)
    return infer_transfer_func(_from_cls, __cls)(params, **kwargs)


def get_closest_cls(cls_list: List[Type[MatchgateParams]], target_cls: Type[MatchgateParams], **kwargs):
    if len(cls_list) == 0:
        raise ValueError("cls_list cannot be empty")
    if len(cls_list) == 1:
        return cls_list[0]
    _target_cls_idx = _NODE_ORDER.index(target_cls)
    _cls_indexes = [_NODE_ORDER.index(cls) for cls in cls_list]
    path_list = [all_pairs_dijkstra_paths[_cls_idx][_target_cls_idx] for _cls_idx in _cls_indexes]
    # pairwise_path_list = [nx.utils.pairwise(path) for path in path_list]
    # cost_list = [
    #     sum(_transfer_adj_matrix[list(path)])
    #     for path in path_list
    # ]
    cost_list = [len(path) for path in path_list]
    min_cost_idx = np.argmin(cost_list)
    return cls_list[min_cost_idx]


def shortest_transfer_to(params_list: List[MatchgateParams], __cls: Type[MatchgateParams], **kwargs):
    if len(params_list) == 0:
        raise ValueError("params_list cannot be empty")
    if len(params_list) == 1:
        return params_to(params_list[0], __cls, **kwargs)
    _cls_list = [type(params) for params in params_list]
    closest_cls = get_closest_cls(_cls_list, __cls, **kwargs)
    _from_params = params_list[_cls_list.index(closest_cls)]
    return params_to(_from_params, __cls, **kwargs)
