from typing import Type, Dict, Callable

from . import (
    MatchgateParams,
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from .. import utils


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
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))

    r0_tilde = MatchgatePolarParams.compute_r_tilde(params.r0, backend=backend)
    r1_tilde = MatchgatePolarParams.compute_r_tilde(params.r1, backend=backend)
    return MatchgateStandardParams(
        a=params.r0 * backend.exp(1j * params.theta0),
        b=r0_tilde * backend.exp(1j * (params.theta2 + params.theta4 - (params.theta1 + backend.pi))),
        c=r0_tilde * backend.exp(1j * params.theta1),
        d=params.r0 * backend.exp(1j * (params.theta2 + params.theta4 - params.theta0)),
        w=params.r1 * backend.exp(1j * params.theta2),
        x=r1_tilde * backend.exp(1j * (params.theta2 + params.theta4 - (params.theta3 + backend.pi))),
        y=r1_tilde * backend.exp(1j * params.theta3),
        z=params.r1 * backend.exp(1j * params.theta4),
        backend=backend,
    )


def standard_to_standard_hamiltonian(params: MatchgateStandardParams, **kwargs) -> MatchgateStandardHamiltonianParams:
    from scipy.linalg import logm

    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))

    gate = params.to_matrix().astype(complex)
    hamiltonian = -1j * logm(gate)
    return MatchgateStandardHamiltonianParams(
        h0=hamiltonian[0, 0],
        h1=hamiltonian[0, 3],
        h2=hamiltonian[1, 1],
        h3=hamiltonian[1, 2],
        h4=hamiltonian[2, 1],
        h5=hamiltonian[2, 2],
        h6=hamiltonian[3, 0],
        h7=hamiltonian[3, 3],
        backend=backend,
    )


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

    :keyword backend: The backend to use for the computation

    :return: The hamiltonian coefficients parameters
    :rtype: MatchgateHamiltonianCoefficientsParams
    """
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    return MatchgateHamiltonianCoefficientsParams(
        h0=-0.25j * (params.h0 + params.h2),
        h1=0.125 * (params.h6 - params.h1 + params.h4 - params.h3),
        h2=0.125j * (params.h3 + params.h4 - params.h1 - params.h6),
        h3=-0.125j * (params.h3 + params.h4 + params.h1 + params.h6),
        h4=0.125 * (params.h1 - params.h6 + params.h4 - params.h3),
        h5=0.25j * (params.h2 - params.h0),
        backend=backend,
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

    :keyword backend: The backend to use for the computation

    :return: The composed hamiltonian parameters
    :rtype: MatchgateComposedHamiltonianParams
    """
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    return MatchgateComposedHamiltonianParams(
        n_x=2 * (params.h2 + params.h3),
        n_y=2 * (params.h1 + params.h4),
        n_z=2 * (params.h0 + params.h5),
        m_x=2 * (params.h3 - params.h2),
        m_y=2 * (params.h1 - params.h4),
        m_z=2 * (params.h0 - params.h5),
        backend=backend,
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

    :keyword backend: The backend to use for the computation

    :return: The hamiltonian coefficients parameters
    :rtype: MatchgateHamiltonianCoefficientsParams
    """
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    return MatchgateHamiltonianCoefficientsParams(
        h0=0.25 * (params.n_z + params.m_z),
        h1=0.25 * (params.n_y + params.m_y),
        h2=0.25 * (params.n_x - params.m_x),
        h3=0.25 * (params.n_x + params.m_x),
        h4=0.25 * (params.n_y - params.m_y),
        h5=0.25 * (params.n_z - params.m_z),
        backend=backend,
    )


def hamiltonian_coefficients_to_standard_hamiltonian(
        params: MatchgateHamiltonianCoefficientsParams, **kwargs
) -> MatchgateStandardHamiltonianParams:
    r"""
    Convert from hamiltonian coefficients to standard hamiltonian parameterization. The conversion is given by

    .. math::
        \begin{align}
            \hat{h}_0 &= 2i (h_0 + h_5) \\
            \hat{h}_1 &= 2(h_4 - h_1) + 2i(h_2 + h_3) \\
            \hat{h}_2 &= 2i(h_0 - h_5) \\
            \hat{h}_3 &= 2i(h_3 - h_2) - 2(h_1 + h_4) \\
            \hat{h}_4 &= 2(h_1 + h_4) + 2i(h_3 - h_2) \\
            \hat{h}_5 &= 2i (h_5 - h_0) \\
            \hat{h}_6 &= 2(h_1 - h_4) + 2i(h_2 + h_3) \\
            \hat{h}_7 &= -2i (h_0 + h_5)
        \end{align}

    :param params: The hamiltonian coefficients parameters
    :type params: MatchgateHamiltonianCoefficientsParams
    :param kwargs: Additional keyword arguments

    :keyword backend: The backend to use for the computation

    :return: The standard hamiltonian parameters
    :rtype: MatchgateStandardHamiltonianParams
    """
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    # hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(params.to_matrix())
    # elements_indexes_as_array = backend.array(MatchgateStandardParams.ELEMENTS_INDEXES)
    # params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    # return MatchgateStandardHamiltonianParams.from_numpy(params_arr)
    return MatchgateStandardHamiltonianParams(
        h0=2j * (params.h0 + params.h5),
        h1=2 * (params.h4 - params.h1) + 2j * (params.h2 + params.h3),
        h2=2j * (params.h0 - params.h5),
        h3=2j * (params.h3 - params.h2) - 2 * (params.h1 + params.h4),
        h4=2 * (params.h1 + params.h4) + 2j * (params.h3 - params.h2),
        h5=2j * (params.h5 - params.h0),
        h6=2 * (params.h1 - params.h4) + 2j * (params.h2 + params.h3),
        h7=-2j * (params.h0 + params.h5),
        backend=backend,
    )


def standard_hamiltonian_to_standard(
        params: MatchgateStandardHamiltonianParams, **kwargs
) -> MatchgateStandardParams:
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    gate = utils.get_unitary_from_hermitian_matrix(params.to_matrix())
    elements_indexes_as_array = backend.array(MatchgateStandardParams.ELEMENTS_INDEXES)
    params_arr = gate[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    return MatchgateStandardParams(*params_arr, backend=backend)


def standard_to_polar(params: MatchgateStandardParams, **kwargs) -> MatchgatePolarParams:
    backend = MatchgateParams.load_backend_lib(kwargs.get("backend", "numpy"))
    a, b, c, d, w, x, y, z = params.to_numpy().astype(complex)
    r0 = backend.sqrt(a * backend.conjugate(a))
    r0_tilde = MatchgatePolarParams.compute_r_tilde(r0, backend=backend)
    r1 = backend.sqrt(w * backend.conjugate(w))
    r1_tilde = MatchgatePolarParams.compute_r_tilde(r1, backend=backend)
    eps = 1e-12
    if backend.isclose(r0, 0) and backend.isclose(r1, 0):
        theta0 = 0
        theta1 = -1j * backend.log(c + eps)
        theta2 = -0.5j * (backend.log(-b + eps) - backend.log(backend.conjugate(c) + eps))
        theta3 = -1j * backend.log(z + eps)
        theta4 = -0.5j * (backend.log(-b + eps) - backend.log(backend.conjugate(c) + eps))
    elif backend.isclose(r0, 0) and backend.isclose(r1, 1):
        theta0 = 0
        theta1 = -1j * backend.log(c + eps)
        theta2 = -1j * backend.log(w + eps)
        theta3 = 0
        theta4 = -1j * backend.log(z + eps)
    elif backend.isclose(r0, 0) and (not backend.isclose(r1, 0) or backend.isclose(r1, 1)):
        theta0 = 0
        theta1 = -1j * backend.log(c + eps)
        theta2 = -1j * (backend.log(w + eps) - backend.log(r1 + eps))
        theta3 = -1j * (backend.log(y + eps) - backend.log(r1_tilde + eps))
        theta4 = -1j * (backend.log(z + eps) - backend.log(r1 + eps))
    elif backend.isclose(r0, 1) and backend.isclose(r1, 0):
        theta0 = -1j * backend.log(a + eps)
        theta1 = 0
        theta2 = -0.5j * (backend.log(d + eps) - backend.log(backend.conjugate(a) + eps))
        theta3 = -1j * backend.log(y + eps)
        theta4 = -0.5j * (backend.log(d + eps) - backend.log(backend.conjugate(a) + eps))
    elif backend.isclose(r0, 1) and backend.isclose(r1, 1):
        theta0 = -1j * backend.log(a + eps)
        theta1 = 0
        theta2 = -1j * backend.log(w + eps)
        theta3 = 0
        theta4 = -1j * backend.log(z + eps)
    elif backend.isclose(r0, 1) and (not backend.isclose(r1, 0) or backend.isclose(r1, 1)):
        theta0 = -1j * backend.log(a + eps)
        theta1 = 0
        theta2 = -1j * (backend.log(w + eps) - backend.log(r1 + eps))
        theta3 = -1j * (backend.log(y + eps) - backend.log(r1_tilde + eps))
        theta4 = -1j * (backend.log(z + eps) - backend.log(r1 + eps))
    elif (not backend.isclose(r0, 0) or backend.isclose(r0, 1)) and (backend.isclose(r1, 0) or backend.isclose(r1, 1)):
        theta0 = -1j * (backend.log(a + eps) - backend.log(r0 + eps))
        theta1 = -1j * (backend.log(c + eps) - backend.log(r0_tilde + eps))
        theta2 = -1j * (backend.log(w + eps) - backend.log(r1 + eps))
        theta3 = -1j * (backend.log(y + eps) - backend.log(r1_tilde + eps))
        theta4 = -1j * (backend.log(z + eps) - backend.log(r1 + eps))
    else:
        theta0 = -1j * (backend.log(a + eps) - backend.log(r0 + eps))
        theta1 = -1j * (backend.log(c + eps) - backend.log(r0_tilde + eps))
        theta2 = -1j * (backend.log(w + eps) - backend.log(r1 + eps))
        theta3 = -1j * (backend.log(y + eps) - backend.log(r1_tilde + eps))
        theta4 = -1j * (backend.log(z + eps) - backend.log(r1 + eps))

    return MatchgatePolarParams(
        r0=r0,
        r1=r1,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        theta4=theta4,
        backend=backend,
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

_transfer_adj_matrix = [
    # (i->j)
    # polar, standard, standard_hamiltonian, hamiltonian_coefficients, composed_hamiltonian
    [identity_transfer, polar_to_standard, None, None, None],  # polar
    [standard_to_polar, identity_transfer, standard_to_standard_hamiltonian, None, None],  # standard
    [
        None, standard_hamiltonian_to_standard, identity_transfer,
        standard_hamiltonian_to_hamiltonian_coefficients, None
    ],  # standard_hamiltonian
    [
        None, None, hamiltonian_coefficients_to_standard_hamiltonian,
        identity_transfer, hamiltonian_coefficients_to_composed_hamiltonian
    ],  # hamiltonian_coefficients
    [None, None, None, composed_hamiltonian_to_hamiltonian_coefficients, identity_transfer],  # composed_hamiltonian
]


def _infer_transfer_func(from_cls: Type[MatchgateParams], to_cls: Type[MatchgateParams]) -> Callable:
    from_cls_idx = _classes_transfer_path.index(from_cls)
    to_cls_idx = _classes_transfer_path.index(to_cls)

    # TODO: find the path from the adjency matrix to transfer from `from_cls` to `to_cls` using dijkstra
    path = utils.dijkstra(_transfer_adj_matrix, from_cls_idx, to_cls_idx)


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
        params: MatchgateStandardParams,
        **kwargs
) -> MatchgateHamiltonianCoefficientsParams:
    params = standard_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)


def standard_to_composed_hamiltonian(params: MatchgateStandardParams, **kwargs) -> MatchgateComposedHamiltonianParams:
    params = standard_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_composed_hamiltonian(params, **kwargs)


def hamiltonian_coefficients_to_standard(
        params: MatchgateHamiltonianCoefficientsParams,
        **kwargs
) -> MatchgateStandardParams:
    params = hamiltonian_coefficients_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_standard(params, **kwargs)


def hamiltonian_coefficients_to_polar(params: MatchgateHamiltonianCoefficientsParams, **kwargs) -> MatchgatePolarParams:
    params = hamiltonian_coefficients_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def composed_hamiltonian_to_standard_hamiltonian(
        params: MatchgateComposedHamiltonianParams,
        **kwargs
) -> MatchgateStandardHamiltonianParams:
    params = composed_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_standard_hamiltonian(params, **kwargs)


def composed_hamiltonian_to_standard(params: MatchgateComposedHamiltonianParams, **kwargs) -> MatchgateStandardParams:
    params = composed_hamiltonian_to_standard_hamiltonian(params, **kwargs)
    return standard_hamiltonian_to_standard(params, **kwargs)


def composed_hamiltonian_to_polar(params: MatchgateComposedHamiltonianParams, **kwargs) -> MatchgatePolarParams:
    params = composed_hamiltonian_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def standard_hamiltonian_to_polar(
        params: MatchgateStandardHamiltonianParams,
        **kwargs
) -> MatchgatePolarParams:
    params = standard_hamiltonian_to_standard(params, **kwargs)
    return standard_to_polar(params, **kwargs)


def standard_hamiltonian_to_composed_hamiltonian(
        params: MatchgateStandardHamiltonianParams,
        **kwargs
) -> MatchgateComposedHamiltonianParams:
    params = standard_hamiltonian_to_hamiltonian_coefficients(params, **kwargs)
    return hamiltonian_coefficients_to_composed_hamiltonian(params, **kwargs)


_transfer_funcs_by_type: Dict[
    Type[MatchgateParams], Dict[
        Type[MatchgateParams], Callable[[MatchgateParams, ...], MatchgateParams]
    ]
] = {
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


def params_to(params, __cls: Type[MatchgateParams], **kwargs) -> MatchgateParams:
    if isinstance(params, __cls):
        return params
    if not isinstance(params, MatchgateParams):
        return __cls(*params, **kwargs)
    return _transfer_funcs_by_type[type(params)][__cls](params, **kwargs)
