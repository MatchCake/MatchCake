from typing import Union

import numpy as np

from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import MatchgateHamiltonianCoefficientsParams
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .. import utils


class MatchgateStandardParams(MatchgateParams):
    ZEROS_INDEXES = [
        (0, 1), (0, 2),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 1), (3, 2),
    ]
    ELEMENTS_INDEXES = [
        (0, 0), (0, 3),  # a, b
        (3, 0), (3, 3),  # c, d
        (1, 1), (1, 2),  # w, x
        (2, 1), (2, 2),  # y, z
    ]

    def __init__(
            self,
            a: Union[float, complex],
            b: Union[float, complex],
            c: Union[float, complex],
            d: Union[float, complex],
            w: Union[float, complex],
            x: Union[float, complex],
            y: Union[float, complex],
            z: Union[float, complex],
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        self._a = complex(a)
        self._b = complex(b)
        self._c = complex(c)
        self._d = complex(d)
        self._w = complex(w)
        self._x = complex(x)
        self._y = complex(y)
        self._z = complex(z)

    @property
    def a(self) -> Union[float, complex]:
        return self._a

    @property
    def b(self) -> Union[float, complex]:
        return self._b

    @property
    def c(self) -> Union[float, complex]:
        return self._c

    @property
    def d(self) -> Union[float, complex]:
        return self._d

    @property
    def w(self) -> Union[float, complex]:
        return self._w

    @property
    def x(self) -> Union[float, complex]:
        return self._x

    @property
    def y(self) -> Union[float, complex]:
        return self._y

    @property
    def z(self) -> Union[float, complex]:
        return self._z

    @staticmethod
    def parse_from_full_params(full_params):
        return MatchgateStandardParams(
            a=full_params[0],
            b=full_params[1],
            c=full_params[2],
            d=full_params[3],
            w=full_params[4],
            x=full_params[5],
            y=full_params[6],
            z=full_params[7],
        )

    @staticmethod
    def parse_from_params(params: 'MatchgateParams', backend="numpy") -> 'MatchgateStandardParams':
        if isinstance(params, MatchgateStandardParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateStandardParams.parse_from_polar_params(params, backend=backend)
        elif isinstance(params, MatchgateHamiltonianCoefficientsParams):
            return MatchgateStandardParams.parse_from_hamiltonian_params(params, backend=backend)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgateStandardParams.parse_from_composed_hamiltonian_params(params, backend=backend)
        elif isinstance(params, MatchgateStandardHamiltonianParams):
            return MatchgateStandardParams.parse_from_standard_hamiltonian_params(params, backend=backend)
        return MatchgateStandardParams(*params)

    @staticmethod
    def parse_from_polar_params(params, backend='numpy') -> 'MatchgateStandardParams':
        _pkg = MatchgateParams.load_backend_lib(backend)

        params = MatchgatePolarParams.parse_from_params(params, backend=backend)
        r0_tilde = MatchgatePolarParams.compute_r_tilde(params.r0, backend=_pkg)
        r1_tilde = MatchgatePolarParams.compute_r_tilde(params.r1, backend=_pkg)
        return MatchgateStandardParams(
            a=params.r0 * _pkg.exp(1j * params.theta0),
            b=r0_tilde * _pkg.exp(1j * (params.theta2 + params.theta4 - (params.theta1 + _pkg.pi))),
            c=r0_tilde * _pkg.exp(1j * params.theta1),
            d=params.r0 * _pkg.exp(1j * (params.theta2 + params.theta4 - params.theta0)),
            w=params.r1 * _pkg.exp(1j * params.theta2),
            x=r1_tilde * _pkg.exp(1j * (params.theta2 + params.theta4 - (params.theta3 + _pkg.pi))),
            y=r1_tilde * _pkg.exp(1j * params.theta3),
            z=params.r1 * _pkg.exp(1j * params.theta4),
            backend=_pkg,
        )

    @staticmethod
    def parse_from_hamiltonian_params(
            params: 'MatchgateHamiltonianCoefficientsParams', backend='numpy'
    ) -> 'MatchgateStandardParams':
        params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params, backend=backend)
        hamiltonian = utils.get_4x4_non_interacting_fermionic_hamiltonian_from_params(params)
        gate = utils.get_unitary_from_hermitian_matrix(hamiltonian)
        elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
        params_arr = gate[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        return MatchgateStandardParams(*params_arr, backend=backend)
        # return MatchgateStandardParams(
        #     a=-2 * (params.h0 + params.h5) + 1,
        #     b=2j * (params.h4 - params.h1) - 2 * (params.h2 + params.h3),
        #     c=2j * (params.h1 - params.h4) - 2 * (params.h2 + params.h3),
        #     d=2 * (params.h0 + params.h5) + 1,
        #     w=2 * (params.h5 - params.h0) + 1,
        #     x=-2j * (params.h1 + params.h4) - 2 * (params.h2 - params.h3),
        #     y=2j * (params.h1 + params.h4) + 2 * (params.h2 - params.h3),
        #     z=2 * (params.h0 - params.h5) + 1,
        # )

    @staticmethod
    def parse_from_composed_hamiltonian_params(
            params: 'MatchgateComposedHamiltonianParams', backend='numpy'
    ) -> 'MatchgateStandardParams':
        hami_params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params, backend=backend)
        return MatchgateStandardParams.parse_from_hamiltonian_params(hami_params, backend=backend)

    @staticmethod
    def parse_from_standard_hamiltonian_params(
            params: 'MatchgateStandardHamiltonianParams', backend='numpy'
    ) -> 'MatchgateStandardParams':
        std_hamil_params = MatchgateStandardHamiltonianParams.parse_from_params(params, backend=backend)
        gate = utils.get_unitary_from_hermitian_matrix(std_hamil_params.to_matrix())
        elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
        params_arr = gate[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        return MatchgateStandardParams(*params_arr, backend=backend)

    @staticmethod
    def to_sympy():
        import sympy as sp
        a, b, c, d = sp.symbols('a b c d')
        w, x, y, z = sp.symbols('w x y z')
        return sp.Matrix([a, b, c, d, w, x, y, z])

    def to_numpy(self):
        return np.asarray([
            self.a,
            self.b,
            self.c,
            self.d,
            self.w,
            self.x,
            self.y,
            self.z,
        ])

    def to_matrix(self):
        return self.backend.asarray([
            [self.a, 0, 0, self.b],
            [0, self.w, self.x, 0],
            [0, self.y, self.z, 0],
            [self.c, 0, 0, self.d],
        ])

    def adjoint(self):
        r"""
        Return the adjoint version of the parameters.

        :return: The adjoint parameters.
        """
        return MatchgateStandardParams(
            a=np.conjugate(self.a),
            b=np.conjugate(self.c),
            c=np.conjugate(self.b),
            d=np.conjugate(self.d),
            w=np.conjugate(self.w),
            x=np.conjugate(self.y),
            y=np.conjugate(self.x),
            z=np.conjugate(self.z),
            backend=self.backend,
        )
