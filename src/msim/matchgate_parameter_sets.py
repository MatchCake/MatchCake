from typing import NamedTuple, Optional, Any, Union
import numpy as np
import importlib

from . import utils


class MatchgateParams:
    r"""
    A matchgate can be represented by several set of parameters and there exists a mapping between them.
    """
    @staticmethod
    def _maybe_cast_to_real(*params):
        is_real = utils.check_if_imag_is_zero(np.array(params))
        if is_real:
            return tuple(np.real(np.array(params)))
        else:
            raise ValueError("The parameters must be real.")

    @staticmethod
    def to_sympy():
        raise NotImplementedError("This method must be implemented in the child class.")

    @staticmethod
    def parse_from_params(params: 'MatchgateParams') -> 'MatchgateParams':
        raise NotImplementedError("This method must be implemented in the child class.")

    @classmethod
    def from_numpy(cls, params: np.ndarray) -> 'MatchgateParams':
        tuple_params = tuple(params.flatten())
        return cls(*tuple_params, backend='numpy')

    @staticmethod
    def parse_from_any(params: Any) -> 'MatchgateParams':
        r"""
        Try to parse the input parameters to a MatchgateParams object.

        :param params: The input parameters.
        :type params: Any
        :return: The parsed parameters.
        :rtype: MatchgateParams
        """
        if isinstance(params, MatchgateParams):
            return params
        elif isinstance(params, np.ndarray):
            return MatchgateParams.from_numpy(params)
        else:
            return MatchgateParams.from_numpy(np.asarray(params))

    @staticmethod
    def load_backend_lib(backend):
        if isinstance(backend, str):
            backend = importlib.import_module(backend)
        return backend

    def __init__(self, backend='numpy'):
        self._backend = self.load_backend_lib(backend)

    @property
    def backend(self):
        return self._backend

    def to_numpy(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def to_string(self):
        return str(self)

    def __repr__(self):
        return f"MatchgateParams({str(self)})"

    def __str__(self):
        params_as_str = ", ".join([f"{p:.4f}" for p in self.to_numpy()])
        return f"[{params_as_str}]"

    def __eq__(self, other):
        return np.allclose(self.to_numpy(), other.to_numpy())

    def __hash__(self):
        return hash(self.to_string())

    def __getitem__(self, item):
        return self.to_numpy()[item]

    def __len__(self):
        return self.to_numpy().size


class MatchgatePolarParams(MatchgateParams):
    def __init__(
            self,
            r0: float,
            r1: float,
            theta0: float,
            theta1: float,
            theta2: float,
            theta3: float,
            # theta4: Optional[float] = None,
            *,
            backend='numpy',
            **kwargs
    ):
        super().__init__(backend=backend)
        # if theta4 is None:
        #     theta4 = -theta2
        # r0, r1, theta0, theta1, theta2, theta3, theta4 = self._maybe_cast_to_real(
        #     r0, r1, theta0, theta1, theta2, theta3, theta4
        # )
        r0, r1, theta0, theta1, theta2, theta3 = self._maybe_cast_to_real(
            r0, r1, theta0, theta1, theta2, theta3
        )
        self._r0 = r0
        self._r1 = r1
        self._theta0 = theta0
        self._theta1 = theta1
        self._theta2 = theta2
        self._theta3 = theta3
        # self._theta4 = theta4
        self._theta4 = theta4 = -theta2

    @property
    def r0(self) -> float:
        return self._r0

    @property
    def r1(self) -> float:
        return self._r1

    @property
    def theta0(self) -> float:
        return self._theta0

    @property
    def theta1(self) -> float:
        return self._theta1

    @property
    def theta2(self) -> float:
        return self._theta2

    @property
    def theta3(self) -> float:
        return self._theta3

    @property
    def theta4(self) -> float:
        return self._theta4

    @staticmethod
    def parse_from_params(params: 'MatchgateParams') -> 'MatchgatePolarParams':
        if isinstance(params, MatchgatePolarParams):
            return params
        elif isinstance(params, MatchgateStandardParams):
            return MatchgatePolarParams.parse_from_standard_params(params)
        elif isinstance(params, MatchgateHamiltonianParams):
            return MatchgatePolarParams.parse_from_hamiltonian_params(params)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgatePolarParams.parse_from_composed_hamiltonian_params(params)
        return MatchgatePolarParams(*params)

    @staticmethod
    def parse_from_standard_params(entries: 'MatchgateStandardParams'):
        entries = MatchgateStandardParams.parse_from_full_params(entries)
        return MatchgatePolarParams(
            r0=np.sqrt(entries.a * np.conjugate(entries.a)),
            r1=np.sqrt(entries.w * np.conjugate(entries.w)),
            theta0=np.angle(entries.a),
            theta1=np.angle(entries.c),
            theta2=np.angle(entries.w),
            theta3=np.angle(entries.y),
            # theta4=np.angle(entries.z),
        )
    
    @staticmethod
    def parse_from_hamiltonian_params(params: 'MatchgateHamiltonianParams') -> 'MatchgatePolarParams':
        std_params = MatchgateStandardParams.parse_from_params(params)
        return MatchgatePolarParams.parse_from_standard_params(std_params)
    
    @staticmethod
    def parse_from_composed_hamiltonian_params(
            params: 'MatchgateComposedHamiltonianParams'
    ) -> 'MatchgatePolarParams':
        hami_params = MatchgateHamiltonianParams.parse_from_params(params)
        return MatchgatePolarParams.parse_from_hamiltonian_params(hami_params)

    @staticmethod
    def to_sympy():
        import sympy as sp
        r0, r1 = sp.symbols('r_0 r_1')
        theta0, theta1, theta2, theta3, theta4 = sp.symbols('\\theta_0 \\theta_1 \\theta_2 \\theta_3 \\theta_4')
        return sp.Matrix([r0, r1, theta0, theta1, theta2, theta3, theta4])

    @staticmethod
    def compute_r_tilde(r, backend='numpy'):
        _pkg = MatchgateParams.load_backend_lib(backend)
        return _pkg.sqrt(1 - r ** 2)

    @property
    def r0_tilde(self) -> float:
        """
        Return :math:`\\Tilde{r}_0 = \\sqrt{1 - r_0^2}`

        :return: :math:`\\Tilde{r}_0`
        """
        return self.compute_r_tilde(self.r0, backend=self.backend)

    @property
    def r1_tilde(self) -> float:
        """
        Return :math:`\\Tilde{r}_1 = \\sqrt{1 - r_1^2}`

        :return: :math:`\\Tilde{r}_1`
        """
        return self.compute_r_tilde(self.r1, backend='numpy')

    def to_numpy(self):
        return np.asarray([
            self.r0,
            self.r1,
            self.theta0,
            self.theta1,
            self.theta2,
            self.theta3,
            # self.theta4,
        ])

    def to_string(self):
        return str(self)

    def __repr__(self):
        return (f"MatchgateParams("
                f"r0={self.r0}, "
                f"r1={self.r1}, "
                f"theta0={self.theta0}, "
                f"theta1={self.theta1}, "
                f"theta2={self.theta2}, "
                f"theta3={self.theta3}, "
                f"theta4={self.theta4})")

    def __str__(self):
        return f"[{self.r0}, {self.r1}, {self.theta0}, {self.theta1}, {self.theta2}, {self.theta3}, {self.theta4}]"

    def __hash__(self):
        return hash(self.to_string())

    def __copy__(self):
        return MatchgatePolarParams(
            r0=self.r0,
            r1=self.r1,
            theta0=self.theta0,
            theta1=self.theta1,
            theta2=self.theta2,
            theta3=self.theta3,
            # theta4=self.theta4,
        )


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
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._w = w
        self._x = x
        self._y = y
        self._z = z

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
    def parse_from_params(params: 'MatchgateParams') -> 'MatchgateStandardParams':
        if isinstance(params, MatchgateStandardParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateStandardParams.parse_from_polar_params(params)
        elif isinstance(params, MatchgateHamiltonianParams):
            return MatchgateStandardParams.parse_from_hamiltonian_params(params)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgateStandardParams.parse_from_composed_hamiltonian_params(params)
        return MatchgateStandardParams(*params)

    @staticmethod
    def parse_from_polar_params(params, backend='numpy') -> 'MatchgateStandardParams':
        _pkg = MatchgateParams.load_backend_lib(backend)

        params = MatchgatePolarParams.parse_from_params(params)
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
    def parse_from_hamiltonian_params(params: 'MatchgateHamiltonianParams') -> 'MatchgateStandardParams':
        params = MatchgateHamiltonianParams.parse_from_params(params)
        hamiltonian = utils.get_4x4_non_interacting_fermionic_hamiltonian_from_params(params)
        gate = utils.get_unitary_from_hermitian_matrix(hamiltonian)
        elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
        params_arr = gate[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        return MatchgateStandardParams(*params_arr)
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
            params: 'MatchgateComposedHamiltonianParams'
    ) -> 'MatchgateStandardParams':
        hami_params = MatchgateHamiltonianParams.parse_from_params(params)
        return MatchgateStandardParams.parse_from_hamiltonian_params(hami_params)

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
        return np.asarray([
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


class MatchgateHamiltonianParams(MatchgateParams):
    def __init__(
            self,
            h0: float,
            h1: float,
            h2: float,
            h3: float,
            h4: float,
            h5: float,
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        h0, h1, h2, h3, h4, h5 = self._maybe_cast_to_real(h0, h1, h2, h3, h4, h5)
        self._h0 = h0
        self._h1 = h1
        self._h2 = h2
        self._h3 = h3
        self._h4 = h4
        self._h5 = h5

    @property
    def h0(self) -> float:
        return self._h0

    @property
    def h1(self) -> float:
        return self._h1

    @property
    def h2(self) -> float:
        return self._h2

    @property
    def h3(self) -> float:
        return self._h3

    @property
    def h4(self) -> float:
        return self._h4

    @property
    def h5(self) -> float:
        return self._h5

    @staticmethod
    def parse_from_params(params: 'MatchgateParams') -> 'MatchgateHamiltonianParams':
        if isinstance(params, MatchgateHamiltonianParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateHamiltonianParams.parse_from_polar_params(params)
        elif isinstance(params, MatchgateStandardParams):
            return MatchgateHamiltonianParams.parse_from_standard_params(params)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgateHamiltonianParams.parse_from_composed_hamiltonian_params(params)
        return MatchgateHamiltonianParams(*params)
    
    @staticmethod
    def parse_from_polar_params(params: 'MatchgatePolarParams') -> 'MatchgateHamiltonianParams':
        std_params = MatchgateStandardParams.parse_from_params(params)
        return MatchgateHamiltonianParams.parse_from_standard_params(std_params)
    
    @staticmethod
    def parse_from_standard_params(params: 'MatchgateStandardParams') -> 'MatchgateHamiltonianParams':
        params = MatchgateStandardParams.parse_from_full_params(params)
        return MatchgateHamiltonianParams(
            h0=(2 - params.a - params.w) / 4,
            h1=(2 * params.x + params.b) / 20j,
            h2=(5 * params.y - 4 * params.b - 2 * params.x) / 20,
            h3=-(7 * params.x + 6 * params.b) / 20,
            h4=-(9 * params.b + 9 * params.x + 5 * params.y) / 20j,
            h5=(params.w - params.a) / 4,
        )

    @staticmethod
    def parse_from_composed_hamiltonian_params(params: 'MatchgateComposedHamiltonianParams') -> 'MatchgateHamiltonianParams':
        return MatchgateHamiltonianParams(
                h0=params.n_z + params.m_z,
                h1=params.n_y + params.m_y,
                h2=params.n_x - params.m_x,
                h3=params.n_x + params.m_x,
                h4=params.n_y - params.m_y,
                h5=params.n_z - params.m_z,
                backend=params.backend,
            )

    @staticmethod
    def to_sympy():
        import sympy as sp
        h0, h1, h2, h3, h4, h5 = sp.symbols('h_0 h_1 h_2 h_3 h_4 h_5')
        return sp.Matrix([h0, h1, h2, h3, h4, h5])

    def to_numpy(self):
        return np.asarray([
            self.h0,
            self.h1,
            self.h2,
            self.h3,
            self.h4,
            self.h5,
        ])

    def __repr__(self):
        return (f"MatchgateParams("
                f"h0={self.h0}, "
                f"h1={self.h1}, "
                f"h2={self.h2}, "
                f"h3={self.h3}, "
                f"h4={self.h4}, "
                f"h5={self.h5})")


class MatchgateComposedHamiltonianParams(MatchgateParams):
    def __init__(
            self,
            n_x: float,
            n_y: float,
            n_z: float,
            m_x: float,
            m_y: float,
            m_z: float,
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        n_x, n_y, n_z, m_x, m_y, m_z = self._maybe_cast_to_real(n_x, n_y, n_z, m_x, m_y, m_z)
        self._n_x = n_x
        self._n_y = n_y
        self._n_z = n_z
        self._m_x = m_x
        self._m_y = m_y
        self._m_z = m_z

    @property
    def n_x(self) -> float:
        return self._n_x

    @property
    def n_y(self) -> float:
        return self._n_y

    @property
    def n_z(self) -> float:
        return self._n_z

    @property
    def m_x(self) -> float:
        return self._m_x

    @property
    def m_y(self) -> float:
        return self._m_y

    @property
    def m_z(self) -> float:
        return self._m_z

    @staticmethod
    def parse_from_params(params: 'MatchgateParams') -> 'MatchgateComposedHamiltonianParams':
        if isinstance(params, MatchgateComposedHamiltonianParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateComposedHamiltonianParams.parse_from_polar_params(params)
        elif isinstance(params, MatchgateStandardParams):
            return MatchgateComposedHamiltonianParams.parse_from_standard_params(params)
        elif isinstance(params, MatchgateHamiltonianParams):
            return MatchgateComposedHamiltonianParams.parse_from_hamiltonian_params(params)
        return MatchgateComposedHamiltonianParams(*params)
    
    @staticmethod
    def parse_from_polar_params(params: 'MatchgatePolarParams') -> 'MatchgateComposedHamiltonianParams':
        std_params = MatchgateStandardParams.parse_from_params(params)
        return MatchgateComposedHamiltonianParams.parse_from_standard_params(std_params)
    
    @staticmethod
    def parse_from_standard_params(params: 'MatchgateStandardParams') -> 'MatchgateComposedHamiltonianParams':
        hamiltonian_params = MatchgateHamiltonianParams.parse_from_standard_params(params)
        return MatchgateComposedHamiltonianParams.parse_from_hamiltonian_params(hamiltonian_params)
    
    @staticmethod
    def parse_from_hamiltonian_params(params: 'MatchgateHamiltonianParams') -> 'MatchgateComposedHamiltonianParams':
        return MatchgateComposedHamiltonianParams(
            n_x=0.5 * (params.h2 + params.h3),
            n_y=0.5 * (params.h1 + params.h4),
            n_z=0.5 * (params.h0 + params.h5),
            m_x=0.5 * (params.h3 - params.h2),
            m_y=0.5 * (params.h4 - params.h1),
            m_z=0.5 * (params.h5 - params.h0),
            backend=params.backend,
        )

    @staticmethod
    def to_sympy():
        import sympy as sp
        n_x, n_y, n_z, m_x, m_y, m_z = sp.symbols('n_x n_y n_z m_x m_y m_z')
        return sp.Matrix([n_x, n_y, n_z, m_x, m_y, m_z])

    def to_numpy(self):
        return np.asarray([
            self.n_x,
            self.n_y,
            self.n_z,
            self.m_x,
            self.m_y,
            self.m_z,
        ])

    def __repr__(self):
        return (f"MatchgateParams("
                f"n_x={self.n_x}, "
                f"n_y={self.n_y}, "
                f"n_z={self.n_z}, "
                f"m_x={self.m_x}, "
                f"m_y={self.m_y}, "
                f"m_z={self.m_z})")
