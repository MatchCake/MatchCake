from typing import Optional

import numpy as np

from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import MatchgateHamiltonianCoefficientsParams
from .matchgate_params import MatchgateParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams


class MatchgatePolarParams(MatchgateParams):
    def __init__(
            self,
            r0: float,
            r1: float,
            theta0: float,
            theta1: float,
            theta2: float,
            theta3: float,
            theta4: Optional[float] = None,
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
        self._theta4 = theta4 if theta4 is not None else -theta2
        self._theta4 = self._maybe_cast_to_real(self._theta4)
        self._force_theta4_in_numpy_repr = kwargs.get('force_theta4_in_numpy_repr', False)

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

    @property
    def theta4_is_relevant(self) -> bool:
        return not np.isclose(self.theta4, -self.theta2)

    @property
    def force_theta4_in_numpy_repr(self) -> bool:
        return self._force_theta4_in_numpy_repr

    @force_theta4_in_numpy_repr.setter
    def force_theta4_in_numpy_repr(self, value: bool):
        self._force_theta4_in_numpy_repr = value

    @staticmethod
    def parse_from_params(params: 'MatchgateParams', backend="numpy") -> 'MatchgatePolarParams':
        # from . import transfer_functions
        # params = transfer_functions.params_to(params, MatchgatePolarParams)
        if isinstance(params, MatchgatePolarParams):
            return params
        elif isinstance(params, MatchgateStandardParams):
            return MatchgatePolarParams.parse_from_standard_params(params, backend=backend)
        elif isinstance(params, MatchgateHamiltonianCoefficientsParams):
            return MatchgatePolarParams.parse_from_hamiltonian_params(params, backend=backend)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgatePolarParams.parse_from_composed_hamiltonian_params(params, backend=backend)
        elif isinstance(params, MatchgateStandardHamiltonianParams):
            return MatchgatePolarParams.parse_from_standard_hamiltonian_params(params, backend=backend)
        return MatchgatePolarParams(*params)

    @staticmethod
    def parse_from_standard_params(params: 'MatchgateStandardParams', backend='numpy') -> 'MatchgatePolarParams':
        params = MatchgateStandardParams.parse_from_params(params, backend=backend)
        a, b, c, d, w, x, y, z = params.to_numpy().astype(complex)
        r0 = np.sqrt(a * np.conjugate(a))
        r0_tilde = MatchgatePolarParams.compute_r_tilde(r0, backend=backend)
        r1 = np.sqrt(w * np.conjugate(w))
        r1_tilde = MatchgatePolarParams.compute_r_tilde(r1, backend=backend)
        eps = 1e-12
        if np.isclose(r0, 0) or np.isclose(r1, 0):
            theta0 = 0
            theta1 = -1j * np.log(c + eps)
            theta2 = -0.5j * (np.log(-b + eps) - np.log(np.conjugate(c) + eps))
            theta3 = -1j * np.log(z + eps)
            theta4 = -0.5j * (np.log(-b + eps) - np.log(np.conjugate(c) + eps))
        elif np.isclose(r0, 0) or np.isclose(r1, 1):
            theta0 = 0
            theta1 = -1j * np.log(c + eps)
            theta2 = -1j * np.log(w + eps)
            theta3 = 0
            theta4 = -1j * np.log(z + eps)
        elif np.isclose(r0, 0) or not np.isclose(r1, 0) or np.isclose(r1, 1):
            theta0 = 0
            theta1 = -1j * np.log(c + eps)
            theta2 = -1j * (np.log(w + eps) - np.log(r1 + eps))
            theta3 = -1j * (np.log(y + eps) - np.log(r1_tilde + eps))
            theta4 = -1j * (np.log(z + eps) - np.log(r1 + eps))
        elif np.isclose(r0, 1) or np.isclose(r1, 0):
            theta0 = -1j * np.log(a + eps)
            theta1 = 0
            theta2 = -0.5j * (np.log(d + eps) - np.log(np.conjugate(a) + eps))
            theta3 = -1j * np.log(y + eps)
            theta4 = -0.5j * (np.log(d + eps) - np.log(np.conjugate(a) + eps))
        elif np.isclose(r0, 1) or np.isclose(r1, 1):
            theta0 = -1j * np.log(a + eps)
            theta1 = 0
            theta2 = -1j * np.log(w + eps)
            theta3 = 0
            theta4 = -1j * np.log(z + eps)
        elif np.isclose(r0, 1) or not np.isclose(r1, 0) or np.isclose(r1, 1):
            theta0 = -1j * np.log(a + eps)
            theta1 = 0
            theta2 = -1j * (np.log(w + eps) - np.log(r1 + eps))
            theta3 = -1j * (np.log(y + eps) - np.log(r1_tilde + eps))
            theta4 = -1j * (np.log(z + eps) - np.log(r1 + eps))
        elif not np.isclose(r0, 0) or np.isclose(r0, 1) or np.isclose(r1, 0) or np.isclose(r1, 1):
            theta0 = -1j * (np.log(a + eps) - np.log(r0 + eps))
            theta1 = -1j * (np.log(c + eps) - np.log(r0_tilde + eps))
            theta2 = -1j * (np.log(w + eps) - np.log(r1 + eps))
            theta3 = -1j * (np.log(y + eps) - np.log(r1_tilde + eps))
            theta4 = -1j * (np.log(z + eps) - np.log(r1 + eps))
        else:
            raise ValueError(f"Invalid parameters: {params}")

        return MatchgatePolarParams(
            r0=r0,
            r1=r1,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
            # theta0=-1j * (np.log(a + eps) - np.log(r0 + eps)),
            # theta1=-1j * (np.log(c + eps) - np.log(r0_tilde + eps)),
            # theta2=-1j * (np.log(w + eps) - np.log(r1 + eps)),
            # theta3=-1j * (np.log(y + eps) - np.log(r1_tilde + eps)),
            # theta4=-1j * (np.log(z + eps) - np.log(r1 + eps)),
            # theta0=-0.5j * np.log(params.a / params.d),
            # theta1=0.5 * (1j * np.log(params.b / params.c) - np.pi),
            # theta2=-0.5j * np.log(params.w / params.z),
            # theta3=0.5 * (1j * np.log(params.x / params.y) - np.pi),
            backend=backend,
        )

    @staticmethod
    def parse_from_hamiltonian_params(
            params: 'MatchgateHamiltonianCoefficientsParams',
            backend='numpy'
    ) -> 'MatchgatePolarParams':
        std_params = MatchgateStandardParams.parse_from_params(params, backend=backend)
        return MatchgatePolarParams.parse_from_standard_params(std_params, backend=backend)

    @staticmethod
    def parse_from_composed_hamiltonian_params(
            params: 'MatchgateComposedHamiltonianParams',
            backend='numpy'
    ) -> 'MatchgatePolarParams':
        hami_params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params, backend=backend)
        return MatchgatePolarParams.parse_from_hamiltonian_params(hami_params, backend=backend)

    @staticmethod
    def parse_from_standard_hamiltonian_params(
            params: 'MatchgateStandardHamiltonianParams',
            backend='numpy'
    ) -> 'MatchgatePolarParams':
        std_hami_params = MatchgateStandardHamiltonianParams.parse_from_params(params, backend=backend)
        std_params = MatchgateStandardParams.parse_from_standard_hamiltonian_params(std_hami_params, backend=backend)
        return MatchgatePolarParams.parse_from_standard_params(std_params, backend=backend)

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
        p_list = [
            self.r0,
            self.r1,
            self.theta0,
            self.theta1,
            self.theta2,
            self.theta3,
        ]
        if self.theta4_is_relevant or self._force_theta4_in_numpy_repr:
            p_list.append(self.theta4)
        return np.asarray(p_list)

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
