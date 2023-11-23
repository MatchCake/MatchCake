from typing import Optional

import numpy as np

from .matchgate_params import MatchgateParams


class MatchgatePolarParams(MatchgateParams):
    N_PARAMS = 7
    RANGE_OF_PARAMS = [(0.0, 1.0) for _ in range(2)] + [(0, 2 * np.pi) for _ in range(N_PARAMS - 2)]
    ALLOW_COMPLEX_PARAMS = False

    def __init__(
            self,
            r0: float = 0.0,
            r1: float = 0.0,
            theta0: float = 0.0,
            theta1: float = 0.0,
            theta2: float = 0.0,
            theta3: float = 0.0,
            theta4: Optional[float] = None,
            *,
            backend='numpy',
            **kwargs
    ):
        super().__init__(backend=backend)
        if not self.ALLOW_COMPLEX_PARAMS:
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
        if not self.ALLOW_COMPLEX_PARAMS:
            self._theta4 = self._maybe_cast_to_real(self._theta4)[0]
        self._force_theta4_in_numpy_repr = kwargs.get('force_theta4_in_numpy_repr', True)

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

    @classmethod
    def to_sympy(cls):
        import sympy as sp
        r0, r1 = sp.symbols('r_0 r_1')
        theta0, theta1, theta2, theta3, theta4 = sp.symbols('\\theta_0 \\theta_1 \\theta_2 \\theta_3 \\theta_4')
        return sp.Matrix([r0, r1, theta0, theta1, theta2, theta3, theta4])

    @staticmethod
    def compute_r_tilde(r, backend='numpy') -> complex:
        _pkg = MatchgateParams.load_backend_lib(backend)
        return _pkg.sqrt(1 - complex(r) ** 2)

    @property
    def r0_tilde(self) -> complex:
        """
        Return :math:`\\Tilde{r}_0 = \\sqrt{1 - r_0^2}`

        :return: :math:`\\Tilde{r}_0`
        """
        return self.compute_r_tilde(self.r0, backend=self.backend)

    @property
    def r1_tilde(self) -> complex:
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
        return (f"{self.__class__.__name__}("
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
            theta4=self.theta4,
        )
    
    def adjoint(self):
        return MatchgatePolarParams(
            r0=self.r0,
            r1=self.r1,
            theta0=-self.theta0,
            theta1=-self.theta1,
            theta2=-self.theta2,
            theta3=-self.theta3,
            theta4=-self.theta4,
        )
    