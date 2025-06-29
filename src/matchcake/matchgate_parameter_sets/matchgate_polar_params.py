import numpy as np
import pennylane as qml

from .. import utils
from .matchgate_params import MatchgateParams


class MatchgatePolarParams(MatchgateParams):
    r"""
    Matchgate polar parameters.

    They are the parameters of a Matchgate operation in the standard form which is a 4x4 matrix

    .. math::

            \begin{bmatrix}
                r_0 e^{i\theta_0} & 0 & 0 & (\sqrt{1 - r_0^2}) e^{-i(\theta_1+\pi)} \\
                0 & r_1 e^{i\theta_2} & (\sqrt{1 - r_1^2}) e^{-i(\theta_3+\pi)} & 0 \\
                0 & (\sqrt{1 - r_1^2}) e^{i\theta_3} & r_1 e^{-i\theta_2} & 0 \\
                (\sqrt{1 - r_0^2}) e^{i\theta_1} & 0 & 0 & r_0 e^{-i\theta_0}
            \end{bmatrix}

        where :math:`r_0, r_1, \theta_0, \theta_1, \theta_2, \theta_3, \theta_4` are the parameters.

    """

    N_PARAMS = 7
    RANGE_OF_PARAMS = [(0.0, 1.0) for _ in range(2)] + [(0, 2 * np.pi) for _ in range(N_PARAMS - 2)]
    ALLOW_COMPLEX_PARAMS = False
    ATTRS = ["r0", "r1", "theta0", "theta1", "theta2", "theta3", "theta4"]

    def __init__(self, *args, **kwargs):
        if "theta2" in kwargs:
            kwargs.setdefault("theta4", -utils.math.astensor(kwargs["theta2"]))
        args, kwargs = self._maybe_cast_inputs_to_real(args, kwargs)
        super().__init__(*args, **kwargs)

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

    @classmethod
    def to_sympy(cls):
        import sympy as sp

        r0, r1 = sp.symbols("r_0 r_1")
        theta0, theta1, theta2, theta3, theta4 = sp.symbols("\\theta_0 \\theta_1 \\theta_2 \\theta_3 \\theta_4")
        return sp.Matrix([r0, r1, theta0, theta1, theta2, theta3, theta4])

    @classmethod
    def compute_r_tilde(cls, r, backend="pennylane.math") -> complex:
        _pkg = MatchgateParams.load_backend_lib(backend)
        return _pkg.sqrt(1 - qml.math.cast(r, dtype=complex) ** 2 + cls.DIVISION_EPSILON)

    @property
    def r0_tilde(self) -> complex:
        """
        Return :math:`\\Tilde{r}_0 = \\sqrt{1 - r_0^2}`

        :return: :math:`\\Tilde{r}_0`
        """
        return self.compute_r_tilde(self.r0)

    @property
    def r1_tilde(self) -> complex:
        """
        Return :math:`\\Tilde{r}_1 = \\sqrt{1 - r_1^2}`

        :return: :math:`\\Tilde{r}_1`
        """
        return self.compute_r_tilde(self.r1)

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

    def __matmul__(self, other):
        if not isinstance(other, MatchgatePolarParams):
            other = MatchgatePolarParams.parse_from_params(other)
        raise NotImplementedError("MatchgatePolarParams.__matmul__ is not implemented yet.")
        r0 = self.r0 * other.r0
        r1 = self.r1 * other.r1
        theta0 = self.theta0 + other.theta0
        theta1 = self.theta1 + other.theta1
        theta2 = self.theta2 + other.theta2
        theta3 = self.theta3 + other.theta3
        theta4 = self.theta4 + other.theta4
        return MatchgatePolarParams(
            r0=r0,
            r1=r1,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
        )
