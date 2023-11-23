import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from .. import utils

from .matchgate_params import MatchgateParams


class MatchgateHamiltonianCoefficientsParams(MatchgateParams):
    r"""
    The hamiltionian of a system of non-interacting fermions can be written as:

    .. math::
        H = -i\sum_{\mu\nu}h_{\mu\nu}c_{\mu}c_{\nu} + \epsilon\mathbb{I}

    where :math:`c_{\mu}` and :math:`c_{\nu}` are the fermionic majorana operators, :math:`\epsilon` is the
    energy offset, :math:`\mathbb{I}` is the identity matrix, and :math:`h_{\mu\nu}` are the
    hamiltonian coefficients that can be expressed in terms of a skew-symmetric matrix :math:`h` as:

    .. math::
        h = \begin{pmatrix}
            0 & h_{0} & h_{1} & h_{2} \\
            -h_{0} & 0 & h_{3} & h_{4} \\
            -h_{1} & -h_{3} & 0 & h_{5} \\
            -h_{2} & -h_{4} & -h_{5} & 0
        \end{pmatrix}

    where :math:`h_{0}, h_{1}, h_{2}, h_{3}, h_{4}, h_{5}` are the hamiltonian coefficients.

    """
    N_PARAMS = 7
    ALLOW_COMPLEX_PARAMS = False
    DEFAULT_PARAMS_TYPE = float
    DEFAULT_RANGE_OF_PARAMS = (-1e6, 1e6)

    def __init__(
            self,
            h0: float = 0.0,
            h1: float = 0.0,
            h2: float = 0.0,
            h3: float = 0.0,
            h4: float = 0.0,
            h5: float = 0.0,
            epsilon: float = 0.0,
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        if not self.ALLOW_COMPLEX_PARAMS:
            h0, h1, h2, h3, h4, h5, epsilon = self._maybe_cast_to_real(h0, h1, h2, h3, h4, h5, epsilon)
        self._h0 = h0
        self._h1 = h1
        self._h2 = h2
        self._h3 = h3
        self._h4 = h4
        self._h5 = h5
        self._epsilon = epsilon

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

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    @property
    def h0_op(self):
        return -2j * utils.get_majorana(0, 2) @ utils.get_majorana(1, 2)

    @property
    def h1_op(self):
        return -2j * utils.get_majorana(0, 2) @ utils.get_majorana(2, 2)

    @property
    def h2_op(self):
        return -2j * utils.get_majorana(0, 2) @ utils.get_majorana(3, 2)

    @property
    def h3_op(self):
        return -2j * utils.get_majorana(1, 2) @ utils.get_majorana(2, 2)

    @property
    def h4_op(self):
        return -2j * utils.get_majorana(1, 2) @ utils.get_majorana(3, 2)

    @property
    def h5_op(self):
        return -2j * utils.get_majorana(2, 2) @ utils.get_majorana(3, 2)

    @property
    def epsilon_op(self):
        return self.backend.eye(4)

    @classmethod
    def to_sympy(cls):
        import sympy as sp
        h0, h1, h2, h3, h4, h5, epsilon = sp.symbols(r'h_0 h_1 h_2 h_3 h_4 h_5 \epsilon')
        return sp.Matrix([h0, h1, h2, h3, h4, h5, epsilon])

    def to_numpy(self):
        return np.asarray([
            self.h0,
            self.h1,
            self.h2,
            self.h3,
            self.h4,
            self.h5,
            self.epsilon,
        ])

    def to_matrix(self, add_epsilon: bool = True):
        eps = 1j * self.epsilon * int(add_epsilon)
        return self.backend.array([
            [eps, self.h0, self.h1, self.h2],
            [-self.h0, eps, self.h3, self.h4],
            [-self.h1, -self.h3, eps, self.h5],
            [-self.h2, -self.h4, -self.h5, eps],
        ])

    def compute_hamiltonian(self):
        return sum([
                self.h0 * self.h0_op,
                self.h1 * self.h1_op,
                self.h2 * self.h2_op,
                self.h3 * self.h3_op,
                self.h4 * self.h4_op,
                self.h5 * self.h5_op,
                self.epsilon * self.epsilon_op
            ])

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"h0={self.h0}, "
                f"h1={self.h1}, "
                f"h2={self.h2}, "
                f"h3={self.h3}, "
                f"h4={self.h4}, "
                f"h5={self.h5}, "
                f"epsilon={self.epsilon})")
