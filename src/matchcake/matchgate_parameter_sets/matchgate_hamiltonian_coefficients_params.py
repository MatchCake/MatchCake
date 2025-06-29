import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

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
    ATTRS = ["h0", "h1", "h2", "h3", "h4", "h5", "epsilon"]

    def __init__(self, *args, **kwargs):
        args, kwargs = self._maybe_cast_inputs_to_real(args, kwargs)
        super().__init__(*args, **kwargs)

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
        if self.is_batched:
            self._epsilon = value * qml.math.ones_like(self.epsilon)
        else:
            assert qml.math.ndim(value) == 0
            self._epsilon = qml.math.array(value).item()

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
        return pnp.eye(4)

    @classmethod
    def to_sympy(cls):
        import sympy as sp

        h0, h1, h2, h3, h4, h5, epsilon = sp.symbols(r"h_0 h_1 h_2 h_3 h_4 h_5 \epsilon")
        return sp.Matrix([h0, h1, h2, h3, h4, h5, epsilon])

    def to_matrix(self, add_epsilon: bool = True):
        eps = qml.math.array(1j * self.epsilon * int(add_epsilon))[..., np.newaxis, np.newaxis]
        if self.is_batched:
            matrix = pnp.zeros((self.batch_size, 4, 4), dtype=complex)
        else:
            matrix = pnp.zeros((4, 4), dtype=complex)
        matrix = utils.math.convert_and_cast_like(matrix, self.h0)
        matrix = qml.math.cast(matrix, dtype=complex)
        matrix[..., 0, 1] = qml.math.cast(self.h0, dtype=complex)
        matrix[..., 0, 2] = qml.math.cast(self.h1, dtype=complex)
        matrix[..., 0, 3] = qml.math.cast(self.h2, dtype=complex)
        matrix[..., 1, 2] = qml.math.cast(self.h3, dtype=complex)
        matrix[..., 1, 3] = qml.math.cast(self.h4, dtype=complex)
        matrix[..., 2, 3] = qml.math.cast(self.h5, dtype=complex)
        matrix[..., :, :] = matrix[..., :, :] - qml.math.swapaxes(matrix, -2, -1)
        matrix[..., :, :] = matrix[..., :, :] + eps * qml.math.eye(4, like=matrix)[np.newaxis, ...]
        return matrix

    def compute_hamiltonian(self):
        return sum(
            [
                self.h0 * self.h0_op,
                self.h1 * self.h1_op,
                self.h2 * self.h2_op,
                self.h3 * self.h3_op,
                self.h4 * self.h4_op,
                self.h5 * self.h5_op,
                self.epsilon * self.epsilon_op,
            ]
        )
