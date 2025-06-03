from typing import Union

import pennylane as qml

from .matchgate_params import MatchgateParams


class MatchgateStandardHamiltonianParams(MatchgateParams):
    r"""
    The hamiltonian in the standard form is given by:

    .. math::
        H = \begin{pmatrix}
                u_0 & 0 & 0 & u_1 \\
                0 & u_2 & u_3 & 0 \\
                0 & u_4 & u_5 & 0 \\
                u_6 & 0 & 0 & u_7
            \end{pmatrix}

    where the :math:`u_i` are the parameters and :math:`H` is the hamiltonian matrix of shape
    :math:`2^n \times 2^n` where :math:`n` is the number of particles in the system. In our case, :math:`n=2`.
    """

    N_PARAMS = 8
    ALLOW_COMPLEX_PARAMS = True
    DEFAULT_RANGE_OF_PARAMS = (-1e12, 1e12)
    PARAMS_TYPES = [float, complex, float, complex, complex, float, complex, float]
    ZEROS_INDEXES = [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    ELEMENTS_INDEXES = [
        (0, 0),
        (0, 3),  # u_0, u_1
        (1, 1),
        (1, 2),  # u_2, u_3
        (2, 1),
        (2, 2),  # u_4, u_5
        (3, 0),
        (3, 3),  # u_6, u_7
    ]
    ATTRS = ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"]

    def __init__(self, *args, **kwargs):
        # real_attrs = [a for i, a in enumerate(self.ATTRS) if self.PARAMS_TYPES[i] == float]
        complex_attrs = [a for i, a in enumerate(self.ATTRS) if self.PARAMS_TYPES[i] == complex]
        real_kwargs = {k: kwargs[k] for k in kwargs if k not in complex_attrs}
        args, real_kwargs = self._maybe_cast_inputs_to_real(args, real_kwargs)
        kwargs.update(real_kwargs)
        super().__init__(*args, **kwargs)

    @property
    def u0(self) -> Union[float, complex]:
        return self._u0

    @property
    def u1(self) -> Union[float, complex]:
        return self._u1

    @property
    def u2(self) -> Union[float, complex]:
        return self._u2

    @property
    def u3(self) -> Union[float, complex]:
        return self._u3

    @property
    def u4(self) -> Union[float, complex]:
        return self._u4

    @property
    def u5(self) -> Union[float, complex]:
        return self._u5

    @property
    def u6(self) -> Union[float, complex]:
        return self._u6

    @property
    def u7(self) -> Union[float, complex]:
        return self._u7

    @classmethod
    def to_sympy(cls):
        import sympy as sp

        u0, u1, u2, u3, u4, u5, u6, u7 = sp.symbols("u_0 u_1 u_2 u_3 u_4 u_5 u_6 u_7")
        return sp.Matrix([u0, u1, u2, u3, u4, u5, u6, u7])

    def adjoint(self):
        r"""
        Return the adjoint version of the parameters.

        :return: The adjoint parameters.
        """
        return MatchgateStandardHamiltonianParams(
            u0=qml.math.conjugate(self.u0),
            u1=qml.math.conjugate(self.u6),
            u2=qml.math.conjugate(self.u2),
            u3=qml.math.conjugate(self.u4),
            u4=qml.math.conjugate(self.u3),
            u5=qml.math.conjugate(self.u5),
            u6=qml.math.conjugate(self.u1),
            u7=qml.math.conjugate(self.u7),
        )
