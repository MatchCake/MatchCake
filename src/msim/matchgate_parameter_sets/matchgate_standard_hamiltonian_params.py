from typing import Union

import numpy as np

from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_hamiltonian_coefficients_params import MatchgateHamiltonianCoefficientsParams
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_params import MatchgateStandardParams


class MatchgateStandardHamiltonianParams(MatchgateParams):
    r"""
    The hamiltonian in the standard form is given by:

    .. math::
        H = \begin{pmatrix}
                \Tilde{h}_0 & 0 & 0 & \Tilde{h}_1 \\
                0 & \Tilde{h}_2 & \Tilde{h}_3 & 0 \\
                0 & \Tilde{h}_4 & \Tilde{h}_5 & 0 \\
                \Tilde{h}_6 & 0 & 0 & \Tilde{h}_7
            \end{pmatrix}

    where the :math:`\Tilde{h}_i` are the parameters and :math:`H` is the hamiltonian matrix of shape
    :math:`2^n \times 2^n` where :math:`n` is the number of particles in the system. In our case, :math:`n=2`.
    """

    def __init__(
            self,
            h0: Union[float, complex],
            h1: Union[float, complex],
            h2: Union[float, complex],
            h3: Union[float, complex],
            h4: Union[float, complex],
            h5: Union[float, complex],
            h6: Union[float, complex],
            h7: Union[float, complex],
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        self._h0 = complex(h0)
        self._h1 = complex(h1)
        self._h2 = complex(h2)
        self._h3 = complex(h3)
        self._h4 = complex(h4)
        self._h5 = complex(h5)
        self._h6 = complex(h6)
        self._h7 = complex(h7)

    @property
    def h0(self) -> Union[float, complex]:
        return self._h0

    @property
    def h1(self) -> Union[float, complex]:
        return self._h1

    @property
    def h2(self) -> Union[float, complex]:
        return self._h2

    @property
    def h3(self) -> Union[float, complex]:
        return self._h3

    @property
    def h4(self) -> Union[float, complex]:
        return self._h4

    @property
    def h5(self) -> Union[float, complex]:
        return self._h5

    @property
    def h6(self) -> Union[float, complex]:
        return self._h6

    @property
    def h7(self) -> Union[float, complex]:
        return self._h7

    @staticmethod
    def parse_from_params(params: 'MatchgateParams', backend="numpy") -> 'MatchgateStandardHamiltonianParams':
        if isinstance(params, MatchgateStandardHamiltonianParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateStandardHamiltonianParams.parse_from_polar_params(params, backend=backend)
        elif isinstance(params, MatchgateStandardParams):
            return MatchgateStandardHamiltonianParams.parse_from_standard_params(params, backend=backend)
        elif isinstance(params, MatchgateHamiltonianCoefficientsParams):
            return MatchgateStandardHamiltonianParams.parse_from_hamiltonian_coefficients_params(
                params, backend=backend
            )
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgateStandardHamiltonianParams.parse_from_composed_hamiltonian_params(params, backend=backend)
        return MatchgateStandardHamiltonianParams(*params, backend=backend)

    @staticmethod
    def parse_from_polar_params(
            params: 'MatchgatePolarParams', backend="numpy"
    ) -> 'MatchgateStandardHamiltonianParams':
        std_params = MatchgateStandardParams.parse_from_params(params)
        return MatchgateStandardHamiltonianParams.parse_from_standard_params(std_params)

    @staticmethod
    def parse_from_standard_params(
            params: 'MatchgateStandardParams', backend="numpy"
    ) -> 'MatchgateStandardHamiltonianParams':
        from scipy.linalg import logm

        std_params = MatchgateStandardParams.parse_from_params(params)
        gate = std_params.to_matrix().astype(complex)
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

    @staticmethod
    def parse_from_hamiltonian_coefficients_params(
            params: 'MatchgateHamiltonianCoefficientsParams', backend="numpy"
    ) -> 'MatchgateStandardHamiltonianParams':
        params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
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

    @staticmethod
    def parse_from_composed_hamiltonian_params(
            params: 'MatchgateComposedHamiltonianParams', backend="numpy"
    ) -> 'MatchgateStandardHamiltonianParams':
        hami_params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
        return MatchgateStandardHamiltonianParams.parse_from_hamiltonian_coefficients_params(hami_params, backend)

    @staticmethod
    def to_sympy():
        import sympy as sp
        h0, h1, h2, h3, h4, h5, h6, h7 = sp.symbols('h_0 h_1 h_2 h_3 h_4 h_5 h_6 h_7')
        return sp.Matrix([h0, h1, h2, h3, h4, h5, h6, h7])

    def to_numpy(self):
        return self.backend.asarray([
            self.h0,
            self.h1,
            self.h2,
            self.h3,
            self.h4,
            self.h5,
            self.h6,
            self.h7,
        ])

    def to_matrix(self):
        return self.backend.asarray([
            [self.h0, 0, 0, self.h1],
            [0, self.h2, self.h3, 0],
            [0, self.h4, self.h5, 0],
            [self.h6, 0, 0, self.h7],
        ])

    def adjoint(self):
        r"""
        Return the adjoint version of the parameters.

        :return: The adjoint parameters.
        """
        return MatchgateStandardHamiltonianParams(
            h0=np.conjugate(self.h0),
            h1=np.conjugate(self.h6),
            h2=np.conjugate(self.h2),
            h3=np.conjugate(self.h4),
            h4=np.conjugate(self.h3),
            h5=np.conjugate(self.h5),
            h6=np.conjugate(self.h1),
            h7=np.conjugate(self.h7),
            backend=self.backend,
        )

    def __repr__(self):
        return (f"MatchgateParams("
                f"h0={self.h0}, "
                f"h1={self.h1}, "
                f"h2={self.h2}, "
                f"h3={self.h3}, "
                f"h4={self.h4}, "
                f"h5={self.h5}, "
                f"h6={self.h6}, "
                f"h7={self.h7})")
