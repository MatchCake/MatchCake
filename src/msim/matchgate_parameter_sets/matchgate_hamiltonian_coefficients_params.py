import numpy as np

from .matchgate_composed_hamiltonian_params import MatchgateComposedHamiltonianParams
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams


class MatchgateHamiltonianCoefficientsParams(MatchgateParams):
    r"""
    The hamiltionian of a system of non-interacting fermions can be written as:

    .. math::
        H = \sum_{\mu\nu}h_{\mu\nu}c_{\mu}c_{\nu}

    where :math:`c_{\mu}` and :math:`c_{\nu}` are the fermionic majorana operators and :math:`h_{\mu\nu}` are the
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
    N_PARAMS = 6

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
    def parse_from_params(params: 'MatchgateParams', backend="numpy") -> 'MatchgateHamiltonianCoefficientsParams':
        if isinstance(params, MatchgateHamiltonianCoefficientsParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateHamiltonianCoefficientsParams.parse_from_polar_params(params, backend=backend)
        elif isinstance(params, MatchgateStandardParams):
            return MatchgateHamiltonianCoefficientsParams.parse_from_standard_params(params, backend=backend)
        elif isinstance(params, MatchgateComposedHamiltonianParams):
            return MatchgateHamiltonianCoefficientsParams.parse_from_composed_hamiltonian_params(
                params, backend=backend
            )
        elif isinstance(params, MatchgateStandardHamiltonianParams):
            return MatchgateHamiltonianCoefficientsParams.parse_from_standard_hamiltonian_params(
                params, backend=backend
            )
        return MatchgateHamiltonianCoefficientsParams(*params)

    @staticmethod
    def parse_from_polar_params(
            params: 'MatchgatePolarParams',
            backend="numpy"
    ) -> 'MatchgateHamiltonianCoefficientsParams':
        std_params = MatchgateStandardParams.parse_from_params(params, backend=backend)
        std_hamil_params = MatchgateStandardHamiltonianParams.parse_from_standard_params(
            std_params, backend=backend
        )
        return MatchgateHamiltonianCoefficientsParams.parse_from_standard_hamiltonian_params(
            std_hamil_params, backend=backend
        )

    @staticmethod
    def parse_from_standard_params(params: 'MatchgateStandardParams',
                                   backend="numpy") -> 'MatchgateHamiltonianCoefficientsParams':
        std_params = MatchgateStandardParams.parse_from_params(params, backend=backend)
        std_hamil_params = MatchgateStandardHamiltonianParams.parse_from_standard_params(
            std_params, backend=backend
        )
        return MatchgateHamiltonianCoefficientsParams.parse_from_standard_hamiltonian_params(
            std_hamil_params, backend=backend
        )

    @staticmethod
    def parse_from_composed_hamiltonian_params(
            params: 'MatchgateComposedHamiltonianParams',
            backend='numpy',
    ) -> 'MatchgateHamiltonianCoefficientsParams':
        return MatchgateHamiltonianCoefficientsParams(
            h0=params.n_z + params.m_z,
            h1=params.n_y + params.m_y,
            h2=params.n_x - params.m_x,
            h3=params.n_x + params.m_x,
            h4=params.n_y - params.m_y,
            h5=params.n_z - params.m_z,
            backend=backend,
        )

    @staticmethod
    def parse_from_standard_hamiltonian_params(
            params: 'MatchgateStandardHamiltonianParams',
            backend='numpy'
    ) -> 'MatchgateHamiltonianCoefficientsParams':
        params = MatchgateStandardHamiltonianParams.parse_from_params(params, backend=backend)
        # h5 = (params.h0 - params.h2) / 4j
        # h3 = -(1j * params.h3 - 0.5 * params.h1 - 2 * params.h4 - params.h6) / 6j
        # return MatchgateHamiltonianCoefficientsParams(
        #     h0=(params.h0 / 2j) - h5,
        #     h1=0.25 * (1j * params.h3 + params.h1 - params.h4 - 1j * h3),
        #     h2=0.25 * (params.h3 + 1j * params.h4) + h3,
        #     h3=h3,
        #     h4=0.5 * (params.h4 + params.h1) - 2j * h3,
        #     h5=h5,
        #     backend=backend,
        # )
        return MatchgateHamiltonianCoefficientsParams(
            h0=-0.5j * (params.h0 + params.h2),
            h1=0.25 * (params.h6 - params.h1 + params.h4 - params.h3),
            h2=0.25j * (params.h3 + params.h4 - params.h1 - params.h6),
            h3=-0.25j * (params.h3 + params.h4 + params.h1 + params.h6),
            h4=0.25 * (params.h1 - params.h6 + params.h4 - params.h3),
            h5=0.5j * (params.h2 - params.h0),
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
