import numpy as np

from .matchgate_hamiltonian_coefficients_params import MatchgateHamiltonianCoefficientsParams
from .matchgate_params import MatchgateParams
from .matchgate_polar_params import MatchgatePolarParams
from .matchgate_standard_hamiltonian_params import MatchgateStandardHamiltonianParams
from .matchgate_standard_params import MatchgateStandardParams


class MatchgateComposedHamiltonianParams(MatchgateParams):
    N_PARAMS = 6

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
    def parse_from_params(
            params: 'MatchgateParams',
            backend="numpy",
    ) -> 'MatchgateComposedHamiltonianParams':
        if isinstance(params, MatchgateComposedHamiltonianParams):
            return params
        elif isinstance(params, MatchgatePolarParams):
            return MatchgateComposedHamiltonianParams.parse_from_polar_params(params, backend=backend)
        elif isinstance(params, MatchgateStandardParams):
            return MatchgateComposedHamiltonianParams.parse_from_standard_params(params, backend=backend)
        elif isinstance(params, MatchgateHamiltonianCoefficientsParams):
            return MatchgateComposedHamiltonianParams.parse_from_hamiltonian_coefficients_params(params,
                                                                                                 backend=backend)
        elif isinstance(params, MatchgateStandardHamiltonianParams):
            return MatchgateComposedHamiltonianParams.parse_from_standard_hamiltonian_params(params, backend=backend)
        return MatchgateComposedHamiltonianParams(*params)

    @staticmethod
    def parse_from_polar_params(
            params: 'MatchgatePolarParams',
            backend='numpy',
    ) -> 'MatchgateComposedHamiltonianParams':
        std_params = MatchgateStandardParams.parse_from_params(params, backend=backend)
        return MatchgateComposedHamiltonianParams.parse_from_standard_params(std_params, backend=backend)

    @staticmethod
    def parse_from_standard_params(
            params: 'MatchgateStandardParams',
            backend='numpy',
    ) -> 'MatchgateComposedHamiltonianParams':
        hamiltonian_params = MatchgateHamiltonianCoefficientsParams.parse_from_standard_params(params, backend=backend)
        return MatchgateComposedHamiltonianParams.parse_from_hamiltonian_coefficients_params(hamiltonian_params,
                                                                                             backend=backend)

    @staticmethod
    def parse_from_hamiltonian_coefficients_params(
            params: 'MatchgateHamiltonianCoefficientsParams',
            backend='numpy',
    ) -> 'MatchgateComposedHamiltonianParams':
        return MatchgateComposedHamiltonianParams(
            n_x=0.5 * (params.h2 + params.h3),
            n_y=0.5 * (params.h1 + params.h4),
            n_z=0.5 * (params.h0 + params.h5),
            m_x=0.5 * (params.h3 - params.h2),
            m_y=0.5 * (params.h1 - params.h4),
            m_z=0.5 * (params.h0 - params.h5),
            backend=backend,
        )

    @staticmethod
    def parse_from_standard_hamiltonian_params(
            params: 'MatchgateStandardHamiltonianParams',
            backend='numpy',
    ) -> 'MatchgateComposedHamiltonianParams':
        std_hamil_params = MatchgateStandardHamiltonianParams.parse_from_params(params, backend=backend)
        hamil_params = MatchgateHamiltonianCoefficientsParams.parse_from_standard_hamiltonian_params(
            std_hamil_params, backend=backend
        )
        return MatchgateComposedHamiltonianParams.parse_from_hamiltonian_coefficients_params(hamil_params,
                                                                                             backend=backend)

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
