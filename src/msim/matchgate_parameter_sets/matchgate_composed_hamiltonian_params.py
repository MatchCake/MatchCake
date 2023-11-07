import numpy as np

from .matchgate_params import MatchgateParams


class MatchgateComposedHamiltonianParams(MatchgateParams):
    N_PARAMS = 6
    ALLOW_COMPLEX_PARAMS = False

    def __init__(
            self,
            n_x: float = 0.0,
            n_y: float = 0.0,
            n_z: float = 0.0,
            m_x: float = 0.0,
            m_y: float = 0.0,
            m_z: float = 0.0,
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        if not self.ALLOW_COMPLEX_PARAMS:
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

    @classmethod
    def to_sympy(cls):
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
