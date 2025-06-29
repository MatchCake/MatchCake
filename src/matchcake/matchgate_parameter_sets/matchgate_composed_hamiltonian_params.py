from .matchgate_params import MatchgateParams


class MatchgateComposedHamiltonianParams(MatchgateParams):
    N_PARAMS = 7
    ALLOW_COMPLEX_PARAMS = False
    DEFAULT_PARAMS_TYPE = float
    DEFAULT_RANGE_OF_PARAMS = (-1e0, 1e0)
    ATTRS = ["n_x", "n_y", "n_z", "m_x", "m_y", "m_z", "epsilon"]

    def __init__(self, *args, **kwargs):
        args, kwargs = self._maybe_cast_inputs_to_real(args, kwargs)
        super().__init__(*args, **kwargs)

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

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @classmethod
    def to_sympy(cls):
        import sympy as sp

        n_x, n_y, n_z, m_x, m_y, m_z, epsilon = sp.symbols(r"n_x n_y n_z m_x m_y m_z \epsilon")
        return sp.Matrix([n_x, n_y, n_z, m_x, m_y, m_z, epsilon])
