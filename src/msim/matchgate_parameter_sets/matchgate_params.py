import importlib
from typing import Any
import warnings

import numpy as np

from .. import utils


class MatchgateParams:
    r"""
    A matchgate can be represented by several set of parameters and there exists a mapping between them.
    """
    N_PARAMS = None
    RAISE_ERROR_IF_INVALID_PARAMS = True
    EQUALITY_ABSOLUTE_TOLERANCE = 1e-4
    EQUALITY_RELATIVE_TOLERANCE = 1e-4

    @classmethod
    def _maybe_cast_to_real(cls, *params):
        is_real = utils.check_if_imag_is_zero(np.array(params))
        real_params = tuple(np.real(np.array(params)))
        if is_real:
            return real_params
        elif cls.RAISE_ERROR_IF_INVALID_PARAMS:
            raise ValueError("The parameters must be real.")
        else:
            warnings.warn("The parameters must be real.")
            return tuple(np.array(params))

    @classmethod
    def to_sympy(cls):
        import sympy as sp
        return sp.symbols(' '.join([f'p{i}' for i in range(cls.N_PARAMS)]))

    @classmethod
    def parse_from_params(cls, params: 'MatchgateParams', **kwargs) -> 'MatchgateParams':
        from . import transfer_functions
        return transfer_functions.params_to(params, cls, **kwargs)

    @classmethod
    def from_numpy(cls, params: np.ndarray) -> 'MatchgateParams':
        tuple_params = tuple(params.flatten())
        return cls(*tuple_params, backend='numpy')

    @classmethod
    def parse_from_any(cls, params: Any) -> 'MatchgateParams':
        r"""
        Try to parse the input parameters to a MatchgateParams object.

        :param params: The input parameters.
        :type params: Any
        :return: The parsed parameters.
        :rtype: MatchgateParams
        """
        if isinstance(params, cls):
            return params
        elif isinstance(params, np.ndarray):
            return cls.from_numpy(params)
        elif isinstance(params, MatchgateParams):
            return cls.parse_from_params(params)
        else:
            return cls.from_numpy(np.asarray(params))

    @staticmethod
    def load_backend_lib(backend):
        if isinstance(backend, str):
            backend = importlib.import_module(backend)
        return backend

    def __init__(self, backend='numpy'):
        self._backend = self.load_backend_lib(backend)

    @property
    def backend(self):
        return self._backend

    def to_numpy(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def to_string(self):
        return str(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self):
        params_as_str = ", ".join([f"{p:.4f}" for p in self.to_numpy()])
        return f"[{params_as_str}]"

    def __eq__(self, other):
        return np.allclose(
            self.to_numpy(),
            other.to_numpy(),
            atol=self.EQUALITY_ABSOLUTE_TOLERANCE,
            rtol=self.EQUALITY_RELATIVE_TOLERANCE,
        )

    def __hash__(self):
        return hash(self.to_string())

    def __getitem__(self, item):
        return self.to_numpy()[item]

    def __len__(self):
        return self.to_numpy().size

    def __iter__(self):
        return iter(self.to_numpy())

    @classmethod
    def random(cls):
        return cls.from_numpy(np.random.rand(cls.N_PARAMS))
