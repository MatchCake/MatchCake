import importlib
from typing import Any

import numpy as np

from .. import utils


class MatchgateParams:
    r"""
    A matchgate can be represented by several set of parameters and there exists a mapping between them.
    """

    @staticmethod
    def _maybe_cast_to_real(*params):
        is_real = utils.check_if_imag_is_zero(np.array(params))
        if is_real:
            return tuple(np.real(np.array(params)))
        else:
            raise ValueError("The parameters must be real.")

    @staticmethod
    def to_sympy():
        raise NotImplementedError("This method must be implemented in the child class.")

    @staticmethod
    def parse_from_params(params: 'MatchgateParams', backend="numpy") -> 'MatchgateParams':
        # TODO: Add the backend argument to the parse_from_params method of the child classes.
        raise NotImplementedError("This method must be implemented in the child class.")

    @classmethod
    def from_numpy(cls, params: np.ndarray) -> 'MatchgateParams':
        tuple_params = tuple(params.flatten())
        return cls(*tuple_params, backend='numpy')

    @staticmethod
    def parse_from_any(params: Any) -> 'MatchgateParams':
        r"""
        Try to parse the input parameters to a MatchgateParams object.

        :param params: The input parameters.
        :type params: Any
        :return: The parsed parameters.
        :rtype: MatchgateParams
        """
        if isinstance(params, MatchgateParams):
            return params
        elif isinstance(params, np.ndarray):
            return MatchgateParams.from_numpy(params)
        else:
            return MatchgateParams.from_numpy(np.asarray(params))

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
        return f"MatchgateParams({str(self)})"

    def __str__(self):
        params_as_str = ", ".join([f"{p:.4f}" for p in self.to_numpy()])
        return f"[{params_as_str}]"

    def __eq__(self, other):
        return np.allclose(self.to_numpy(), other.to_numpy())

    def __hash__(self):
        return hash(self.to_string())

    def __getitem__(self, item):
        return self.to_numpy()[item]

    def __len__(self):
        return self.to_numpy().size

    def __iter__(self):
        return iter(self.to_numpy())
