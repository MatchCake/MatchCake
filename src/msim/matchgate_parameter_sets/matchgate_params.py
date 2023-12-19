import importlib
from typing import Any, Optional
import warnings
import pennylane as qml

import numpy as np

from .. import utils


class MatchgateParams:
    r"""
    A matchgate can be represented by several set of parameters and there exists a mapping between them.
    # TODO: add the possibility to batch the parameters.
    """
    N_PARAMS = None
    RANGE_OF_PARAMS = None
    DEFAULT_RANGE_OF_PARAMS = (-1e12, 1e12)
    PARAMS_TYPES = None
    DEFAULT_PARAMS_TYPE = float
    ALLOW_COMPLEX_PARAMS = True
    RAISE_ERROR_IF_INVALID_PARAMS = True
    EQUALITY_ABSOLUTE_TOLERANCE = 1e-4
    EQUALITY_RELATIVE_TOLERANCE = 1e-4
    ELEMENTS_INDEXES = None

    @classmethod
    def get_short_name(cls):
        r"""
        Remove the "Matchgate" prefix from the class name and the "Params" suffix.

        :return: The short name of the class.
        """
        long_name = cls.__name__
        short_name = long_name
        if long_name.lower().startswith('matchgate'):
            short_name = long_name[9:]
        if long_name.lower().endswith('params'):
            short_name = short_name[:-6]
        return short_name

    @classmethod
    def _maybe_cast_to_real(cls, *params, **kwargs):
        is_real = utils.check_if_imag_is_zero(np.array(params))
        real_params = tuple(np.real(np.array(params)))
        if is_real:
            return real_params
        elif kwargs.get("force_cast_to_real", False):
            return qml.math.cast(real_params, dtype=float)
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
    def from_matrix(cls, matrix: np.ndarray, **kwargs) -> 'MatchgateParams':
        elements_indexes_as_array = np.array(cls.ELEMENTS_INDEXES)
        params_arr = matrix[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        return cls.from_numpy(params_arr)

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
        return utils.load_backend_lib(backend)

    def __init__(self, backend='numpy'):
        self._backend = self.load_backend_lib(backend)

    @property
    def backend(self):
        return self._backend
    
    @property
    def is_batched(self):
        return qml.math.ndim(self.to_numpy()) > 1

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
    def random(cls, seed: Optional[int] = None):
        rn_state = np.random.RandomState(seed)
        ranges = cls.RANGE_OF_PARAMS
        types = cls.PARAMS_TYPES
        if ranges is None:
            ranges = [cls.DEFAULT_RANGE_OF_PARAMS for _ in range(cls.N_PARAMS)]
        ranges = np.asarray(ranges, dtype=float)
        if types is None:
            types = [cls.DEFAULT_PARAMS_TYPE for _ in range(cls.N_PARAMS)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vector = np.asarray([
                rn_state.uniform(*r, size=2).view(np.complex128).astype(dtype)
                for r, dtype in zip(ranges, types)
            ]).flatten()
        return cls.from_numpy(vector)

    @classmethod
    def random_numpy(cls, *args, **kwargs):
        return cls.random(*args, **kwargs).to_numpy()

    @classmethod
    def random_batch_numpy(cls, batch_size: int = 1, seed: Optional[int] = None, **kwargs):
        rn_state = np.random.RandomState(seed)
        ranges = cls.RANGE_OF_PARAMS
        types = cls.PARAMS_TYPES
        if ranges is None:
            ranges = [cls.DEFAULT_RANGE_OF_PARAMS for _ in range(cls.N_PARAMS)]
        ranges = np.asarray(ranges, dtype=float)
        if types is None:
            types = [cls.DEFAULT_PARAMS_TYPE for _ in range(cls.N_PARAMS)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = np.asarray([
                rn_state.uniform(*r, size=(batch_size, 2)).view(np.complex128).astype(dtype)
                for r, dtype in zip(ranges, types)
            ]).squeeze().transpose()
        return batch
