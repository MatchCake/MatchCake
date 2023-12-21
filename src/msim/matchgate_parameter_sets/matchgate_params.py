import importlib
from typing import Any, Optional
import warnings
import pennylane as qml
from pennylane import numpy as pnp

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
    ATTRS = []
    _ATTR_FORMAT = "_{}"
    ATTRS_DEFAULT_VALUES = {}
    
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
        list_params = list(params)
        for i, p in enumerate(params):
            np_params = qml.math.array(p)
            is_real = utils.check_if_imag_is_zero(qml.math.array(np_params))
            real_params = pnp.real(qml.math.array(np_params))
            if is_real:
                list_params[i] = real_params
            elif kwargs.get("force_cast_to_real", False):
                list_params[i] = qml.math.cast(real_params, dtype=float)
            elif cls.RAISE_ERROR_IF_INVALID_PARAMS:
                raise ValueError("The parameters must be real.")
        return tuple(list_params)

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
    
    @classmethod
    def zeros_numpy(cls, batch_size: Optional[int] = None) -> pnp.ndarray:
        if batch_size is None:
            return pnp.zeros_like(cls.random_numpy())
        return pnp.zeros_like(cls.random_batch_numpy(batch_size=batch_size))
    
    def __init__(self, *args, **kwargs):
        self._backend = self.load_backend_lib(kwargs.get("backend", pnp))
        self._set_attrs(args, **kwargs)

    @property
    def backend(self):
        return self._backend
    
    @property
    def batch_size(self) -> Optional[int]:
        if self.is_batched:
            return self.to_numpy().shape[0]
        return None
    
    @property
    def is_batched(self):
        return qml.math.ndim(self.to_numpy()) > 1
    
    def _maybe_cast_inputs_to_real(self, args, kwargs):
        # TODO: cast args as well.
        attr_in_kwargs = [attr for attr in self.ATTRS if attr in kwargs]
        attrs_values = [kwargs[k] for k in attr_in_kwargs]
        if not self.ALLOW_COMPLEX_PARAMS:
            attrs_values = self._maybe_cast_to_real(*attrs_values, **kwargs)
        for attr, value in zip(attr_in_kwargs, attrs_values):
            kwargs[attr] = value
        return args, kwargs
    
    def _infer_batch_size_from_input(self, values, **kwargs):
        values = qml.math.reshape(values, (-1, self.N_PARAMS))
        attr_values = [kwargs.get(attr, []) for attr in self.ATTRS]
        batch_sizes = [values.shape[0]] + [qml.math.reshape(v, (-1,)).shape[0] for v in attr_values]
        batch_sizes = [s for s in batch_sizes if s > 1]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"The batch size of the input parameters is not consistent. Got {batch_sizes} batch sizes."
            )
        batch_size = batch_sizes[0] if batch_sizes else 1
        return batch_size
    
    def _set_attrs(self, values, **kwargs):
        values_size = qml.math.prod(qml.math.shape(values))
        batch_size = self._infer_batch_size_from_input(values, **kwargs)
        if values_size == 0:
            values = self.zeros_numpy(batch_size)
        else:
            values = qml.math.reshape(values, (-1, self.N_PARAMS))
        for i, attr in enumerate(self.ATTRS):
            value = kwargs.get(attr, values[..., i]) * pnp.ones_like(values[..., i])
            attr_str = self._ATTR_FORMAT.format(attr)
            setattr(self, attr_str, value)
        return self

    def to_numpy(self):
        return qml.math.stack([getattr(self, attr) for attr in self.ATTRS], axis=-1)

    def to_string(self):
        return str(self)

    def __repr__(self):
        self_np = self.to_numpy()
        attrs_str = ", ".join([
            f"{attr}={np.array2string(v, precision=4, floatmode='maxprec')}" for attr, v in zip(self.ATTRS, self_np)
        ])
        return f"{self.__class__.__name__}({attrs_str})"

    def __str__(self):
        params_as_str = ", ".join([np.array2string(p, precision=4, floatmode='maxprec') for p in self.to_numpy()])
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
    
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if item in self.ATTRS:
            return getattr(self, self._ATTR_FORMAT.format(item), None)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {item}.")
    
    def __copy__(self):
        return self.__class__(self.to_numpy())
    
    def __array__(self):
        return self.to_numpy()
    
    @classmethod
    def random(cls, *args, **kwargs):
        return cls.from_numpy(cls.random_numpy(*args, **kwargs))
    
    @classmethod
    def random_numpy(cls, *args, **kwargs):
        rn_state = np.random.RandomState(kwargs.get("seed", None))
        ranges = cls.RANGE_OF_PARAMS
        types = cls.PARAMS_TYPES
        if ranges is None:
            ranges = [cls.DEFAULT_RANGE_OF_PARAMS for _ in range(cls.N_PARAMS)]
        ranges = np.asarray(ranges, dtype=float)
        if types is None:
            types = [cls.DEFAULT_PARAMS_TYPE for _ in range(cls.N_PARAMS)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vector = np.asarray(
                [
                    rn_state.uniform(*r, size=2).view(np.complex128).astype(dtype)
                    for r, dtype in zip(ranges, types)
                ]
            ).flatten()
        return vector

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
    
    def adjoint(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement adjoint.")
    
    def to_matrix(self) -> np.ndarray:
        params_arr = self.to_numpy()
        if self.is_batched:
            matrix = self.backend.zeros((self.batch_size, 4, 4), dtype=self.DEFAULT_PARAMS_TYPE)
        else:
            matrix = self.backend.zeros((4, 4), dtype=self.DEFAULT_PARAMS_TYPE)
        elements_indexes_as_array = np.array(self.ELEMENTS_INDEXES)
        matrix[..., elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]] = params_arr
        return matrix
