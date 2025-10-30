import importlib
import warnings
from typing import Any, Literal, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.typing import TensorLike

from .. import utils
from ..utils.torch_utils import to_numpy, to_tensor


class MatchgateParams:
    r"""
    A matchgate can be represented by several set of parameters and there exists a mapping between them.
    """

    N_PARAMS = None
    RANGE_OF_PARAMS = None
    DEFAULT_RANGE_OF_PARAMS = (-1e12, 1e12)
    PARAMS_TYPES = None
    DEFAULT_PARAMS_TYPE = float
    DEFAULT_ARRAY_DTYPE = complex
    ALLOW_COMPLEX_PARAMS = True
    RAISE_ERROR_IF_INVALID_PARAMS = True
    FORCE_CAST_PARAMS_TO_REAL = True
    EQUALITY_ABSOLUTE_TOLERANCE = 1e-4
    EQUALITY_RELATIVE_TOLERANCE = 1e-4
    ELEMENTS_INDEXES = None
    ATTRS = []
    _ATTR_FORMAT = "_{}"
    ATTRS_DEFAULT_VALUES = {}
    UNPICKEABLE_ATTRS = []
    DIVISION_EPSILON = 1e-12

    @classmethod
    def get_short_name(cls):
        r"""
        Remove the "Matchgate" prefix from the class name and the "Params" suffix.

        :return: The short name of the class.
        """
        long_name = cls.__name__
        short_name = long_name
        if long_name.lower().startswith("matchgate"):
            short_name = long_name[9:]
        if long_name.lower().endswith("params"):
            short_name = short_name[:-6]
        return short_name

    @classmethod
    def _maybe_cast_to_real(cls, *params, **kwargs):
        list_params = list(params)
        for i, p in enumerate(params):
            np_params = utils.math.astensor(p)
            is_real = utils.check_if_imag_is_zero(np_params)
            real_params = qml.math.real(np_params)
            if is_real:
                list_params[i] = real_params
            elif kwargs.get("force_cast_to_real", cls.FORCE_CAST_PARAMS_TO_REAL):
                list_params[i] = qml.math.cast(real_params, dtype=float)
            elif cls.RAISE_ERROR_IF_INVALID_PARAMS:
                raise ValueError("The parameters must be real.")
        return tuple(list_params)

    @classmethod
    def to_sympy(cls):
        import sympy as sp

        return sp.symbols(" ".join([f"p{i}" for i in range(cls.N_PARAMS)]))

    @classmethod
    def parse_from_params(cls, params: "MatchgateParams", **kwargs) -> "MatchgateParams":
        from . import transfer_functions

        return transfer_functions.params_to(params, cls, **kwargs)

    @classmethod
    def from_numpy(cls, params: np.ndarray) -> "MatchgateParams":
        return cls(params)

    @classmethod
    def from_tensor(cls, params: TensorLike) -> "MatchgateParams":
        """
        Parse the input tensor to a MatchgateParams object.
        The input is a matchgate in his vector form i.e. the non-zero elements of the matchgate matrix.
        The input tensor has the shape (N_PARAMS,) or (batch_size, N_PARAMS).

        :param params: The tensor representation of the matchgate.
        :return: The parsed parameters.
        """
        return cls(params)

    @classmethod
    def from_vector(cls, params: TensorLike) -> "MatchgateParams":
        """
        Parse the input vector to a MatchgateParams object.
        The input is a matchgate in his vector form i.e. the elements in ELEMENTS_INDEXES order.
        The input vector has the shape (N_PARAMS,) or (batch_size, N_PARAMS).

        :param params: The vector representation of the matchgate.
        :return: The parsed parameters.
        """
        return cls(params)

    @classmethod
    def from_matrix(cls, matrix: TensorLike, **kwargs) -> "MatchgateParams":
        """
        Parse the input matrix to a MatchgateParams object.
        The input is a matchgate in his matrix form of shape (4, 4) or (batch_size, 4, 4).

        :param matrix: The matrix representation of the matchgate.
        :param kwargs: Additional arguments.
        :return: The parsed parameters.
        """
        elements_indexes_as_array = np.array(cls.ELEMENTS_INDEXES)
        params_arr = matrix[..., elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        return cls.from_tensor(params_arr)

    @classmethod
    def parse_from_any(cls, params: Any) -> "MatchgateParams":
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
            return cls.from_numpy(qml.math.array(params))

    @staticmethod
    def load_backend_lib(backend):
        return utils.load_backend_lib(backend)

    @classmethod
    def zeros_numpy(cls, batch_size: Optional[int] = None) -> pnp.ndarray:
        if batch_size is None or batch_size == 0:
            return pnp.zeros(cls.N_PARAMS, dtype=cls.DEFAULT_PARAMS_TYPE)
        return pnp.zeros((batch_size, cls.N_PARAMS), dtype=cls.DEFAULT_PARAMS_TYPE)

    def __init__(self, *args, **kwargs):
        self._set_attrs(args, **kwargs)

    @property
    def batch_size(self) -> Optional[int]:
        if self.is_batched:
            return self.to_numpy().shape[0]
        return None

    @property
    def is_batched(self):
        return qml.math.ndim(self.to_numpy()) > 1

    @property
    def is_cuda(self):
        try:
            import torch
        except ImportError:
            return False
        self_arr = self.to_tensor()
        if isinstance(self_arr, torch.Tensor):
            return self_arr.is_cuda
        return False

    @property
    def requires_grad(self):
        try:
            import torch
        except ImportError:
            return False
        self_arr = self.to_tensor()
        if isinstance(self_arr, torch.Tensor):
            return self_arr.requires_grad
        return False

    @property
    def grad(self):
        try:
            import torch
        except ImportError:
            return None
        self_arr = self.to_vector()
        if isinstance(self_arr, torch.Tensor):
            return self_arr.grad
        return None

    def requires_grad_(self, requires_grad: bool = True):
        try:
            import torch
        except ImportError:
            return self
        for attr in self.ATTRS:
            attr_str = self._ATTR_FORMAT.format(attr)
            attr_tensor = getattr(self, attr_str)
            if isinstance(attr_tensor, torch.Tensor):
                attr_tensor.requires_grad_(requires_grad)
        return self

    def __getstate__(self):
        state = {attr: value for attr, value in self.__dict__.copy().items() if attr not in self.UNPICKEABLE_ATTRS}
        return state

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
        if len(values) > 0:
            values = qml.math.stack(values, axis=0)
        values_size = qml.math.prod(qml.math.shape(values))
        batch_size = self._infer_batch_size_from_input(values, **kwargs)
        if values_size == 0:
            values = self.zeros_numpy(batch_size)
        else:
            values = qml.math.reshape(values, (-1, self.N_PARAMS))
        for i, attr in enumerate(self.ATTRS):
            attr_values = kwargs.get(attr, values[..., i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ones = utils.math.convert_and_cast_like(qml.math.ones_like(values[..., i]), attr_values)
            value = attr_values * ones
            attr_str = self._ATTR_FORMAT.format(attr)
            setattr(self, attr_str, value)
        return self

    def to_numpy(self, dtype=None):
        dtype = dtype or self.DEFAULT_ARRAY_DTYPE
        return to_numpy(self.to_vector(), dtype=dtype)

    def to_tensor(self, dtype=None):
        """
        Return the parameters as a tensor of shape (N_PARAMS,) or (batch_size, N_PARAMS) and as a torch tensor.
        See also to_vector.

        :param dtype: The data type of the tensor. Default is None (torch.cfloat).
        :return: The parameters as a tensor.
        """
        import torch

        dtype = dtype or torch.cfloat
        return to_tensor(self.to_vector(), dtype=dtype)

    def to_vector(self):
        """
        Return the parameters as a vector of shape (N_PARAMS,) or (batch_size, N_PARAMS).
        The vector is the elements in ELEMENTS_INDEXES order.

        :return: The parameters as a vector.
        """
        vector = qml.math.stack([getattr(self, attr) for attr in self.ATTRS], axis=-1)
        # Need to make sure that the grad attribute follows in the new variable
        # try:
        #     import torch
        # except ImportError:
        #     return vector
        # with warnings.catch_warnings():
        #     if isinstance(vector, torch.Tensor) and vector.requires_grad:
        #         grads = [getattr(getattr(self, f"{attr}"), "grad", None) for attr in self.ATTRS]
        #         grads_is_not_none = [grad is not None for grad in grads]
        #         if all(grads_is_not_none):
        #             vector.grad = torch.stack(grads, axis=-1)
        #         elif any(grads_is_not_none):
        #             raise ValueError("The gradients of the parameters are not consistent.")
        return vector

    def to_string(self):
        return str(self)

    def __repr__(self):
        self_np = qml.math.reshape(self.to_numpy(), (-1, self.N_PARAMS)).T
        attrs_str = ", ".join(
            [f"{attr}={np.array2string(v, precision=4, floatmode='maxprec')}" for attr, v in zip(self.ATTRS, self_np)]
        )
        _repr = f"{self.__class__.__name__}({attrs_str}"
        if self.is_batched:
            _repr += f", batch_size={self.batch_size}"
        if self.is_cuda:
            _repr += ", device='cuda'"
        if self.requires_grad:
            _repr += ", requires_grad=True"
        _repr += ")"
        return _repr

    def __str__(self):
        params_as_str = ", ".join([np.array2string(p, precision=4, floatmode="maxprec") for p in self.to_numpy()])
        return f"[{params_as_str}]"

    def __eq__(self, other):
        return qml.math.allclose(
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
        return self.__class__(self.to_vector())

    def __array__(self):
        return self.to_vector()

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
                [rn_state.uniform(*r, size=2).view(np.complex128).astype(dtype) for r, dtype in zip(ranges, types)]
            ).flatten()
        return vector

    @classmethod
    def random_batch_numpy(cls, batch_size: int = 1, seed: Optional[int] = None, **kwargs):
        if batch_size is None or batch_size == 0:
            return cls.random_numpy(seed=seed)
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
            batch = (
                np.asarray(
                    [
                        rn_state.uniform(*r, size=(batch_size, 2)).view(np.complex128).astype(dtype)
                        for r, dtype in zip(ranges, types)
                    ]
                )
                .squeeze()
                .transpose()
            )
        return batch

    def adjoint(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement adjoint.")

    def to_matrix(self) -> TensorLike:
        params_arr = self.to_vector()
        dtype = qml.math.get_dtype_name(params_arr)
        if self.is_batched:
            matrix = pnp.zeros((self.batch_size, 4, 4), dtype=dtype)
        else:
            matrix = pnp.zeros((4, 4), dtype=dtype)
        matrix = qml.math.convert_like(matrix, params_arr)
        elements_indexes_as_array = np.array(self.ELEMENTS_INDEXES)
        matrix[..., elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]] = params_arr
        # Need to make sure that the grad attribute follows in the new variable
        # with warnings.catch_warnings():
        #     try:
        #         import torch
        #     except ImportError:
        #         return matrix
        #     if isinstance(matrix, torch.Tensor) and params_arr.grad is not None:
        #         grads = torch.ones_like(matrix)
        #         grads[..., elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]] = params_arr.grad
        #         matrix.grad = grads
        return matrix

    def to_interface(self, interface: Literal["numpy", "torch"], dtype=None) -> "MatchgateParams":
        if interface == "numpy":
            vec = self.to_numpy(dtype=dtype)
            return self.from_numpy(vec)
        elif interface == "torch":
            vec = self.to_tensor(dtype=dtype)
            return self.from_tensor(vec)
        else:
            raise ValueError(f"Unknown interface: {interface}.")
