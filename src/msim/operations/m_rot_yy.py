from typing import Union

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps

from .matchgate_operation import MatchgateOperation
from .. import utils


class MatchgateRotationYY(MatchgateOperation):
    num_wires = 2
    num_params = 2

    def __init__(
            self,
            params: Union[pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            *,
            backend=pnp,
            **kwargs
    ):
        shape = qml.math.shape(params)[-1:]
        n_params = shape[0]
        if n_params != self.num_params:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.num_params} parameters; got {n_params}."
            )
        m_params = mps.MatchgateStandardParams(
            a=pnp.cos(params[0] / 2), b=-pnp.sin(params[0] / 2),
            c=pnp.sin(params[0] / 2), d=pnp.cos(params[0] / 2),
            w=pnp.cos(params[1] / 2), x=-pnp.sin(params[1] / 2),
            y=pnp.sin(params[1] / 2), z=pnp.cos(params[1] / 2),
        )
        in_params = mps.MatchgatePolarParams.parse_from_params(m_params, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, backend=backend, **kwargs)
    
    def get_implicite_parameters(self):
        params = [
            2 * np.arccos(self.standard_params.a),
            2 * np.arccos(self.standard_params.w),
        ]
        is_real = utils.check_if_imag_is_zero(np.array(params))
        if is_real:
            params = qml.math.cast(params, dtype=float)
        return params
    
    def __repr__(self):
        """Constructor-call-like representation."""
        if self.parameters:
            params = ", ".join([repr(p) for p in self.get_implicite_parameters()])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"
    
    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.RX(1.23456, wires=0)
        >>> op.label()
        "RX"
        >>> op.label(decimals=2)
        "RX\n(1.23)"
        >>> op.label(base_label="my_label")
        "my_label"
        >>> op.label(decimals=2, base_label="my_label")
        "my_label\n(1.23)"

        If the operation has a matrix-valued parameter and a cache dictionary is provided,
        unique matrices will be cached in the ``'matrices'`` key list. The label will contain
        the index of the matrix in the ``'matrices'`` list.

        >>> op2 = qml.QubitUnitary(np.eye(2), wires=0)
        >>> cache = {'matrices': []}
        >>> op2.label(cache=cache)
        'U(M0)'
        >>> cache['matrices']
        [tensor([[1., 0.],
         [0., 1.]], requires_grad=True)]
        >>> op3 = qml.QubitUnitary(np.eye(4), wires=(0,1))
        >>> op3.label(cache=cache)
        'U(M1)'
        >>> cache['matrices']
        [tensor([[1., 0.],
                [0., 1.]], requires_grad=True),
        tensor([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]], requires_grad=True)]

        """
        if self.draw_label_params is not None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)
        
        op_label = base_label or self.__class__.__name__

        if self.num_params == 0:
            return op_label

        params = self.get_implicite_parameters()

        if len(qml.math.shape(params[0])) != 0:
            # assume that if the first parameter is matrix-valued, there is only a single parameter
            # this holds true for all current operations and templates unless parameter broadcasting
            # is used
            # TODO[dwierichs]: Implement a proper label for broadcasted operators
            if (
                cache is None
                or not isinstance(cache.get("matrices", None), list)
                or len(params) != 1
            ):
                return op_label

            for i, mat in enumerate(cache["matrices"]):
                if qml.math.shape(params[0]) == qml.math.shape(mat) and qml.math.allclose(
                    params[0], mat
                ):
                    return f"{op_label}(M{i})"

            # matrix not in cache
            mat_num = len(cache["matrices"])
            cache["matrices"].append(params[0])
            return f"{op_label}(M{mat_num})"

        if decimals is None:
            return op_label

        def _format(x):
            try:
                return format(qml.math.toarray(x), f".{decimals}f")
            except ValueError:
                # If the parameter can't be displayed as a float
                return format(x)

        param_string = ",\n".join(_format(p) for p in params)
        return f"{op_label}\n({param_string})"


MRotYY = MatchgateRotationYY
MRotYY.__name__ = "MRotYY"

