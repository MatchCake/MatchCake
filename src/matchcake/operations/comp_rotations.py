from typing import Literal, Sequence, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.wires import Wires

from .single_particle_transition_matrices.single_particle_transition_matrix import SingleParticleTransitionMatrixOperation
from .single_particle_transition_matrices.sptm_comp_rzrz import SptmCompRzRz
from .single_particle_transition_matrices.sptm_comp_rxrx import SptmCompRxRx
from .single_particle_transition_matrices.sptm_comp_ryry import SptmCompRyRy

from .matchgate_operation import MatchgateOperation
from .. import utils


def _make_rot_matrix(param, direction: Literal["X", "Y", "Z"]):
    param_shape = qml.math.shape(param)
    ndim = len(param_shape)
    if ndim not in [0, 1]:
        raise ValueError(f"Invalid number of dimensions {len(param_shape)}.")
    batch_size = param_shape[0] if ndim == 1 else 1
    param = qml.math.reshape(param, (batch_size,))
    matrix = qml.math.cast(qml.math.convert_like(np.zeros((batch_size, 2, 2)), param), dtype=complex)

    if direction == "X":
        matrix[:, 0, 0] = qml.math.cos(param / 2)
        matrix[:, 0, 1] = -1j * qml.math.sin(param / 2)
        matrix[:, 1, 0] = -1j * qml.math.sin(param / 2)
        matrix[:, 1, 1] = qml.math.cos(param / 2)
    elif direction == "Y":
        matrix[:, 0, 0] = qml.math.cos(param / 2)
        matrix[:, 0, 1] = -qml.math.sin(param / 2)
        matrix[:, 1, 0] = qml.math.sin(param / 2)
        matrix[:, 1, 1] = qml.math.cos(param / 2)
    elif direction == "Z":
        matrix[:, 0, 0] = qml.math.exp(-1j * param / 2)
        matrix[:, 1, 1] = qml.math.exp(1j * param / 2)
    else:
        raise ValueError(f"Invalid direction {direction}.")
    if ndim == 0:
        matrix = matrix[0]
    return matrix


def _make_complete_rot_matrix(
        params,
        directions: Sequence[Literal["X", "Y", "Z"]],
):
    params_shape = qml.math.shape(params)
    ndim = len(params_shape)
    if ndim not in [1, 2]:
        raise ValueError(f"Invalid number of dimensions {ndim}.")
    if params_shape[-1] != len(directions) and params_shape[-1] != 2:
        raise ValueError(f"Number of parameters ({params_shape[-1]}) and directions ({len(directions)}) must be equal.")
    batch_size = params_shape[0] if ndim == 2 else 1
    matrix = qml.math.cast(qml.math.convert_like(np.zeros((batch_size, 4, 4)), params), dtype=complex)
    outer_matrices = _make_rot_matrix(
        params[..., 0],
        directions[1],
    )
    outer_matrices = qml.math.reshape(outer_matrices, (batch_size, 2, 2))
    inner_matrices = _make_rot_matrix(
        params[..., 1],
        directions[0],
    )
    inner_matrices = qml.math.reshape(inner_matrices, (batch_size, 2, 2))
    matrix[:, 0, 0] = outer_matrices[:, 0, 0]
    matrix[:, 0, 3] = outer_matrices[:, 0, 1]
    matrix[:, 1, 1] = inner_matrices[:, 0, 0]
    matrix[:, 1, 2] = inner_matrices[:, 0, 1]
    matrix[:, 2, 1] = inner_matrices[:, 1, 0]
    matrix[:, 2, 2] = inner_matrices[:, 1, 1]
    matrix[:, 3, 0] = outer_matrices[:, 1, 0]
    matrix[:, 3, 3] = outer_matrices[:, 1, 1]

    if ndim == 1:
        matrix = matrix[0]
    return matrix


class CompRotation(MatchgateOperation):
    r"""
    Composition of rotations

    .. math::
        U = M(R_{P0}(\theta), R_{P1}(\phi))

    where :math:`M` is a matchgate, :math:`P0` and :math:`P1` are the paulis.
    """

    num_wires = 2
    num_params = 1

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.uniform(0, 2 * np.pi, params_shape)

    @classmethod
    def random(
            cls,
            wires: Wires,
            *,
            batch_size: Optional[int] = None,
            dtype: torch.dtype = torch.complex128,
            device: Optional[torch.device] = None,
            seed: Optional[int] = None,
            **kwargs
    ) -> "CompRotation":
        params = cls.random_params(batch_size=batch_size, dtype=dtype, device=device, seed=seed, **kwargs)
        return cls(params, wires=wires, dtype=dtype, device=device, **kwargs)

    def __init__(
            self,
            params,
            directions: Sequence[Literal["X", "Y", "Z"]],
            wires=None,
            id=None,
            **kwargs
    ):
        shape = qml.math.shape(params)[-1:]
        n_angles = shape[0]
        if n_angles != 2:
            raise ValueError(f"{self.__class__.__name__} requires 2 angles; got {n_angles}.")
        if len(directions) != 2:
            raise ValueError(f"{self.__class__.__name__} requires two directions; got {directions}.")
        super().__init__(
            _make_complete_rot_matrix(params, directions),
            wires=wires,
            id=id,
            _given_params=params,
            _directions=''.join(directions),
            **kwargs
        )
        self._given_params = params
        self._directions = ''.join(directions)

    def get_implicit_parameters(self):
        with torch.no_grad():
            params = self._given_params
            is_real = utils.check_if_imag_is_zero(params)
            if is_real:
                params = qml.math.cast(params, dtype=float)
        return params

    def __repr__(self):
        """Constructor-call-like representation."""
        if self.parameters:
            params = ", ".join([repr(p) for p in self.get_implicit_parameters()])
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

        params = self.get_implicit_parameters()

        if len(qml.math.shape(params[0])) != 0:
            # assume that if the first parameter is matrix-valued, there is only a single parameter
            # this holds true for all current operations and templates unless parameter broadcasting
            # is used
            # TODO[dwierichs]: Implement a proper label for broadcasted operators
            if cache is None or not isinstance(cache.get("matrices", None), list) or len(params) != 1:
                return op_label

            for i, mat in enumerate(cache["matrices"]):
                if qml.math.shape(params[0]) == qml.math.shape(mat) and qml.math.allclose(params[0], mat):
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


class CompRxRx(CompRotation):
    r"""
    Composition of rotation XX which mean a matchgate

    .. math::
        U = M(R_X(\theta), R_X(\phi))
    """

    def __init__(
            self,
            params,
            wires=None,
            id=None,
            **kwargs
    ):
        super().__init__(params, directions=["X", "X"], wires=wires, id=id, **kwargs)

    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SptmCompRxRx(self._given_params, wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)


class CompRyRy(CompRotation):
    r"""
    Composition of rotation YY which mean a matchgate

    .. math::
        U = M(R_Y(\theta), R_Y(\phi))
    """
    def __init__(
            self,
            params,
            wires=None,
            id=None,
            **kwargs
    ):
        super().__init__(params, directions=["Y", "Y"], wires=wires, id=id, **kwargs)

    # def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
    #     return SptmCompRyRy(self._given_params, wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)


class CompRzRz(CompRotation):
    r"""
    Composition of rotation ZZ which mean a matchgate

    .. math::
        U = M(R_Z(\theta), R_Z(\phi))
    """

    def __init__(
            self,
            params,
            wires=None,
            id=None,
            **kwargs
    ):
        super().__init__(params, directions=["Z", "Z"], wires=wires, id=id, **kwargs)

    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SptmCompRzRz(self._given_params, wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)
