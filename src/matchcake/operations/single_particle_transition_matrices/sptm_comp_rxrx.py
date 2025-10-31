import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from ...utils.math import convert_and_cast_like, dagger, det, svd
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompRxRx(SingleParticleTransitionMatrixOperation):
    r"""
    This class represents the single-particle transition matrix for the composition Rx, Rx rotation gates.

    This matrix is defined as

    .. math::
        \begin{align}
            M(Rx(2\theta), Rx(2\phi)) = U(2\theta, 2\phi) = \begin{pmatrix}
                \cos(\phi - \theta) & 0 & 0 & -\sin(\phi - \theta) \\
                0 & \cos(\phi + \theta) & \sin(\phi + \theta) & 0 \\
                0 & -\sin(\phi + \theta) & \cos(\phi + \theta) & 0 \\
                \sin(\phi - \theta) & 0 & 0 & \cos(\phi - \theta)
            \end{pmatrix}.
        \end{align}
    """

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.uniform(0, 2 * np.pi, params_shape)

    def __init__(self, params, wires=None, id=None, **kwargs):
        params_shape = qml.math.shape(params)
        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if len(params_shape) == 1:
            matrix = np.zeros((4, 4), dtype=float)
        elif len(params_shape) == 2:
            matrix = np.zeros((params_shape[0], 4, 4), dtype=float)
        else:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        matrix = convert_and_cast_like(matrix, params)
        theta, phi = params[..., 0] / 2, params[..., 1] / 2

        matrix[..., 0, 0] = qml.math.cos(phi - theta)
        matrix[..., 0, 3] = -qml.math.sin(phi - theta)
        matrix[..., 1, 1] = qml.math.cos(phi + theta)
        matrix[..., 1, 2] = qml.math.sin(phi + theta)
        matrix[..., 2, 1] = -qml.math.sin(phi + theta)
        matrix[..., 2, 2] = qml.math.cos(phi + theta)
        matrix[..., 3, 0] = qml.math.sin(phi - theta)
        matrix[..., 3, 3] = qml.math.cos(phi - theta)
        super().__init__(matrix, wires=wires, id=id, **kwargs)
