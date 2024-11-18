import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from ...utils.math import convert_and_cast_like
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmRxRx(SingleParticleTransitionMatrixOperation):

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        return np.random.uniform(0, 2 * np.pi, params_shape)

    def __init__(self, params, wires=None, id=None, **kwargs):
        params_shape = qml.math.shape(params)
        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if len(params_shape) == 1:
            matrix = np.zeros((4, 4), dtype=complex)
        elif len(params_shape) == 2:
            matrix = np.zeros((params_shape[0], 4, 4), dtype=complex)
        else:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        matrix = convert_and_cast_like(matrix, params)
        theta, phi = params[..., 0], params[..., 1]

        matrix[..., 0, 0] = qml.math.cos(phi/2 - theta/2)
        matrix[..., 0, 3] = -qml.math.sin(phi/2 - theta/2)
        matrix[..., 1, 1] = qml.math.cos(phi/2 + theta/2)
        matrix[..., 1, 2] = qml.math.sin(phi/2 + theta/2)
        matrix[..., 2, 1] = -qml.math.sin(phi/2 + theta/2)
        matrix[..., 2, 2] = qml.math.cos(phi/2 + theta/2)
        matrix[..., 3, 0] = qml.math.sin(phi/2 - theta/2)
        matrix[..., 3, 3] = qml.math.cos(phi/2 - theta/2)
        super().__init__(matrix, wires=wires, id=id, **kwargs)

