import numpy as np
import pennylane as qml
from ...utils.math import convert_and_cast_like
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmRyRy(SingleParticleTransitionMatrixOperation):
    ALLOWED_ANGLES = [-np.pi, np.pi]
    DEFAULT_CHECK_ANGLES = False

    def __init__(self, params, wires=None, id=None, check_angles: bool = DEFAULT_CHECK_ANGLES, **kwargs):
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

        if check_angles:
            if not np.all(np.isin(params, self.ALLOWED_ANGLES)):
                raise ValueError(f"Invalid angles: {params}. Expected: {self.ALLOWED_ANGLES}")

        matrix = convert_and_cast_like(matrix, params)
        theta, phi = params[..., 0], params[..., 1]
        theta_plus_phi = (theta + phi) / 2
        theta_minus_phi = (theta - phi) / 2

        matrix[..., 0, 0] = qml.math.cos(theta_plus_phi)
        matrix[..., 0, 2] = -qml.math.sin(theta_plus_phi)

        matrix[..., 1, 1] = qml.math.cos(theta_minus_phi)
        matrix[..., 1, 2] = -qml.math.sin(theta_minus_phi)

        matrix[..., 2, 0] = qml.math.sin(theta_plus_phi)
        matrix[..., 2, 2] = qml.math.cos(theta_plus_phi)

        matrix[..., 3, 0] = qml.math.sin(theta_minus_phi)
        matrix[..., 3, 3] = qml.math.cos(theta_minus_phi)
        super().__init__(matrix, wires=wires, id=id, **kwargs)

