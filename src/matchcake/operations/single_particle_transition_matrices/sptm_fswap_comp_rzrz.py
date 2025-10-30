import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from ...constants import _CIRCUIT_MATMUL_DIRECTION
from ...utils import make_wires_continuous
from ...utils.math import convert_and_cast_like, dagger
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmFSwapCompRzRz(SingleParticleTransitionMatrixOperation):
    ALLOWED_ANGLES = [-np.pi, np.pi]

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.choice(cls.ALLOWED_ANGLES, size=params_shape)

    def __init__(self, params, wires=None, *, id=None, **kwargs):
        params_shape = qml.math.shape(params)
        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        all_wires = make_wires_continuous(wires)
        n_wires = len(all_wires)

        if len(params_shape) == 1:
            matrix = np.zeros((2 * n_wires, 2 * n_wires), dtype=complex)
        elif len(params_shape) == 2:
            matrix = np.zeros((params_shape[0], 2 * n_wires, 2 * n_wires), dtype=complex)
        else:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if self.hyperparameters.get("check_angles", self.DEFAULT_CHECK_ANGLES):
            self.check_angles(params)
        if self.hyperparameters.get("clip_angles", self.DEFAULT_CLIP_ANGLES):
            params = self.clip_angles(params)

        matrix = qml.math.convert_like(matrix, params)
        rows, cols = np.where(np.eye(2 * n_wires, k=0))
        matrix[..., rows, cols] = 1
        theta, phi = params[..., 0], params[..., 1]

        exp_theta, exp_phi = qml.math.exp(1j * theta), qml.math.exp(1j * phi)
        exp_theta_phi = qml.math.exp(-1j * (theta + phi) / 2)

        # apply this on wire 0
        matrix[..., 0, 0] = (exp_theta + exp_phi) * exp_theta_phi / 2
        matrix[..., 0, 1] = (exp_phi - exp_theta) * exp_theta_phi / 2
        matrix[..., 1, 0] = (exp_theta - exp_phi) * exp_theta_phi / 2
        matrix[..., 1, 1] = (exp_theta + exp_phi) * exp_theta_phi / 2

        # apply this on wire 1
        matrix[..., -2, -2] = qml.math.cos(phi / 2 + theta / 2)
        matrix[..., -2, -1] = qml.math.sin(phi / 2 + theta / 2)
        matrix[..., -1, -2] = -qml.math.sin(phi / 2 + theta / 2)
        matrix[..., -1, -1] = qml.math.cos(phi / 2 + theta / 2)
        super().__init__(matrix, wires=all_wires, id=id, **kwargs)
