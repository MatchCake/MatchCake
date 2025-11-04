import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from ...utils.math import convert_and_cast_like
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompRzRz(SingleParticleTransitionMatrixOperation):
    r"""
    Single Particle Transition Matrix (Sptm) Composition of rotations ZZ of a matchgate

    .. math::
        U = M(R_Z(\theta), R_Z(\phi))

    where M is the matchgate. The current class represent

    .. math::
        R_{\mu\nu} = \frac{1}{4} \text{Tr}{\left(M c_\mu M^\dagger\right)c_\nu}

    where R is the Sptm and c are the majoranas.
    """

    ALLOWED_ANGLES = [np.pi, 3 * np.pi]
    EQUAL_ALLOWED_ANGLES = [0, np.pi, 2 * np.pi, 3 * np.pi]

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

        if len(params_shape) == 1:
            matrix = np.zeros((4, 4), dtype=complex)
        elif len(params_shape) == 2:
            matrix = np.zeros((params_shape[0], 4, 4), dtype=complex)
        else:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if self.hyperparameters.get("check_angles", self.DEFAULT_CHECK_ANGLES):
            self.check_angles(params)
        if self.hyperparameters.get("clip_angles", self.DEFAULT_CLIP_ANGLES):
            params = self.clip_angles(params)

        matrix = qml.math.convert_like(matrix, params)
        theta, phi = params[..., 0], params[..., 1]

        exp_theta, exp_phi = qml.math.exp(1j * theta), qml.math.exp(1j * phi)
        exp_theta_phi = qml.math.exp(-1j * (theta + phi) / 2)

        matrix[..., 0, 0] = qml.math.cos(phi / 2 + theta / 2)
        matrix[..., 0, 1] = qml.math.sin(phi / 2 + theta / 2)
        matrix[..., 1, 0] = -qml.math.sin(phi / 2 + theta / 2)
        matrix[..., 1, 1] = qml.math.cos(phi / 2 + theta / 2)

        matrix[..., 2, 2] = (exp_theta + exp_phi) * exp_theta_phi / 2
        matrix[..., 2, 3] = (exp_phi - exp_theta) * exp_theta_phi / 2
        matrix[..., 3, 2] = (exp_theta - exp_phi) * exp_theta_phi / 2
        matrix[..., 3, 3] = (exp_theta + exp_phi) * exp_theta_phi / 2
        super().__init__(matrix, wires=wires, id=id, **kwargs)
