import numpy as np
import pennylane as qml

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

    ALLOWED_ANGLES = None
    EQUAL_ALLOWED_ANGLES = None

    HAS_BULK_FORMULA = True

    @staticmethod
    def compute_sptm(params):
        """Compute the single-particle transition matrices of gates of this class from their angles.

        :param params: The angles, shape (..., 2).
        :return: The matrices, shape (..., 4, 4).
        """
        theta, phi = params[..., 0], params[..., 1]
        half_sum, half_difference = (theta + phi) / 2, (theta - phi) / 2
        cos_sum, sin_sum = qml.math.cos(half_sum), qml.math.sin(half_sum)
        cos_difference, sin_difference = qml.math.cos(half_difference), qml.math.sin(half_difference)
        zero = qml.math.zeros_like(cos_sum)
        return SingleParticleTransitionMatrixOperation.matrix_from_rows(
            [
                [cos_sum, sin_sum, zero, zero],
                [-sin_sum, cos_sum, zero, zero],
                [zero, zero, cos_difference, sin_difference],
                [zero, zero, -sin_difference, cos_difference],
            ]
        )

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.uniform(0, 2 * np.pi, params_shape)

    def __init__(self, params, wires=None, **kwargs):
        params_shape = qml.math.shape(params)
        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if len(params_shape) > 2:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if self.hyperparameters.get("check_angles", self.DEFAULT_CHECK_ANGLES):  # pragma: no cover
            self.check_angles(params)  # pragma: no cover
        if self.hyperparameters.get("clip_angles", self.DEFAULT_CLIP_ANGLES):
            params = self.clip_angles(params)

        self._init_from_params(params, wires=wires, **kwargs)
        self._given_params = params

    def to_matchgate(self):
        from ..comp_rotations import CompRzRz

        return CompRzRz(params=self._given_params, wires=self.wires)
