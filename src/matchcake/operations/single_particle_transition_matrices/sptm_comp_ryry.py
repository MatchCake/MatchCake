import numpy as np
import pennylane as qml

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompRyRy(SingleParticleTransitionMatrixOperation):
    r"""
    Single Particle Transition Matrix (Sptm) Composition of rotations YY of a matchgate

    .. math::
        U = M(R_Y(\theta), R_Y(\phi))

    where M is the matchgate. The current class represent

    .. math::
        R_{\mu\nu} = \frac{1}{4} \text{Tr}{\left(U c_\mu U^\dagger\right)c_\nu}

    where R is the Sptm and c are the majoranas.
    """

    DEFAULT_CLIP_ANGLES = False
    ALLOWED_ANGLES = [np.pi, 3 * np.pi]
    EQUAL_ALLOWED_ANGLES = [0, np.pi, 2 * np.pi, 3 * np.pi]

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [2]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.choice(cls.ALLOWED_ANGLES, size=params_shape)

    def __init__(self, params, wires=None, **kwargs):
        params_shape = qml.math.shape(params)
        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        if len(params_shape) > 2:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if kwargs.get("check_angles", self.DEFAULT_CHECK_ANGLES):
            self.check_angles(params)
        if kwargs.get("clip_angles", self.DEFAULT_CLIP_ANGLES):
            params = self.clip_angles(params)
        theta, phi = params[..., 0], params[..., 1]
        half_sum, half_difference = (theta + phi) / 2, (theta - phi) / 2
        cos_sum, sin_sum = qml.math.cos(half_sum), qml.math.sin(half_sum)
        cos_difference, sin_difference = qml.math.cos(half_difference), qml.math.sin(half_difference)
        zero = qml.math.zeros_like(cos_sum)
        matrix = self.matrix_from_rows(
            [
                [cos_sum, zero, -sin_sum, zero],
                [zero, cos_difference, zero, sin_difference],
                [sin_sum, zero, cos_sum, zero],
                [zero, -sin_difference, zero, cos_difference],
            ]
        )
        super().__init__(matrix, wires=wires, **kwargs)
        self._given_params = params

    def to_matchgate(self):
        from ..comp_rotations import CompRyRy

        return CompRyRy(params=self._given_params, wires=self.wires)
