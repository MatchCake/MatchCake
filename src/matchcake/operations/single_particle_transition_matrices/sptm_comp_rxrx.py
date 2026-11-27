import numpy as np
import pennylane as qml

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

    HAS_BULK_FORMULA = True

    @staticmethod
    def compute_sptm(params):
        """Compute the single-particle transition matrices of gates of this class from their angles.

        :param params: The angles, shape (..., 2).
        :return: The matrices, shape (..., 4, 4).
        """
        theta, phi = params[..., 0] / 2, params[..., 1] / 2
        cos_difference, sin_difference = qml.math.cos(phi - theta), qml.math.sin(phi - theta)
        cos_sum, sin_sum = qml.math.cos(phi + theta), qml.math.sin(phi + theta)
        zero = qml.math.zeros_like(cos_sum)
        return SingleParticleTransitionMatrixOperation.matrix_from_rows(
            [
                [cos_difference, zero, zero, -sin_difference],
                [zero, cos_sum, sin_sum, zero],
                [zero, -sin_sum, cos_sum, zero],
                [sin_difference, zero, zero, cos_difference],
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

        self._init_from_params(params, wires=wires, **kwargs)
        self._given_params = params

    def to_matchgate(self):
        from ..comp_rotations import CompRxRx

        return CompRxRx(params=self._given_params, wires=self.wires)
