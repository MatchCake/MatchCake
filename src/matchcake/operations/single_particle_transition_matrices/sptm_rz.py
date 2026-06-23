import numpy as np
import pennylane as qml

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmRz(SingleParticleTransitionMatrixOperation):
    r"""Single Particle Transition Matrix (Sptm) of a single-qubit :math:`R_Z` rotation.

    A single-qubit :math:`R_Z(\theta) = e^{-i\theta Z / 2}` is a Gaussian (matchgate) operation: its
    generator :math:`Z_k = -i c_{2k} c_{2k+1}` is quadratic in the Majorana operators of qubit
    :math:`k`, so it acts on the single-particle sector as a rotation of that qubit's two Majorana
    modes and leaves every other mode untouched. (Contrast :math:`R_X`/:math:`R_Y`, whose generators
    are a single Majorana dressed by a Jordan-Wigner string and are therefore not Gaussian on one
    qubit.)

    The resulting transition matrix is the :math:`2\times2` block acting on modes
    :math:`(2k, 2k+1)`,

    .. math::
        R = \begin{pmatrix}
            \cos\theta & \sin\theta \\
            -\sin\theta & \cos\theta
        \end{pmatrix},

    where :math:`R_{\mu\nu} = \frac{1}{4} \text{Tr}\left(\left(U c_\mu U^\dagger\right) c_\nu\right)`
    and :math:`c` are the Majorana operators. It is the single-qubit specialisation of
    :class:`~.SptmCompRzRz`, whose :math:`(2k, 2k+1)` block is the rotation by :math:`(\theta+\phi)/2`.
    """

    ALLOWED_ANGLES = None
    EQUAL_ALLOWED_ANGLES = None

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [1]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.uniform(0, 2 * np.pi, params_shape)

    def __init__(self, params, wires=None, **kwargs):
        params_shape = qml.math.shape(params)
        if len(params_shape) not in (0, 1):
            raise ValueError(f"Invalid shape for the parameters: {params_shape}. Expected a scalar or a 1D batch.")

        if self.hyperparameters.get("check_angles", self.DEFAULT_CHECK_ANGLES):  # pragma: no cover
            self.check_angles(params)  # pragma: no cover
        if self.hyperparameters.get("clip_angles", self.DEFAULT_CLIP_ANGLES):
            params = self.clip_angles(params)

        if len(params_shape) == 0:
            matrix = np.zeros((2, 2), dtype=complex)
        else:
            matrix = np.zeros((params_shape[0], 2, 2), dtype=complex)
        matrix = qml.math.cast(qml.math.convert_like(matrix, params), dtype=complex)

        cos_theta, sin_theta = qml.math.cos(params), qml.math.sin(params)
        matrix[..., 0, 0] = cos_theta
        matrix[..., 0, 1] = sin_theta
        matrix[..., 1, 0] = -sin_theta
        matrix[..., 1, 1] = cos_theta
        super().__init__(matrix, wires=wires, **kwargs)
        self._given_params = params
