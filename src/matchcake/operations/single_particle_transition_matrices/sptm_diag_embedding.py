import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmDiagEmbedding(SingleParticleTransitionMatrixOperation):
    r"""
    Embedding operation for the diagonal single-particle transition matrix.
    The input features will be embedded into the diagonal of the single-particle transition matrix of
    size :math:`2N \times 2N`, where :math:`N` is the number of qubits. This operation can then encode
    :math:`2N` features into the diagonal of the single-particle transition matrix.

    :param params: The parameters of the operation

    """

    num_wires = AnyWires

    @classmethod
    def get_n_required_wires(cls, params):
        if isinstance(params, int):
            n_params = params
        else:
            params_shape = qml.math.shape(params)
            params_batched = qml.math.reshape(params, (-1, *params_shape[-2:]))
            params_batched_flatten = qml.math.reshape(params_batched, (params_shape[0], -1))
            n_params = params_batched_flatten.shape[-1]
        return int(np.ceil(n_params / 2))

    def __init__(self, params, wires=None, id=None):
        params_shape = qml.math.shape(params)
        params_batched = qml.math.reshape(params, (-1, *params_shape[-2:]))
        params_batched_flatten = qml.math.reshape(params_batched, (params_shape[0], -1))
        n_required_wires = int(np.ceil(params_batched_flatten.shape[-1] / 2))
        n_wires = len(wires)
        if n_required_wires > n_wires:
            raise ValueError(
                f"Number of wires must be at least {n_required_wires} for the given parameters. "
                f"Got {n_wires} wires."
            )
        matrix = qml.math.zeros((params_shape[0], 2 * n_wires, 2 * n_wires), dtype=complex)
        matrix = qml.math.convert_like(matrix, params)
        indexes = np.arange(2 * n_wires, dtype=int)

        # normalize the params so the sum equals to 2 * n_wires
        params_normed = (
            2 * n_wires * params_batched_flatten / qml.math.sum(params_batched_flatten, axis=-1).reshape(-1, 1)
        )
        matrix[..., indexes, indexes] = qml.math.cast_like(params_normed, matrix)
        super().__init__(matrix, wires=wires, id=id, normalize=False)
