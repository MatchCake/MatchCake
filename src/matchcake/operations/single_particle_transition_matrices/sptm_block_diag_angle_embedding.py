import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmBlockDiagAngleEmbedding(SingleParticleTransitionMatrixOperation):
    r"""
    Embedding operation for the block diagonal angle single-particle transition matrix.
    The input features will be embedded into the block diagonal of the single-particle transition matrix of
    size :math:`2N \times 2N`, where :math:`N` is the number of qubits. This operation can then encode
    :math:`N` features into the diagonal of the single-particle transition matrix. Here is the formula for the
    block diagonal angle embedding:

    .. math::
        \begin{align}
        R = \begin{bmatrix}
            \cos(f_0) & -\sin(f_0) & 0 & 0 & \cdots & 0 \\
            \sin(f_0) & \cos(f_0) & 0 & 0 & \cdots & 0 \\
            0 & 0 & \cos(f_1) & -\sin(f_1) & \cdots & 0 \\
            0 & 0 & \sin(f_1) & \cos(f_1) & \cdots & 0 \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & 0 & 0 & \cdots & \cos(f_{N-1}) & -\sin(f_{N-1}) \\
            0 & 0 & 0 & 0 & \cdots & \sin(f_{N-1}) & \cos(f_{N-1})
        \end{bmatrix}

    :param params: The parameters of the operation

    """

    num_wires = AnyWires

    def __init__(self, params, wires=None, id=None):
        params_shape = qml.math.shape(params)
        params_batched = qml.math.reshape(params, (-1, *params_shape[-2:]))
        params_batched_flatten = qml.math.reshape(params_batched, (params_shape[0], -1))
        n_required_wires = int(params_batched_flatten.shape[-1])
        n_wires = len(wires)
        if n_required_wires > n_wires:
            raise ValueError(
                f"Number of wires must be at least {n_required_wires} for the given parameters. "
                f"Got {n_wires} wires."
            )
        matrix = qml.math.zeros((params_shape[0], 2 * n_wires, 2 * n_wires), dtype=complex)
        matrix = qml.math.convert_like(matrix, params)
        diag_indexes = np.arange(2 * n_wires, dtype=int)
        matrix[..., diag_indexes, diag_indexes] = 1

        for p_idx, i in enumerate(range(0, 2 * n_required_wires, 2)):
            matrix[..., i, i] = qml.math.cos(params_batched_flatten[..., p_idx])
            matrix[..., i + 1, i] = qml.math.sin(params_batched_flatten[..., p_idx])
            matrix[..., i, i + 1] = -qml.math.sin(params_batched_flatten[..., p_idx])
            matrix[..., i + 1, i + 1] = qml.math.cos(params_batched_flatten[..., p_idx])

        super().__init__(matrix, wires=wires, id=id, normalize=False)
