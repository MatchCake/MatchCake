from typing import List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class DisjointSptmOperation(SingleParticleTransitionMatrixOperation):
    r"""
    The product of single-particle transition matrix operations acting on disjoint wires.

    The operations are kept as separate blocks instead of being written into one matrix padded with
    the identity. Multiplying another operation by this one, ``other @ self`` or ``other @= self``,
    then only updates the columns the blocks act on, see
    :meth:`SingleParticleTransitionMatrixOperation.multiply_columns`. The dense matrix on the
    continuous span of all the wires is built the first time it is asked for, so every other use is
    that of a :class:`SingleParticleTransitionMatrixOperation`.

    :func:`pennylane.equal` compares the dense matrices: ``qml.equal(dense_operation, self)`` is True
    for the same matrix on the same wires, however it is split into blocks. With the arguments the
    other way around it is False, because PennyLane first requires its second argument to be an
    instance of the type of the first.
    """

    def __init__(self, blocks: Sequence[SingleParticleTransitionMatrixOperation], **kwargs):
        """
        :param blocks: The operations, acting on disjoint continuous wires. Their order does not matter.
        :type blocks: Sequence[SingleParticleTransitionMatrixOperation]
        :param kwargs: Hyperparameters of :class:`SingleParticleTransitionMatrixOperation`.
        :raises ValueError: If two blocks share a wire or if there is no block.
        """
        self._blocks = sorted(blocks, key=lambda block: block.cs_wires[0])
        if len(self._blocks) == 0:
            raise ValueError(f"Cannot build a {self.__class__.__name__} without blocks.")
        block_wires = [wire for block in self._blocks for wire in block.cs_wires]
        if len(set(block_wires)) != len(block_wires):
            raise ValueError(f"The blocks of a {self.__class__.__name__} must act on disjoint wires.")
        wires = self.make_wires_continuous(Wires(block_wires))
        self._dense_matrix: Optional[TensorLike] = None
        self._wires = wires
        Operation.__init__(self, *[block.single_particle_transition_matrix for block in self._blocks], wires=wires)
        self._hyperparameters = {
            "clip_angles": kwargs.pop("clip_angles", self.DEFAULT_CLIP_ANGLES),
            "check_angles": kwargs.pop("check_angles", self.DEFAULT_CHECK_ANGLES),
            "check_matrix": kwargs.pop("check_matrix", self.DEFAULT_CHECK_MATRIX),
            "check_real": kwargs.pop("check_real", self.DEFAULT_CHECK_REAL),
        }
        kwargs.pop("normalize", None)
        self.kwargs = kwargs

    def decomposition(self) -> List[Operation]:
        """Decompose this operation into its blocks, which commute since they act on disjoint wires.

        :return: The blocks, queued when a recording is active.
        :rtype: List[Operation]
        """
        if qml.QueuingManager.recording():
            return [qml.apply(block) for block in self._blocks]
        return list(self._blocks)

    def column_blocks(self, wires: Wires) -> List[Tuple[int, TensorLike]]:
        return [pair for block in self._blocks for pair in block.column_blocks(wires)]

    @property
    def _matrix(self) -> TensorLike:
        if self._dense_matrix is None:
            size = 2 * len(self.wires)
            matrix = self.multiply_columns(np.eye(size, dtype=np.int64), self.column_blocks(Wires(self.wires)))
            if matrix is None:
                matrices = [block.single_particle_transition_matrix for block in self._blocks]
                matrices = utils.math.convert_tensors_to_same_type_and_cast_to(matrices, self.casting_priorities)
                batch_sizes = [qml.math.shape(sub)[0] for sub in matrices if qml.math.ndim(sub) > 2]
                shape = (*batch_sizes[:1], size, size)
                matrix = np.zeros(shape)
                matrix[...] = np.eye(size)
                # only the backend is taken from the blocks: an integer block must not make the matrix integer
                matrix = qml.math.convert_like(matrix, matrices[0])
                for (start, _), sub in zip(self.column_blocks(Wires(self.wires)), matrices):
                    matrix[..., start : start + sub.shape[-2], start : start + sub.shape[-1]] = sub
            self._dense_matrix = matrix
        return self._dense_matrix

    @property
    def num_params(self) -> int:  # type: ignore[override]
        return len(self._blocks)

    @property
    def blocks(self) -> Optional[List[SingleParticleTransitionMatrixOperation]]:
        return list(self._blocks)

    @property
    def batch_size(self):
        batch_sizes = [block.batch_size for block in self._blocks if block.batch_size is not None]
        return batch_sizes[0] if batch_sizes else None


SingleParticleTransitionMatrixOperation.DISJOINT_PRODUCT_CLASS = DisjointSptmOperation
