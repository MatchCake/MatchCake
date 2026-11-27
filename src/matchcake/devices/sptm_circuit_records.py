from typing import List, Optional, Tuple, Type

import pennylane as qml
from pennylane.typing import TensorLike

from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)


class SptmCircuitRecords:
    r"""
    The single-particle transition matrices of the operations applied to a device, recorded in the
    order they are applied and without being multiplied.

    An operation is recorded either as blocks, the matrices it multiplies some columns of the global
    matrix by, or as a gate: its class and its parameters, without its matrix. The device keeps the
    records until its global matrix is read. Its contraction strategy then decides how they are
    multiplied, see
    :meth:`~matchcake.devices.contraction_strategies.contraction_strategy.ContractionStrategy.build_plan`.
    Recording works the same whether the operations come one by one from a generator or as a list.
    """

    def __init__(self):
        # One entry per block: (first column, block size, gate class or None, parameters or block).
        self._entries: List[Tuple[int, int, Optional[Type[SingleParticleTransitionMatrixOperation]], TensorLike]] = []
        # Number of blocks of each recorded operation, whose blocks act on disjoint columns.
        self._operation_sizes: List[int] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add_gate(
        self, gate_class: Type[SingleParticleTransitionMatrixOperation], start: int, size: int, params: TensorLike
    ) -> None:
        """Record a gate by its class and parameters.

        :param gate_class: The class of the gate, which defines ``compute_sptm``.
        :type gate_class: Type[SingleParticleTransitionMatrixOperation]
        :param start: The first column the gate acts on.
        :type start: int
        :param size: The number of columns the gate acts on.
        :type size: int
        :param params: The parameters of the gate, shape (n_params,) or (batch, n_params).
        :type params: TensorLike
        """
        self._entries.append((start, size, gate_class, params))
        self._operation_sizes.append(1)

    def add_blocks(self, blocks: List[Tuple[int, TensorLike]]) -> None:
        """Record an operation by its blocks.

        :param blocks: ``(start, block)`` pairs, the blocks having shape (2k, 2k) or (batch, 2k, 2k).
        :type blocks: List[Tuple[int, TensorLike]]
        """
        for start, block in blocks:
            self._entries.append((start, qml.math.shape(block)[-1], None, block))
        self._operation_sizes.append(len(blocks))

    def clear(self) -> None:
        """Forget every recorded operation."""
        self._entries = []
        self._operation_sizes = []

    def block_layers(self, dtype_name: str) -> List[List[Tuple[int, TensorLike]]]:
        """Return the recorded blocks operation by operation, the matrix of each gate computed on its own.

        :param dtype_name: The name of the real dtype of the blocks, such as ``"float64"``.
        :type dtype_name: str
        :return: One layer per recorded operation, in order: the ``(start, block)`` pairs of its blocks,
            which act on disjoint columns.
        :rtype: List[List[Tuple[int, TensorLike]]]
        """
        layers = []
        position = 0
        for n_blocks in self._operation_sizes:
            layers.append(
                [
                    (start, qml.math.cast(block if gate_class is None else gate_class.compute_sptm(block), dtype_name))
                    for start, _, gate_class, block in self._entries[position : position + n_blocks]
                ]
            )
            position += n_blocks
        return layers

    @property
    def entries(
        self,
    ) -> List[Tuple[int, int, Optional[Type[SingleParticleTransitionMatrixOperation]], TensorLike]]:
        """The recorded blocks, in the order they are applied.

        :return: One ``(first column, block size, gate class or None, parameters or block)`` per block.
        :rtype: List[Tuple[int, int, Optional[Type[SingleParticleTransitionMatrixOperation]], TensorLike]]
        """
        return self._entries
