from typing import Any, Literal

from ..constants import (
    _CIRCUIT_MATMUL_DIRECTION,
    _FOP_MATMUL_DIRECTION,
    MatmulDirectionType,
)
from ..operations.matchgate_operation import MatchgateOperation
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..utils.math import (
    circuit_matmul,
    convert_and_cast_like,
    dagger,
    fermionic_operator_matmul,
)


def circuit_or_fop_matmul(
    first_matrix: Any,
    second_matrix: Any,
    *,
    fop_direction: MatmulDirectionType = _FOP_MATMUL_DIRECTION,
    circuit_direction: MatmulDirectionType = _CIRCUIT_MATMUL_DIRECTION,
    operator: Literal["einsum", "matmul", "@"] = "@",
):
    """
    Matmul two operators together. The direction of the matmul will depend on the type of the
    operators. If both operators are MatchgateOperations, the direction of the matmul will be
    determined by the `circuit_direction` parameter. If both operators are SingleParticleTransitionMatrixOperations,
    the direction of the matmul will be determined by the `fop_direction` parameter. If one operator is a
    MatchgateOperation and the other is a SingleParticleTransitionMatrixOperation, the MatchgateOperation
    will be converted to a SingleParticleTransitionMatrixOperation and the matmul will be performed in the
    direction of the SingleParticleTransitionMatrixOperation.

    If the type of the operator is not recognized, a ValueError will be raised.
    """
    if isinstance(first_matrix, MatchgateOperation) and isinstance(second_matrix, MatchgateOperation):
        return circuit_matmul(first_matrix, second_matrix, direction=circuit_direction, operator=operator)

    if isinstance(first_matrix, SingleParticleTransitionMatrixOperation) and isinstance(
        second_matrix, SingleParticleTransitionMatrixOperation
    ):
        return fermionic_operator_matmul(first_matrix, second_matrix, direction=fop_direction, operator=operator)

    if isinstance(first_matrix, MatchgateOperation):
        first_matrix = first_matrix.to_sptm_operation()

    if isinstance(second_matrix, MatchgateOperation):
        second_matrix = second_matrix.to_sptm_operation()

    if not isinstance(first_matrix, SingleParticleTransitionMatrixOperation) or not isinstance(
        second_matrix, SingleParticleTransitionMatrixOperation
    ):
        raise ValueError(f"Cannot multiply {type(first_matrix)} with {type(second_matrix)}")

    return fermionic_operator_matmul(first_matrix, second_matrix, direction=fop_direction, operator=operator)
