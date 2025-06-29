from enum import Enum
from typing import Literal


class MatmulDirectionType(Enum):
    LR = "lr"
    RL = "rl"

    def __eq__(self, other):
        if isinstance(other, MatmulDirectionType):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    @classmethod
    def place_ops(cls, direction: "MatmulDirectionType", *ops):
        if direction == cls.RL:
            return ops
        return reversed(ops)


# The direction of the matrix multiplication. If "lr", the matrix is multiplied from the left to the right.
# If "rl", the matrix is multiplied from the right to the left.
_CIRCUIT_MATMUL_DIRECTION: MatmulDirectionType = MatmulDirectionType.LR
# direction: lr
# U_{p} U_{p-1} ... U_{1} U_{0} |psi>
# => U = U_{p} U_{p-1} ... U_{1} U_{0}

# direction: rl
# U_{p} U_{p-1} ... U_{1} U_{0} |psi>
# => U = U_{0} U_{1} ... U_{p-1} U_{p}

_FOP_MATMUL_DIRECTION: MatmulDirectionType = MatmulDirectionType.RL
