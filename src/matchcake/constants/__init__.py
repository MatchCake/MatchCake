from typing import Literal

# The direction of the matrix multiplication. If "lr", the matrix is multiplied from the left to the right.
# If "rl", the matrix is multiplied from the right to the left.
_MATMUL_DIRECTION: Literal["rl", "lr"] = "lr"
# direction: lr
# U_{p} U_{p-1} ... U_{1} U_{0} |psi>
# => U = U_{p} U_{p-1} ... U_{1} U_{0}
# => R = U_{p} U_{p-1} ... U_{1} U_{0}

# direction: rl
# U_{p} U_{p-1} ... U_{1} U_{0} |psi>
# => U = U_{0} U_{1} ... U_{p-1} U_{p}
# => R = U_{0} U_{1} ... U_{p-1} U_{p}


