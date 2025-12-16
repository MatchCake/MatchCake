import warnings
from typing import Callable

import numpy as np
import pennylane as qml
import pythonbasictools as pbt
from pennylane import numpy as pnp
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from ...base.lookup_table import NonInteractingFermionicLookupTable
from .probability_strategy import ProbabilityStrategy
from typing import Union

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.ops.op_math import Prod, SProd, Sum
from pennylane.ops.qubit import Projector
from pennylane.pauli import pauli_word_to_string

from ...typing import TensorLike
from ...utils.majorana import majorana_to_pauli


class CliffordSumStrategy(ProbabilityStrategy):
    NAME: str = "CliffordSum"
    REQUIRES_KWARGS = ["global_sptm", "all_wires", "state_prep_op"]

    @staticmethod
    def compute_clifford_expvals(state_prep_op: Operator, majorana_indexes: np.ndarray) -> np.ndarray:
        wires = state_prep_op.wires

        def clifford_circuit():
            state_prep_op.queue()
            return [
                qml.expval(qml.prod(*[majorana_to_pauli(i) for i in indices]))
                for indices in majorana_indexes
            ]

        clifford_q_node = qml.QNode(clifford_circuit, device=qml.device("default.clifford", wires=wires))
        return clifford_q_node()

    def __init__(self):
        self.majorana_getter = None

    def _create_basis_state(self, index, num_wires):
        """
        Create a computational basis state over all wires.

        :param index: integer representing the computational basis state
        :type index: int
        :return: complex array of shape ``[2]*self.num_wires`` representing the statevector of the basis state

        :Note: This function does not support broadcasted inputs yet.
        :Note: This function comes from the ``default.qubit`` device.
        """
        state = np.zeros(2**num_wires, dtype=np.complex128)
        state[index] = 1
        state = qml.math.cast(state, dtype=complex)
        return np.reshape(state, [2] * num_wires)

    def __call__(
        self,
        *,
        system_state: TensorLike,
        target_binary_state: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        self.check_required_kwargs(kwargs)

        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        all_wires = kwargs["all_wires"]
        num_wires = len(all_wires)
        global_sptm = kwargs["global_sptm"]
        state_prep_op: Operator = kwargs["state_prep_op"]
        majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(target_binary_state)
        # np_iterator = np.ndindex(tuple([2 * num_wires for _ in range(2 * len(target_binary_state))]))
        expval_indexes = np.ndindex(tuple([2 * num_wires for _ in range(len(majorana_indexes))]))
        expvals = self.compute_clifford_expvals(state_prep_op, list(expval_indexes))
        print(expvals)
