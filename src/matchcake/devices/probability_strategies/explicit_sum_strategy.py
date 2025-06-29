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


class ExplicitSumStrategy(ProbabilityStrategy):
    NAME: str = "ExplicitSum"
    REQUIRES_KWARGS = ["transition_matrix", "all_wires"]

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
        transition_matrix = kwargs["transition_matrix"]
        self.majorana_getter = kwargs.get("majorana_getter", utils.MajoranaGetter(num_wires, maxsize=256))
        n_workers = kwargs.get("n_workers", 0)

        if len(target_binary_state) > 4:
            warnings.warn(
                f"Computing the probability of a target state with more than 4 bits "
                f"may take a long time. Please consider using the lookup table strategy instead.",
                UserWarning,
            )

        ket_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(system_state)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))
        zero_state = self._create_basis_state(0, num_wires).flatten()
        # TODO: Dont compute the zero explicitly and only take the 00 element of the operator
        bra = utils.recursive_2in_operator(
            qml.math.dot,
            [
                zero_state.T.conj(),
                *[self.majorana_getter(i) for i in bra_majorana_indexes],
            ],
        )
        ket = utils.recursive_2in_operator(
            qml.math.dot,
            [*[self.majorana_getter(i) for i in ket_majorana_indexes], zero_state],
        )

        np_iterator = np.ndindex(tuple([2 * num_wires for _ in range(2 * len(target_binary_state))]))
        sum_elements = pbt.apply_func_multiprocess(
            func=self._compute_partial_prob_of_m_n_vector,
            iterable_of_args=[
                (transition_matrix, m_n_vector, target_binary_state, wires, bra, ket) for m_n_vector in np_iterator
            ],
            nb_workers=n_workers,
            verbose=False,
        )
        target_prob = sum(sum_elements, start=0.0)
        return pnp.real(target_prob)

    def _compute_partial_prob_of_m_n_vector(
        self,
        transition_matrix,
        m_n_vector,
        target_binary_state,
        wires,
        bra,
        ket,
    ):
        inner_op_list = [
            self.majorana_getter((1 - b) * i + b * j, (1 - b) * j + b * i)
            for i, j, b in zip(m_n_vector[::2], m_n_vector[1::2], target_binary_state)
        ]
        inner_product = utils.recursive_2in_operator(qml.math.dot, [bra, *inner_op_list, ket])
        t_wire_m = qml.math.prod(transition_matrix[wires, m_n_vector[::2]])
        t_wire_n = qml.math.prod(pnp.conjugate(transition_matrix[wires, m_n_vector[1::2]]))
        product_coeff = t_wire_m * t_wire_n
        return product_coeff * inner_product
