import warnings

import numpy as np
import pennylane as qml
import pythonbasictools as pbt
from pennylane import numpy as pnp
from pennylane.operation import StatePrepBase
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
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

    def can_execute(self, state_prep_op: StatePrepBase) -> bool:
        """Return True for preparations reducible to one system basis state.

        Accepts :class:`~pennylane.BasisState` unconditionally, and a
        :class:`~matchcake.operations.state_preparation.ProductState` only when it is
        unbatched and encodes a computational-basis state. Subclasses of ``ProductState``
        such as :class:`~matchcake.operations.state_preparation.StatePrepFromGates` are
        covered by that same check, so a non-basis preparation like
        :class:`~matchcake.operations.state_preparation.PlusState` falls through to
        :class:`~matchcake.devices.probability_strategies.ProductStateProbabilityStrategy`
        instead of reaching a code path that cannot represent it.

        A batched ``ProductState`` is rejected: the explicit sum contracts a single Majorana
        decomposition of one system state, so a batch of preparations has no single bra/ket
        pair to sum over. Such inputs fall through to
        :class:`~matchcake.devices.probability_strategies.ProductStateProbabilityStrategy`.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :return: True when this strategy can consume the preparation as one system state.
        :rtype: bool
        """
        from pennylane.ops.qubit.state_preparation import BasisState

        from ...operations.state_preparation.product_state import ProductState

        if isinstance(state_prep_op, BasisState):
            return True
        if isinstance(state_prep_op, ProductState):
            if state_prep_op.batch_size is not None:
                return False
            return bool(state_prep_op.is_basis_state)
        return False

    def _compute_single(
        self,
        *,
        state_prep_op: StatePrepBase,
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
                "Computing the probability of a target state with more than 4 bits "
                "may take a long time. Please consider using the lookup table strategy instead.",
                UserWarning,
            )  # pragma: no cover

        system_state = self.system_basis_state_from_state_prep_op(state_prep_op)
        ket_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(system_state)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))
        zero_state = self._create_basis_state(0, num_wires).flatten()
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
