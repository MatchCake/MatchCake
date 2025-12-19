from typing import List, Tuple

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from ... import utils
from ...typing import TensorLike
from ...utils.majorana import majorana_to_pauli
from .probability_strategy import ProbabilityStrategy


class CliffordSumStrategy(ProbabilityStrategy):
    NAME: str = "CliffordSum"
    REQUIRES_KWARGS = ["all_wires", "state_prep_op", "transition_matrix"]

    @staticmethod
    def compute_clifford_expvals(state_prep_op: Operator, indexes_shape: Tuple[int, ...]) -> np.ndarray:
        wires = state_prep_op.wires

        def clifford_circuit():
            state_prep_op.queue()
            return [
                qml.expval(qml.prod(*[majorana_to_pauli(i) for i in indices])) for indices in np.ndindex(indexes_shape)
            ]

        clifford_q_node = qml.QNode(clifford_circuit, device=qml.device("default.clifford", wires=wires))
        expvals = clifford_q_node()
        return qml.math.stack(expvals).reshape(indexes_shape)

    @staticmethod
    def _basis_state_to_fermi_ops(basis_state: TensorLike, wires: Wires) -> qml.fermi.FermiWord:
        r"""
        ... :math:
            a_j a_j^\dagger = |0\rangle\langle 0|
            a_j^\dagger a_j = |1\rangle\langle 1|

        where :math:`a_j^\dagger` is the jth fermionic creation operator and
        :math:`a_j` is the jth fermionic annihilation operator.
        """
        ops = []
        for w, b in zip(wires, basis_state):
            if qml.math.isclose(b, 0):
                ops.append(qml.fermi.FermiA(int(w)) * qml.fermi.FermiC(int(w)))
            else:
                ops.append(qml.fermi.FermiC(int(w)) * qml.fermi.FermiA(int(w)))
        return qml.math.prod(ops)

    @staticmethod
    def _apply_wicks_contraction(fermi_ops: qml.fermi.FermiWord) -> qml.math.FermiWord:
        # from sympy.physics.secondquant import wicks, AnnihilateFermion, CreateFermion
        # from sympy import lambdify
        # sympy_ops = [
        #     (CreateFermion(position) if kind == '+' else AnnihilateFermion(position))
        #     for (orbital, position), kind in fermi_ops.items()
        # ]
        # fermi_product = qml.math.prod(sympy_ops)
        # contracted_product = wicks(
        #     fermi_product,
        #     simplify_kronecker_deltas=True,
        #     simplify_dummies=True,
        #     keep_only_fully_contracted=True
        # )
        # lambdify(contracted_product.free_symbols, contracted_product, modules="numpy")(0, 1)
        return fermi_ops

    @staticmethod
    def _gather_transition_vectors(fermi_ops: qml.fermi.FermiWord, transition_matrix: TensorLike) -> List[TensorLike]:
        transition_vectors = []
        for key in fermi_ops:
            orbital, position = key
            if fermi_ops[key] == "+":
                transition_vectors.append(qml.math.conjugate(transition_matrix[..., position, :]))
            else:
                transition_vectors.append(transition_matrix[..., position, :])
        return transition_vectors

    @staticmethod
    def _create_transition_tensor(transition_vectors: List[TensorLike]) -> TensorLike:
        outer_prod = lambda x, y: qml.math.einsum("...i,...j->...ij", x, y)
        return utils.recursive_2in_operator(outer_prod, transition_vectors)

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
        state_prep_op: Operator = kwargs["state_prep_op"]
        fermi_ops = self._basis_state_to_fermi_ops(target_binary_state, wires)
        fermi_ops = self._apply_wicks_contraction(fermi_ops)
        transition_vectors = self._gather_transition_vectors(fermi_ops, transition_matrix)
        transition_tensor = self._create_transition_tensor(transition_vectors)

        indexes_shape = tuple([2 * num_wires for _ in range(len(transition_vectors))])
        expvals = self.compute_clifford_expvals(state_prep_op, indexes_shape)
        summands_indices = [Ellipsis, *list(range(len(transition_vectors)))]
        probs = qml.math.einsum(transition_tensor, summands_indices, expvals, summands_indices, [Ellipsis])
        return probs
