from typing import Iterable, Tuple, Union

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.ops.op_math import Prod, SProd, Sum
from pennylane.ops.qubit import Projector
from pennylane.pauli import pauli_word_to_string

from .... import utils
from ....typing import TensorLike
from ....utils.majorana import majorana_to_pauli
from ....utils.math import dagger
from ....utils.torch_utils import to_tensor
from ..expval_strategy import ExpvalStrategy
from ._pauli_map import _MAJORANA_COEFFS_MAP, _MAJORANA_INDICES_LAMBDAS


class CliffordExpvalStrategy(ExpvalStrategy):
    NAME = "CliffordExpvalStrategy"

    @staticmethod
    def compute_clifford_expvals(state_prep_op: Operator):
        wires = state_prep_op.wires
        triu_indices = np.triu_indices(2 * len(wires), k=1)

        def clifford_circuit():
            state_prep_op.queue()
            return [qml.expval(majorana_to_pauli(mu) @ majorana_to_pauli(nu)) for mu, nu in zip(*triu_indices)]

        clifford_q_node = qml.QNode(clifford_circuit, device=qml.device("default.clifford", wires=wires))
        return clifford_q_node()

    def __call__(
        self, state_prep_op: Union[qml.StatePrep, qml.BasisState], observable: Operator, **kwargs
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")
        assert "global_sptm" in kwargs, "The global SPTM `global_sptm` must be provided as a keyword argument."
        global_sptm: TensorLike = kwargs["global_sptm"]
        global_sptm = to_tensor(global_sptm, dtype=torch.complex128)
        global_sptm = qml.math.einsum("...ij->...ji", global_sptm)  # TODO: why do I need this transpose?
        expvals = self._compute_full_clifford_expvals(state_prep_op, global_sptm)
        hamiltonian = self._format_observable(observable)

        pauli_kinds = self._hamiltonian_to_pauli_str(hamiltonian)
        majorana_coeffs = np.asarray([_MAJORANA_COEFFS_MAP[p] for p in pauli_kinds])
        majorana_indices = np.asarray(
            [_MAJORANA_INDICES_LAMBDAS[p](min(op.wires.tolist())) for p, op in zip(pauli_kinds, hamiltonian.ops)]
        )
        result = self._compute_sum(
            majorana_coeffs=majorana_coeffs,
            coeffs=hamiltonian.coeffs,
            global_sptm=global_sptm,
            majorana_indices=majorana_indices,
            expvals=expvals,
        )
        return result

    def can_execute(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> bool:
        if isinstance(observable, (Projector,)):
            return False
        hamiltonian = self._format_observable(observable)
        pauli_kinds = self._hamiltonian_to_pauli_str(hamiltonian)
        conditions = [
            all([p in _MAJORANA_COEFFS_MAP for p in pauli_kinds]),
            all(len(p.replace("I", "")) <= 2 for p in pauli_kinds),
        ]
        return all(conditions)

    @staticmethod
    def _format_observable(observable):
        try:
            terms = observable.terms()
        except TermsUndefinedError:
            terms = torch.ones((1,)), [observable]
        return qml.Hamiltonian(*terms)

    @staticmethod
    def _hamiltonian_to_pauli_str(hamiltonian: qml.Hamiltonian):
        return [pauli_word_to_string(op, wire_map={w: i for i, w in enumerate(op.wires)}) for op in hamiltonian.ops]

    def _compute_full_clifford_expvals(self, state_prep_op, global_sptm):
        triu_indices = np.triu_indices(global_sptm.shape[-1], k=1)
        expvals = torch.eye(global_sptm.shape[-1], dtype=global_sptm.dtype, device=global_sptm.device)
        expvals[triu_indices[0], triu_indices[1]] = to_tensor(
            qml.math.stack(self.compute_clifford_expvals(state_prep_op)),
            dtype=global_sptm.dtype,
            device=global_sptm.device,
        )
        expvals[triu_indices[1], triu_indices[0]] = -expvals[triu_indices[0], triu_indices[1]]
        return expvals

    def _compute_sum(self, majorana_coeffs, coeffs, global_sptm, majorana_indices, expvals):
        transition_matrices = qml.math.einsum("...kij->i...kj", global_sptm[..., majorana_indices, :])
        transition_tensor = self._create_transition_tensor(transition_matrices)
        result = qml.math.einsum(
            "k,k,...kij,ij->...",
            to_tensor(majorana_coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            to_tensor(coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            transition_tensor,
            expvals,
        )
        return result

    def _create_transition_tensor(self, transition_matrices: Iterable[TensorLike]) -> TensorLike:
        return utils.recursive_2in_operator(self._outer_prod, transition_matrices)

    @staticmethod
    def _outer_prod(x: TensorLike, y: TensorLike) -> TensorLike:
        return qml.math.einsum("...i,...j->...ij", x, y)
