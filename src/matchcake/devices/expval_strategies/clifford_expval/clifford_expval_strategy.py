from typing import Union

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator
from pennylane.ops.op_math import Prod, SProd
from pennylane.pauli import pauli_word_to_string

from ....typing import TensorLike
from ....utils.majorana import majorana_to_pauli
from ....utils.torch_utils import to_tensor
from ..expval_strategy import ExpvalStrategy
from ._pauli_map import _MAJORANA_COEFFS_MAP, _MAJORANA_INDICES_LAMBDAS


class CliffordExpvalStrategy(ExpvalStrategy):
    NAME = "CliffordExpvalStrategy"

    def __call__(
        self, state_prep_op: Union[qml.StatePrep, qml.BasisState], observable: Operator, **kwargs
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")
        assert "global_sptm" in kwargs, "The global SPTM `global_sptm` must be provided as a keyword argument."
        global_sptm: TensorLike = kwargs["global_sptm"]
        wires = state_prep_op.wires
        n_qubits = len(wires)
        global_sptm = to_tensor(global_sptm, dtype=torch.complex128)
        clifford_device = qml.device("default.clifford", wires=wires)
        triu_indices = np.triu_indices(2 * n_qubits, k=1)

        @qml.qnode(clifford_device)
        def clifford_circuit():
            state_prep_op.queue()
            return [
                qml.expval(majorana_to_pauli(mu) @ majorana_to_pauli(nu))
                for mu, nu in zip(triu_indices[0], triu_indices[1])
            ]

        expvals = torch.eye(2 * n_qubits, dtype=global_sptm.dtype, device=global_sptm.device)
        expvals[triu_indices] = to_tensor(
            qml.math.stack(clifford_circuit()),
            dtype=global_sptm.dtype,
            device=global_sptm.device,
        )
        expvals[np.tril_indices(2 * n_qubits, k=-1)] = -expvals[triu_indices]
        hamiltonian = self._format_observable(observable)
        pauli_kinds = self._hamiltonian_to_pauli_str(hamiltonian)

        majorana_coeffs = np.asarray([_MAJORANA_COEFFS_MAP[p] for p in pauli_kinds])
        majorana_indices = np.asarray(
            [_MAJORANA_INDICES_LAMBDAS[p](op.wires[0]) for p, op in zip(pauli_kinds, observable.ops)]
        )

        transition_matrices = qml.math.einsum("...kij->i...kj", global_sptm[..., majorana_indices, :])
        transition_tensor = qml.math.einsum("...i,...j->...ij", *transition_matrices)
        result = qml.math.einsum(
            "k,k,...kij,ij->...",
            to_tensor(majorana_coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            to_tensor(hamiltonian.coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            transition_tensor,
            expvals,
        )
        return result

    def can_execute(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> bool:
        if not isinstance(observable, (qml.Hamiltonian, Prod, SProd)):
            return False
        hamiltonian = self._format_observable(observable)
        pauli_kinds = self._hamiltonian_to_pauli_str(hamiltonian)
        return all([p in _MAJORANA_COEFFS_MAP for p in pauli_kinds])

    @staticmethod
    def _format_observable(observable):
        if isinstance(observable, Prod):
            ops = [observable]
            observable_coeffs = torch.ones((1,))
        elif isinstance(observable, SProd):
            ops = [observable.base]
            observable_coeffs = observable.scalar
        else:
            ops = observable.ops
            observable_coeffs = observable.coeffs
        return qml.Hamiltonian(observable_coeffs, ops)

    @staticmethod
    def _hamiltonian_to_pauli_str(hamiltonian: qml.Hamiltonian):
        return [pauli_word_to_string(op, wire_map={w: i for i, w in enumerate(op.wires)}) for op in hamiltonian.ops]
