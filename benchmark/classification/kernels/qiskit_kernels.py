from typing import Optional, List

import numpy as np
import qiskit
from pennylane.templates.broadcast import PATTERN_TO_WIRES
from pennylane.wires import Wires
from qiskit_aer import AerSimulator

from matchcake.ml.kernels.ml_kernel import (
    MLKernel,
)


class QiskitKernel(MLKernel):
    UNPICKLABLE_ATTRIBUTES = ['_simulator', "_circuit"]

    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._simulator = AerSimulator(
            method='matrix_product_state',
            device='GPU' if self.use_cuda else 'CPU'
        )
        self._circuit = None
        self._x_parameters = None
        self._depth = kwargs.get("depth", None)

    @property
    def depth(self):
        return self._depth

    def pre_initialize(self):
        self._circuit = qiskit.QuantumCircuit(self.size)

    def single_distance(self, x0, x1, **kwargs):
        pass

    def batch_distance(self, x0, x1, **kwargs):
        pass


class QiskitPQCKernel(QiskitKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._x_circuit = None
        self._x_prime_circuit = None

    def _compute_default_size(self):
        _size = max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))
        if _size % 2 != 0:
            _size += 1
        return _size

    def pre_initialize(self):
        self._x_circuit = qiskit.QuantumCircuit(self.size)
        self._x_prime_circuit = qiskit.QuantumCircuit(self.size)
        self._circuit = None

    def ansatz(self, x: List, circuit: qiskit.QuantumCircuit):
        wires = Wires(list(range(self.size)))
        wires_double = PATTERN_TO_WIRES["double"](wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[layer * self.size: (layer + 1) * self.size]

    def make_circuit(self):
        """
        Will create the PQC circuit by adding Ry(theta_j) R_z(theta_j) cnots in the x_circuit with the x parameters.
        Also it will add the adjoint of x_circuit into x_prime_circuit with the x_prime_parameter.

        :return: The whole circuit.
        """

    def initialize_parameters(self):
        self._depth = self.kwargs.get("depth", max(1, (self.X_.shape[-1] // self.size) - 1))
        if self._parameters is None:
            self._parameters = self.parameters_rng.uniform(0.0, 1.0, size=self.X_.shape[-1])
