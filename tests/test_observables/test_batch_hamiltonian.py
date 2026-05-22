import numpy as np
import pennylane as qml
import pytest

from matchcake.observables import BatchHamiltonian


class TestBatchHamiltonian:
    def test_eigvals(self):
        ops = [qml.PauliZ(0), qml.PauliZ(1)]
        coeffs = [1.0, 1.0]
        h = BatchHamiltonian(coeffs, ops)
        eigvals = h.eigvals()
        assert eigvals.shape[0] == 2

    def test_reduce(self):
        ops = [qml.PauliZ(0), qml.PauliZ(1)]
        coeffs = np.array([0.5, 0.5])
        h = BatchHamiltonian(coeffs, ops)
        expectation_values = np.array([1.0, -1.0])
        result = h.reduce(expectation_values)
        np.testing.assert_allclose(result, 0.0, atol=1e-8)

    def test_name(self):
        ops = [qml.PauliZ(0)]
        h = BatchHamiltonian([1.0], ops)
        assert h.name == "BatchHamiltonian"
