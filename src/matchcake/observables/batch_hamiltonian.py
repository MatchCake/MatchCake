import pennylane as qml
from pennylane.wires import Wires

from matchcake.utils import get_eigvals_on_z_basis


class BatchHamiltonian(qml.Hamiltonian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wires = Wires([op.wires for op in self.ops], _override=True)

    @property
    def name(self):
        return self.__class__.__name__

    def eigvals(self):
        return qml.math.stack([op.eigvals() for op in self.ops])

    def eigvals_on_z_basis(self):
        return qml.math.stack([get_eigvals_on_z_basis(op) for op in self.ops])

    def reduce(self, expectation_values):
        """
        Use the expectation values and the coefficients of the Hamiltonian to compute the energy.
        """
        return qml.math.einsum("...i,...i->...", self.coeffs, expectation_values)
