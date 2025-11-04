from pennylane import numpy as pnp
from pennylane.wires import Wires

from .. import matchgate_parameter_sets as mps
from .. import utils
from .matchgate_operation import MatchgateOperation

paulis_map = {
    "X": utils.PAULI_X,
    "Y": utils.PAULI_Y,
    "Z": utils.PAULI_Z,
    "I": utils.PAULI_I,
}


class CompPauli(MatchgateOperation):
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, paulis="XX", id=None, *, backend=pnp, **kwargs):
        self._paulis = paulis.upper()
        if len(self._paulis) != 2:
            raise ValueError(f"{self.__class__.__name__} requires two paulis; got {self._paulis}.")
        m_params = mps.MatchgateStandardParams.from_sub_matrices(
            paulis_map[self._paulis[0]], paulis_map[self._paulis[1]]
        )
        in_params = mps.MatchgatePolarParams.parse_from_params(m_params, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, backend=backend, **kwargs)

    def get_implicit_parameters(self):
        return self._paulis

    def __repr__(self):
        """Constructor-call-like representation."""
        if self.parameters:
            params = ", ".join([repr(p) for p in self.get_implicit_parameters()])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name


class CompXX(CompPauli):
    def __init__(self, wires=None, id=None, *, backend=pnp, **kwargs):
        super().__init__(wires=wires, paulis="XX", id=id, backend=backend, **kwargs)


class CompYY(CompPauli):
    def __init__(self, wires=None, id=None, *, backend=pnp, **kwargs):
        super().__init__(wires=wires, paulis="YY", id=id, backend=backend, **kwargs)


class CompZZ(CompPauli):
    def __init__(self, wires=None, id=None, *, backend=pnp, **kwargs):
        super().__init__(wires=wires, paulis="ZZ", id=id, backend=backend, **kwargs)
