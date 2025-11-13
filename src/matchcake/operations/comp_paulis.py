from typing import Sequence, Literal

from pennylane.wires import Wires

from .matchgate_operation import MatchgateOperation
from .. import utils

paulis_map = {
    "X": utils.PAULI_X,
    "Y": utils.PAULI_Y,
    "Z": utils.PAULI_Z,
    "I": utils.PAULI_I,
}


class CompPauli(MatchgateOperation):
    r"""
    Represents a matchgate composition of paulis gates

    .. math::
        U = M(P_0, P_1)

    where :math:`M` is a matchgate, :math:`P_0` and :math:`P_1` are the paulis.

    """
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __new__(cls, paulis: Sequence[Literal["X", "Y", "Z", "I"]], wires=None, id=None, **kwargs):
        if len(paulis) != 2:
            raise ValueError(f"{cls.__name__} requires two paulis; got {paulis}.")
        return cls.from_sub_matrices(
            paulis_map[paulis[0]], paulis_map[paulis[1]],
            wires=wires,
            id=id,
            _paulis=''.join(paulis),
            **kwargs
        )

    def __init__(self, *args, **kwargs):
        self._paulis = kwargs.pop("_paulis", None)
        super().__init__(*args, **kwargs)

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
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["X", "X"], wires=wires, id=id, **kwargs)


class CompYY(CompPauli):
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["Y", "Y"], wires=wires, id=id, **kwargs)


class CompZZ(CompPauli):
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["Z", "Z"], wires=wires, id=id, **kwargs)
