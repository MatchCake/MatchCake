from typing import Literal, Sequence

from pennylane.wires import Wires

from .. import utils
from .matchgate_operation import MatchgateOperation

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
            paulis_map[paulis[0]], paulis_map[paulis[1]], wires=wires, id=id, _paulis="".join(paulis), **kwargs
        )

    def __init__(self, *args, **kwargs):  # pragma: no cover
        self._paulis = kwargs.pop("_paulis", None)  # pragma: no cover
        super().__init__(*args, **kwargs)  # pragma: no cover

    def get_implicit_parameters(self):  # pragma: no cover
        return self._paulis  # pragma: no cover

    def __repr__(self):  # pragma: no cover
        """Constructor-call-like representation."""
        if self.parameters:  # pragma: no cover
            params = ", ".join([repr(p) for p in self.get_implicit_parameters()])  # pragma: no cover
            return f"{self.name}({params}, wires={self.wires.tolist()})"  # pragma: no cover
        return f"{self.name}(wires={self.wires.tolist()})"  # pragma: no cover

    def label(self, decimals=None, base_label=None, cache=None):  # pragma: no cover
        return base_label or self.name  # pragma: no cover


class CompXX(CompPauli):
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["X", "X"], wires=wires, id=id, **kwargs)


class CompYY(CompPauli):
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["Y", "Y"], wires=wires, id=id, **kwargs)


class CompZZ(CompPauli):
    def __new__(cls, wires=None, id=None, **kwargs):
        return super().__new__(cls, paulis=["Z", "Z"], wires=wires, id=id, **kwargs)
