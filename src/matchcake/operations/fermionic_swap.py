from pennylane.wires import Wires

from .single_particle_transition_matrices.single_particle_transition_matrix import SingleParticleTransitionMatrixOperation
from .single_particle_transition_matrices.sptm_fswap import SptmCompZX
from .matchgate_operation import MatchgateOperation
from ..utils import PAULI_Z, PAULI_X


class CompZX(MatchgateOperation):
    r"""
    Represents a specific quantum operation that is defined using the composition
    of the Pauli-Z and Pauli-X matrices. This operation is usually called the fSWAP
    which is the SWAP gate with an additional phase.

    .. math::
        U = M(Z, X) \\
         = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & -1
        \end{bmatrix}

    """
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __new__(cls, wires=None, id=None, **kwargs):
        return cls.from_sub_matrices(
            PAULI_Z, PAULI_X,
            wires=wires,
            id=id,
            **kwargs
        )

    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SptmCompZX(wires=self.wires, id=self.id, **self.hyperparameters, **self.kwargs)


FermionicSWAP = CompZX
FermionicSWAP.__name__ = "FermionicSWAP"

fSWAP = FermionicSWAP
fSWAP.__name__ = "fSWAP"


def fswap_chain_gen(wires, **kwargs):
    """
    Generate a sequence of fSWAP operations for a given range of wires.

    The `fswap_chain_gen` function creates a generator that yields fSWAP
    operations. The direction of generation is determined by the order of the
    wires in the input list. If the first wire is greater than the second, the
    function generates operations in ascending order of wires; otherwise, it
    generates operations in descending order.

    :param wires: A list or tuple containing two integers that specify the range
                  of wires to operate on. The range is determined by the smaller
                  and larger of the two integers.
    :param kwargs: Additional keyword arguments to be passed to the fSWAP operation.
    :return: A generator that yields fSWAP operations for the specified range and
             direction of wires.
    """
    is_reverse = wires[0] > wires[1]
    wire0, wire1 = list(sorted(wires))
    wires_gen = range(wire0, wire1) if is_reverse else reversed(range(wire0, wire1))
    for tmp_wire0 in wires_gen:
        yield fSWAP(wires=[tmp_wire0, tmp_wire0 + 1], **kwargs)
    return


def fswap_chain(wires, **kwargs):
    """
    Generate and return a list of operations resulting from the fswap_chain_gen function.

    Operations are dynamically generated based on the provided wires and the
    additional keyword arguments. This function serves as a convenient way
    to collect and group all the generated operations into a list and return it.

    :param wires: A list or similar iterable specifying the wires to be affected.
    :type wires: Iterable
    :param kwargs: Additional keyword arguments to configure the behaviour of
                   the operation generation.
    :return: A list of operations generated based on the input wires and keyword
             arguments.
    :rtype: List
    """
    return [op for op in fswap_chain_gen(wires, **kwargs)]
