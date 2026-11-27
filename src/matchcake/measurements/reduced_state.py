from typing import Optional

from pennylane.measurements import StateMeasurement
from pennylane.wires import Wires, WiresLike

from ..typing import TensorLike


class ReducedStateMP(StateMeasurement):
    r"""
    Measurement process that returns the reduced state on a subset of wires as a state preparation.

    Please refer to :func:`reduced_state` for detailed documentation.

    :param wires: The wires to keep. ``None`` keeps all the wires of the device.
    :type wires: Optional[Wires]
    :param reduced_wires: Wire labels of the returned state preparation. ``None`` labels them
        ``range(k)`` for ``k`` kept wires.
    :type reduced_wires: Optional[Wires]
    :param gate_barrier: Whether the returned state preparation carries the gate barriers left by the
        removed wires.
    :type gate_barrier: bool
    """

    _shortname = "reduced_state"

    def __init__(self, wires: Optional[Wires] = None, reduced_wires: Optional[Wires] = None, gate_barrier: bool = True):
        self.reduced_wires = None if reduced_wires is None else Wires(reduced_wires)
        self.gate_barrier = bool(gate_barrier)
        super().__init__(wires=wires)

    def __str__(self) -> str:
        return "reduced_state"

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires) -> TensorLike:
        r"""
        The reduced state of a dense density matrix is not a fermionic Gaussian state in general.

        :param density_matrix: Unused.
        :type density_matrix: TensorLike
        :param wire_order: Unused.
        :type wire_order: Wires
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The reduced_state measurement returns a GaussianStatePrep and is only supported by nif.qubit. On a "
            "dense simulator, use qml.density_matrix to get the reduced density matrix."
        )

    def process_state(self, state: TensorLike, wire_order: Wires) -> TensorLike:
        r"""
        The reduced state of a dense state vector is not a fermionic Gaussian state in general.

        :param state: Unused.
        :type state: TensorLike
        :param wire_order: Unused.
        :type wire_order: Wires
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The reduced_state measurement returns a GaussianStatePrep and is only supported by nif.qubit. On a "
            "dense simulator, use qml.density_matrix to get the reduced density matrix."
        )

    def _flatten(self):
        metadata = (
            ("wires", self.raw_wires),
            ("reduced_wires", self.reduced_wires),
            ("gate_barrier", self.gate_barrier),
        )
        return (None, None), metadata

    @property
    def hash(self) -> int:
        reduced_wires = None if self.reduced_wires is None else tuple(self.reduced_wires.tolist())
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), reduced_wires, self.gate_barrier)
        return hash(fingerprint)


def reduced_state(
    wires: Optional[WiresLike] = None,
    *,
    reduced_wires: Optional[WiresLike] = None,
    gate_barrier: bool = True,
) -> ReducedStateMP:
    r"""
    Reduced state on a subset of the wires, the other wires being traced out, as a state preparation.

    The result is the :class:`~matchcake.operations.state_preparation.GaussianStatePrep` returned by
    ``reduced_state`` of ``nif.qubit``: the covariance matrix of the kept wires, the displacement vector
    when the state has one, and the gate barriers left by the removed wires. It is queued as the first
    operation of another circuit to continue the simulation on the kept wires:

    .. code-block:: python

        @qml.qnode(qml.device("nif.qubit", wires=4))
        def circuit():
            qml.BasisState([1, 0, 1, 0], wires=range(4))
            mc.operations.CompRxRx([0.3, 0.7], wires=[1, 2])
            return mc.reduced_state(wires=[0, 1, 3])

        @qml.qnode(qml.device("nif.qubit", wires=3))
        def follow_up(state):
            qml.apply(state)
            mc.operations.CompRyRy([0.1, 0.2], wires=[0, 1])
            return qml.probs()

        follow_up(circuit())

    Here wire 2 is removed between wires 1 and 3, so the state has a barrier before its last wire. The
    device then raises a :class:`~pennylane.exceptions.DeviceError` for a gate on the last two wires and
    for an observable such as ``qml.X(1) @ qml.X(2)``, whose results would not be those of the reduced
    state of the qubits. Gates on one side of a barrier, probabilities, samples and the observables with
    an even number of :math:`X` and :math:`Y` on the right of every barrier are exact. With
    ``gate_barrier=False``, the state is returned without barriers and with a warning.

    The measurement is analytic, differentiable with torch, and only supported by ``nif.qubit``.

    :param wires: The wires to keep, in the order of the device. ``None`` keeps all the wires.
    :type wires: Optional[WiresLike]
    :param reduced_wires: Wire labels of the returned state preparation. Defaults to ``range(k)`` for
        ``k`` kept wires.
    :type reduced_wires: Optional[WiresLike]
    :param gate_barrier: Whether the returned state preparation carries its gate barriers.
    :type gate_barrier: bool
    :return: The measurement process, whose result is a ``GaussianStatePrep``.
    :rtype: ReducedStateMP
    """
    return ReducedStateMP(
        wires=None if wires is None else Wires(wires),
        reduced_wires=None if reduced_wires is None else Wires(reduced_wires),
        gate_barrier=gate_barrier,
    )
