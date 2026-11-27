import warnings
from typing import Optional

import pennylane as qml
from pennylane.measurements import StateMeasurement
from pennylane.wires import Wires, WiresLike

from ..typing import TensorLike
from ..utils.covariance_matrix_toolbox import CovarianceMatrixToolbox


class CovarianceMatrixMP(StateMeasurement):
    r"""
    Measurement process that returns the Majorana covariance matrix of the state on a subset of wires.

    Please refer to :func:`covariance_matrix` for detailed documentation.

    :param wires: The wires to keep. ``None`` keeps all the wires of the device.
    :type wires: Optional[Wires]
    :param gate_barrier: Whether to refuse the wires for which the covariance matrix alone, without gate
        barriers, would describe the reduced state of the fermionic modes and not of the qubits.
    :type gate_barrier: bool
    """

    _shortname = "covariance_matrix"

    def __init__(self, wires: Optional[Wires] = None, gate_barrier: bool = True):
        self.gate_barrier = bool(gate_barrier)
        super().__init__(wires=wires)

    def __str__(self) -> str:
        return "covariance_matrix"

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires) -> TensorLike:
        r"""
        Compute the covariance matrix from a dense density matrix.

        :param density_matrix: Density matrix of shape ``(..., 2^n, 2^n)`` on ``wire_order``.
        :type density_matrix: TensorLike
        :param wire_order: Wires of the density matrix, in register order.
        :type wire_order: Wires
        :return: Covariance matrix of shape ``(..., 2k, 2k)`` on the kept wires.
        :rtype: TensorLike
        """
        wire_order = Wires(wire_order)
        wire_indices = wire_order.indices(self.wires) if self.wires else list(range(len(wire_order)))
        CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices, allow_gaps=True)
        if not CovarianceMatrixToolbox.are_wires_contiguous(wire_indices):
            if self.gate_barrier:
                raise ValueError(
                    f"The kept wires (register positions {list(wire_indices)}) are not a contiguous block "
                    f"listed in register order. A covariance matrix cannot carry the gate barriers that the "
                    f"removed wires leave between them. Use matchcake.reduced_state on nif.qubit to get the "
                    f"reduced state with its barriers, or pass gate_barrier=False to get the covariance "
                    f"matrix of the reduced state of the fermionic modes."
                )
            warnings.warn(CovarianceMatrixToolbox.FERMIONIC_REDUCTION_WARNING, UserWarning, stacklevel=2)
        return CovarianceMatrixToolbox.reduce(CovarianceMatrixToolbox.from_density_matrix(density_matrix), wire_indices)

    def process_state(self, state: TensorLike, wire_order: Wires) -> TensorLike:
        r"""
        Compute the covariance matrix from a dense state vector.

        :param state: State vector of shape ``(..., 2^n)`` on ``wire_order``.
        :type state: TensorLike
        :param wire_order: Wires of the state, in register order.
        :type wire_order: Wires
        :return: Covariance matrix of shape ``(..., 2k, 2k)`` on the kept wires.
        :rtype: TensorLike
        """
        return self.process_density_matrix(qml.math.dm_from_state_vector(state), wire_order)

    def shape(self, shots: Optional[int] = None, num_device_wires: int = 0) -> tuple:
        r"""
        Shape of the result for a single, unbatched execution.

        :param shots: Unused, the measurement is analytic.
        :type shots: Optional[int]
        :param num_device_wires: Number of wires of the device, used when all the wires are kept.
        :type num_device_wires: int
        :return: ``(2k, 2k)`` for ``k`` kept wires.
        :rtype: tuple
        """
        n_wires = len(self.wires) if self.wires else num_device_wires
        return 2 * n_wires, 2 * n_wires

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("gate_barrier", self.gate_barrier))
        return (None, None), metadata

    @property
    def hash(self) -> int:
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), self.gate_barrier)
        return hash(fingerprint)

    @property
    def numeric_type(self) -> type:
        return float


def covariance_matrix(wires: Optional[WiresLike] = None, gate_barrier: bool = True) -> CovarianceMatrixMP:
    r"""
    Majorana covariance matrix of the state on a subset of the wires, the other wires being traced out.

    The covariance matrix is :math:`\Lambda_{\mu\nu} = i\, \mathrm{Tr}(\rho\, c_\mu c_\nu)` for
    :math:`\mu \neq \nu`, with the Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and
    :math:`c_{2k+1} = Z_{<k} Y_k` of the kept wires. After a matchgate circuit on a computational-basis
    state, the reduced state on a contiguous block of wires is a fermionic Gaussian state, generally
    mixed, and its covariance matrix describes it completely. The result can be passed to
    :class:`~matchcake.operations.state_preparation.GaussianStatePrep` to start another circuit from the
    reduced state:

    .. code-block:: python

        @qml.qnode(qml.device("nif.qubit", wires=4))
        def circuit():
            qml.BasisState([1, 0, 1, 0], wires=range(4))
            mc.operations.CompRxRx([0.3, 0.7], wires=[1, 2])
            return mc.covariance_matrix(wires=[0, 1])

        @qml.qnode(qml.device("nif.qubit", wires=2))
        def follow_up(reduced_covariance):
            mc.GaussianStatePrep(reduced_covariance, wires=[0, 1])
            mc.operations.CompRyRy([0.1, 0.2], wires=[0, 1])
            return qml.expval(qml.Z(0))

        follow_up(circuit())

    The kept wires must form a contiguous block listed in register order. Otherwise the Jordan-Wigner
    strings of the kept Majorana operators run through the removed wires, and the principal submatrix
    describes the reduced state of the fermionic modes, which is not the reduced state of the qubits.
    A matrix cannot carry the gate barriers that make this safe: use :func:`matchcake.reduced_state` to
    get the reduced state with its barriers, or pass ``gate_barrier=False`` to get the covariance matrix
    of the fermionic reduced state, with a warning.

    The measurement is analytic, differentiable, and supported by ``nif.qubit`` as well as by the dense
    simulators (``default.qubit``, ``default.mixed``).

    :param wires: The wires to keep. ``None`` keeps all the wires of the device.
    :type wires: Optional[WiresLike]
    :param gate_barrier: Whether to refuse kept wires that are not a contiguous block in register order.
        With ``False``, the covariance matrix of the fermionic reduced state is returned for them.
    :type gate_barrier: bool
    :return: The measurement process, of result shape ``(2k, 2k)`` for ``k`` kept wires.
    :rtype: CovarianceMatrixMP
    """
    return CovarianceMatrixMP(wires=None if wires is None else Wires(wires), gate_barrier=gate_barrier)
