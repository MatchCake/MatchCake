import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.operations import (
    CompRxRx,
    CompRyRy,
    CompRzRz,
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
)
from matchcake.operations.state_preparation import GaussianStatePrep, ProductState
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox

from ....configs import (
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    TEST_SEED,
    set_seed,
)

N_WIRES = 4
KEPT_WIRES = [1, 2]
BASIS_STATE = np.array([1, 0, 1, 0])


@pytest.mark.filterwarnings("ignore:The reduced state is returned without gate barriers:UserWarning")
class TestNonInteractingFermionicDeviceGaussianStateGradients:
    r"""
    Guard the gradient chain through the Gaussian state preparation and the reduced-state
    measurements with ``torch.autograd.gradcheck``.

    The differentiated inputs are the parameters of the circuit gates, the amplitudes of a
    :class:`ProductState` and the covariance matrix of a :class:`GaussianStatePrep`.
    """

    MEASUREMENTS = {
        "expval": lambda: qml.expval(qml.X(1) @ qml.X(2) + 0.5 * qml.Z(0) @ qml.Z(3)),
        "probs": lambda: qml.probs(wires=[0, 1]),
        "covariance_matrix": lambda: mc.covariance_matrix(wires=KEPT_WIRES),
        "fermionic_covariance_matrix": lambda: mc.covariance_matrix(wires=[0, 2], gate_barrier=False),
        "vn_entropy": lambda: qml.vn_entropy(wires=KEPT_WIRES),
        "purity": lambda: qml.purity(wires=KEPT_WIRES),
        "density_matrix": lambda: qml.density_matrix(wires=KEPT_WIRES),
    }

    @staticmethod
    def as_real_outputs(result):
        """Split a possibly complex result into real tensors, as ``gradcheck`` compares real outputs.

        :param result: A tensor returned by a QNode.
        :return: The tensor if it is real, otherwise its real and imaginary parts.
        """
        if torch.is_complex(result):
            return torch.real(result), torch.imag(result)
        return result

    @staticmethod
    def matchgates(angles):
        """Queue an entangling matchgate circuit on ``N_WIRES`` wires.

        :param angles: Tensor of shape ``(..., 6)``.
        """
        CompRxRx(angles[..., 0:2], wires=[0, 1])
        CompRyRy(angles[..., 2:4], wires=[1, 2])
        CompRzRz(angles[..., 4:6], wires=[2, 3])
        CompRxRx(angles[..., 1:3], wires=[2, 3])
        CompRyRy(angles[..., 3:5], wires=[0, 1])
        CompRxRx(angles[..., 0:2] * angles[..., 4:6], wires=[1, 2])

    @staticmethod
    def sptm_gates(angles):
        """Queue an entangling circuit of single-particle transition matrix operations.

        :param angles: Tensor of shape ``(..., 6)``.
        """
        SptmCompRxRx(angles[..., 0:2], wires=[0, 1])
        SptmCompRyRy(angles[..., 2:4], wires=[1, 2])
        SptmCompRxRx(angles[..., 4:6], wires=[2, 3])
        SptmCompRyRy(angles[..., 1:3], wires=[0, 1])
        SptmCompRxRx(angles[..., 3:5], wires=[1, 2])

    @staticmethod
    def random_angles(batch_size=None) -> torch.Tensor:
        """Draw the angles of a test circuit.

        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :return: Tensor of shape ``(6,)`` or ``(batch_size, 6)`` that requires a gradient.
        :rtype: torch.Tensor
        """
        rng = np.random.default_rng(TEST_SEED)
        shape = (6,) if batch_size is None else (batch_size, 6)
        return torch.tensor(rng.uniform(0.2, 1.2, size=shape), dtype=torch.float64, requires_grad=True)

    @staticmethod
    def random_covariance(n_wires: int, batch_size=None) -> torch.Tensor:
        """Build a random mixed-state covariance matrix.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :return: Tensor of shape ``(2 * n_wires, 2 * n_wires)`` or batched, that requires a gradient.
        :rtype: torch.Tensor
        """
        rng = np.random.default_rng(TEST_SEED)
        shape = (n_wires,) if batch_size is None else (batch_size, n_wires)
        polarizations = rng.uniform(-0.8, 0.8, size=shape)
        rotation = SingleParticleTransitionMatrixOperation.random_params(
            batch_size=batch_size, wires=list(range(n_wires)), seed=TEST_SEED
        ).real
        covariance = np.einsum(
            "...ij,...ik,...kl->...jl",
            rotation,
            np.asarray(CovarianceMatrixToolbox.from_polarizations(polarizations)),
            rotation,
        )
        return torch.tensor(covariance, dtype=torch.float64, requires_grad=True)

    def setup_method(self):
        set_seed(TEST_SEED)

    def assert_gradcheck(self, function, inputs):
        assert torch.autograd.gradcheck(function, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True)

    @pytest.mark.parametrize("measurement_name", list(MEASUREMENTS))
    @pytest.mark.parametrize("gates_name", ["matchgates", "sptm_gates"])
    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_gate_parameters_with_basis_state(self, measurement_name, gates_name, batch_size):
        gates = getattr(self, gates_name)
        measurement = self.MEASUREMENTS[measurement_name]

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(angles):
            qml.BasisState(BASIS_STATE, wires=range(N_WIRES))
            gates(angles)
            return measurement()

        self.assert_gradcheck(lambda angles: self.as_real_outputs(circuit(angles)), (self.random_angles(batch_size),))

    @pytest.mark.parametrize("measurement_name", ["expval", "probs"])
    @pytest.mark.parametrize("state_wires", [[0, 1, 2, 3], [1, 3]])
    def test_product_state_amplitudes(self, measurement_name, state_wires):
        measurement = self.MEASUREMENTS[measurement_name]
        rng = np.random.default_rng(TEST_SEED)
        bloch_angles = torch.tensor(
            rng.uniform(0.2, 1.2, size=(len(state_wires), 2)), dtype=torch.float64, requires_grad=True
        )
        gate_angles = self.random_angles()

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_bloch_angles, angles):
            polar, azimuth = raw_bloch_angles[..., 0], raw_bloch_angles[..., 1]
            amplitudes = torch.stack([torch.cos(polar / 2), torch.exp(1j * azimuth) * torch.sin(polar / 2)], dim=-1)
            ProductState(amplitudes, wires=state_wires)
            self.matchgates(angles)
            return measurement()

        self.assert_gradcheck(circuit, (bloch_angles, gate_angles))

    @pytest.mark.parametrize(
        "measurement_name", ["covariance_matrix", "vn_entropy", "purity", "density_matrix", "expval", "probs"]
    )
    def test_basis_product_state_phases_and_gate_parameters(self, measurement_name):
        r"""
        The reduced-state measurements need an initial state without displacement. The phases of a
        computational-basis :class:`ProductState` keep it one, so they can be differentiated together
        with the gate parameters.
        """
        measurement = self.MEASUREMENTS[measurement_name]
        phases = torch.tensor([0.3, -0.6, 1.1, 0.2], dtype=torch.float64, requires_grad=True)
        bits = torch.tensor(BASIS_STATE)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_phases, angles):
            phase_factors = torch.exp(1j * raw_phases)
            amplitudes = torch.stack([phase_factors * (1 - bits), phase_factors * bits], dim=-1)
            ProductState(amplitudes, wires=range(N_WIRES))
            self.matchgates(angles)
            return measurement()

        self.assert_gradcheck(
            lambda raw_phases, angles: self.as_real_outputs(circuit(raw_phases, angles)),
            (phases, self.random_angles()),
        )

    @pytest.mark.parametrize(
        "measurement_name, batch_size",
        [(measurement_name, None) for measurement_name in MEASUREMENTS] + [("expval", 2), ("vn_entropy", 2)],
    )
    def test_gaussian_state_covariance_and_gate_parameters(self, measurement_name, batch_size):
        measurement = self.MEASUREMENTS[measurement_name]
        covariance = self.random_covariance(N_WIRES, batch_size)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_covariance, angles):
            GaussianStatePrep(0.5 * (raw_covariance - raw_covariance.mT), wires=range(N_WIRES))
            self.matchgates(angles)
            return measurement()

        self.assert_gradcheck(
            lambda raw_covariance, angles: self.as_real_outputs(circuit(raw_covariance, angles)),
            (covariance, self.random_angles(batch_size)),
        )

    def test_gaussian_state_on_part_of_the_wires(self):
        covariance = self.random_covariance(2)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_covariance, angles):
            GaussianStatePrep(0.5 * (raw_covariance - raw_covariance.mT), wires=[1, 3])
            self.matchgates(angles)
            return qml.expval(qml.X(1) @ qml.X(2)), qml.probs(wires=[2, 3])

        self.assert_gradcheck(circuit, (covariance, self.random_angles()))

    def test_gaussian_state_from_polarizations(self):
        polarizations = torch.tensor([0.4, -0.3, 0.7, 0.1], dtype=torch.float64, requires_grad=True)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_polarizations, angles):
            GaussianStatePrep.from_polarizations(raw_polarizations, wires=range(N_WIRES))
            self.matchgates(angles)
            return qml.expval(qml.Z(0) @ qml.Z(1)), qml.vn_entropy(wires=KEPT_WIRES)

        self.assert_gradcheck(circuit, (polarizations, self.random_angles()))

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_two_circuit_chain_through_the_measurement(self, batch_size):
        r"""
        Gate parameters of a first circuit, its reduced covariance matrix, a
        :class:`GaussianStatePrep` on a smaller device, and the outputs of a second circuit.
        """

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def first_circuit(angles):
            qml.BasisState(BASIS_STATE, wires=range(N_WIRES))
            self.matchgates(angles)
            return mc.covariance_matrix(wires=[0, 1, 2])

        @qml.qnode(qml.device("nif.qubit", wires=3), interface="torch")
        def second_circuit(reduced_covariance, follow_up_angles):
            GaussianStatePrep(reduced_covariance, wires=range(3))
            CompRxRx(follow_up_angles[..., 0:2], wires=[0, 1])
            CompRyRy(follow_up_angles[..., 2:4], wires=[1, 2])
            return (
                qml.expval(qml.X(0) @ qml.Z(1) @ qml.X(2)),
                qml.probs(wires=[0, 1]),
                qml.vn_entropy(wires=[0, 1]),
                qml.purity(wires=[1, 2]),
            )

        def chain(angles, follow_up_angles):
            return second_circuit(first_circuit(angles), follow_up_angles)

        self.assert_gradcheck(chain, (self.random_angles(batch_size), self.random_angles(batch_size)))

    def test_two_circuit_chain_through_reduced_state(self):
        r"""
        Same chain with the device methods instead of the measurement: ``apply``, ``reduced_state``,
        and the returned state preparation queued in a second circuit.
        """

        def chain(angles, follow_up_angles):
            device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
            with qml.queuing.AnnotatedQueue() as queue:
                qml.BasisState(BASIS_STATE, wires=range(N_WIRES))
                self.matchgates(angles)
            device.apply(queue.queue)
            reduced_state = device.partial_trace([3])

            @qml.qnode(qml.device("nif.qubit", wires=3), interface="torch")
            def second_circuit():
                qml.apply(reduced_state)
                CompRxRx(follow_up_angles[0:2], wires=[0, 1])
                CompRyRy(follow_up_angles[2:4], wires=[1, 2])
                return qml.expval(qml.Y(0) @ qml.Y(1)), qml.probs(wires=[2])

            return second_circuit()

        self.assert_gradcheck(chain, (self.random_angles(), self.random_angles()))

    def test_entropy_gradient_is_finite_for_a_pure_reduced_state(self):
        r"""
        Without entangling gates across the cut, the reduced state is pure and its polarizations sit at
        the boundary :math:`|\lambda| = 1`, where the entropy terms need a safe logarithm.
        """
        angles = self.random_angles()

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(raw_angles):
            qml.BasisState(BASIS_STATE, wires=range(N_WIRES))
            CompRxRx(raw_angles[0:2], wires=[0, 1])
            CompRyRy(raw_angles[2:4], wires=[2, 3])
            return qml.vn_entropy(wires=[0, 1]), qml.purity(wires=[0, 1])

        entropy, purity = circuit(angles)
        (gradient,) = torch.autograd.grad(entropy + purity, angles)
        assert torch.all(torch.isfinite(gradient))
