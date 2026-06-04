import itertools
from typing import cast

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.expval_strategies.m_pfaffian import (
    MPfaffianExpvalStrategy,
    displacement_vector,
    extended_covariance_matrix,
)
from matchcake.operations.comp_rotations import CompRyRy, CompRzRz
from matchcake.operations.fermionic_swap import fSWAP
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils._pfaffian import signed_pfaffian
from matchcake.utils.jordan_wigner import JordanWigner

ATOL = 1e-9


def random_product_state(n: int, seed: int = 0) -> np.ndarray:
    """Return (n, 2) complex array of unit-norm per-qubit amplitudes."""
    rng = np.random.RandomState(seed)
    state = np.zeros((n, 2), dtype=complex)
    for k in range(n):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        state[k, 0] = np.cos(theta / 2)
        state[k, 1] = np.exp(1j * phi) * np.sin(theta / 2)
    return state


def brute_force_expval(psi_flat: np.ndarray, obs: qml.operation.Operator, n: int) -> float:
    """<psi|obs|psi> via full matrix multiplication."""
    P = obs.matrix(wire_order=list(range(n)))
    return float(np.real(psi_flat.conj() @ P @ psi_flat))


def random_circuit_gates(n: int, n_gates: int, rng: np.random.RandomState) -> list:
    """Return a list of (gate_class, params_or_None, wires) for a random matchgate circuit.

    Gates are drawn from {CompRzRz, CompRyRy, fSWAP} on random adjacent wire pairs.
    """
    gate_classes = [CompRzRz, CompRyRy, fSWAP]
    gates: list = []
    for _ in range(n_gates):
        w = int(rng.randint(0, n - 1))
        gate_cls = gate_classes[int(rng.randint(0, len(gate_classes)))]
        if gate_cls is fSWAP:
            gates.append((gate_cls, None, [w, w + 1]))
        else:
            angles = rng.uniform(0, 2 * np.pi, 2)
            gates.append((gate_cls, angles, [w, w + 1]))
    return gates


def random_pauli_obs(n: int, rng: np.random.RandomState) -> qml.operation.Operator:
    """Return a random tensor-product Pauli observable on n qubits (no identity factors)."""
    pauli_map = {0: qml.X, 1: qml.Y, 2: qml.Z}
    terms = [pauli_map[int(rng.randint(0, 3))](k) for k in range(n)]
    obs = terms[0]
    for t in terms[1:]:
        obs = obs @ t
    return obs


def build_tilde_lambda(prod_state: ProductState, wires: list) -> torch.Tensor:
    """Build the extended covariance matrix from a ProductState with no circuit."""
    cov_mat = cast(torch.Tensor, prod_state.covariance_matrix)
    Lambda = cov_mat.detach().numpy()
    d = displacement_vector(torch.tensor(prod_state.data[0]), wires)
    return cast(torch.Tensor, extended_covariance_matrix(torch.tensor(Lambda), d))


class TestJordanWigner:
    """Tests for JordanWigner.pauli_to_majorana correctness against known table and brute-force matrices."""

    def setup_method(self):
        self.jw = JordanWigner(3)

    @pytest.mark.parametrize(
        "pauli,wires,exp_indices,exp_phase",
        [
            ("X", [0], [0], 1 + 0j),
            ("Y", [0], [1], 1 + 0j),
            ("Z", [0], [0, 1], -1j),
            ("I", [0], [], 1 + 0j),
            ("XX", [0, 1], [1, 2], -1j),
            ("YX", [0, 1], [0, 2], 1j),
            ("XY", [0, 1], [1, 3], -1j),
            ("YY", [0, 1], [0, 3], 1j),
            ("ZZ", [0, 1], [0, 1, 2, 3], -1 + 0j),
            ("IZ", [0, 1], [2, 3], -1j),
            ("ZI", [0, 1], [0, 1], -1j),
        ],
    )
    def test_known_cases(self, pauli, wires, exp_indices, exp_phase):
        mu, ph = self.jw.pauli_to_majorana(pauli, wires)
        np.testing.assert_array_equal(mu, exp_indices)
        assert abs(ph - exp_phase) < ATOL, f"phase {ph} != {exp_phase}"

    def test_brute_force_n1(self):
        """For all 1-qubit Paulis, verify via matrix: P == phase * product(c_mu)."""
        from matchcake.utils.majorana import get_majorana

        jw = JordanWigner(1)
        for char in "IXYZ":
            mu, ph = jw.pauli_to_majorana(char, [0])
            if len(mu) == 0:
                mat = np.eye(2)
            else:
                mat = get_majorana(mu[0], 1)
                for idx in mu[1:]:
                    mat = mat @ get_majorana(idx, 1)
            target = qml.pauli.string_to_pauli_word(char if char != "I" else "I").matrix()
            np.testing.assert_allclose(ph * mat, target, atol=ATOL)

    def test_brute_force_n2(self):
        """For all 2-qubit Paulis, verify via matrix."""
        from matchcake.utils.majorana import get_majorana

        jw = JordanWigner(2)
        for chars in itertools.product("IXYZ", repeat=2):
            pauli_str = "".join(chars)
            mu, ph = jw.pauli_to_majorana(pauli_str, [0, 1])
            if len(mu) == 0:
                mat = np.eye(4)
            else:
                mat = get_majorana(mu[0], 2)
                for idx in mu[1:]:
                    mat = mat @ get_majorana(idx, 2)
            pw = qml.pauli.string_to_pauli_word(pauli_str)
            target = pw.matrix(wire_order=[0, 1])
            np.testing.assert_allclose(ph * mat, target, atol=ATOL, err_msg=f"Failed for {pauli_str}")


class TestDisplacementVector:
    """Tests for displacement_vector correctness on known single- and multi-qubit states."""

    def test_zero_state(self):
        """Computational |0> has zero displacement."""
        psi = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
        d = displacement_vector(psi, [0])
        np.testing.assert_allclose(d.numpy(), [0.0, 0.0], atol=ATOL)

    def test_plus_state_single(self):
        """|+> on one qubit: <c_0>=1, <c_1>=0."""
        psi = torch.tensor([[1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=torch.complex128)
        d = displacement_vector(psi, [0])
        np.testing.assert_allclose(d.numpy(), [1.0, 0.0], atol=ATOL)

    def test_plus_plus(self):
        """|++>: only <c_0>=1, rest zero because <Z_0>=0 kills the JW string."""
        psi = torch.tensor([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=torch.complex128)
        d = displacement_vector(psi, [0, 1])
        np.testing.assert_allclose(d.numpy(), [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_one_state(self):
        """Computational |1> has displacement [0, 0] (X and Y expectations are 0)."""
        psi = torch.tensor([[0.0, 1.0]], dtype=torch.complex128)
        d = displacement_vector(psi, [0])
        np.testing.assert_allclose(d.numpy(), [0.0, 0.0], atol=ATOL)


class TestExtendedCovarianceMatrix:
    """Tests for the extended covariance matrix layout and antisymmetry."""

    def test_vacuum_gives_padded_lambda(self):
        """|0>^n: displacement zero, tilde_Lambda = Lambda padded with zeros."""
        n = 2
        psi = np.zeros((n, 2), dtype=complex)
        psi[:, 0] = 1.0
        op = ProductState(psi, wires=list(range(n)))
        Lambda = op.covariance_matrix.detach().numpy()
        d = np.zeros(2 * n)
        tilde = extended_covariance_matrix(torch.tensor(Lambda), torch.tensor(d)).numpy()
        # Top-left block = Lambda
        np.testing.assert_allclose(tilde[: 2 * n, : 2 * n], Lambda, atol=ATOL)
        # Last row and column are zero
        np.testing.assert_allclose(tilde[2 * n, :], 0.0, atol=ATOL)
        np.testing.assert_allclose(tilde[:, 2 * n], 0.0, atol=ATOL)

    def test_plus_single_qubit(self):
        """tilde_Lambda for |+> matches the worked example: [[0,0,1],[0,0,0],[-1,0,0]]."""
        psi = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=complex)
        op = ProductState(psi, wires=[0])
        Lambda = op.covariance_matrix.detach().numpy()
        d = displacement_vector(torch.tensor(op.data[0]), [0]).numpy()
        tilde = extended_covariance_matrix(torch.tensor(Lambda), torch.tensor(d)).numpy()
        expected = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
        np.testing.assert_allclose(tilde, expected, atol=ATOL)

    def test_antisymmetric(self):
        """tilde_Lambda must be antisymmetric for any product state."""
        n = 3
        psi = random_product_state(n, seed=7)
        op = ProductState(psi, wires=list(range(n)))
        Lambda = op.covariance_matrix.detach().numpy()
        d = displacement_vector(torch.tensor(op.data[0]), list(range(n))).numpy()
        tilde = extended_covariance_matrix(torch.tensor(Lambda), torch.tensor(d)).numpy()
        np.testing.assert_allclose(tilde + tilde.T, 0.0, atol=ATOL)


class TestMPfaffianPrecisionControl:
    """The m-Pfaffian building blocks must preserve input precision, not force float64."""

    @pytest.mark.parametrize(
        "amp_dtype, expected_real_dtype",
        [(torch.complex64, torch.float32), (torch.complex128, torch.float64)],
    )
    def test_displacement_vector_precision(self, amp_dtype, expected_real_dtype):
        psi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=amp_dtype)
        d = displacement_vector(psi, [0, 1])
        assert d.dtype == expected_real_dtype

    @pytest.mark.parametrize("r_dtype", [torch.float32, torch.float64])
    def test_extended_covariance_matrix_precision(self, r_dtype):
        n = 2
        cov = torch.zeros(2 * n, 2 * n, dtype=r_dtype)
        d = torch.zeros(2 * n, dtype=r_dtype)
        tilde = extended_covariance_matrix(cov, d)
        assert tilde.dtype == r_dtype

    @pytest.mark.parametrize(
        "r_dtype, c_dtype",
        [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
    )
    def test_end_to_end_device_precision(self, r_dtype, c_dtype):
        """Device r_dtype/c_dtype must control the whole expval pipeline dtype."""
        inv = 1 / np.sqrt(2)
        dev = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype, c_dtype=c_dtype)
        amps = torch.tensor([[1.0, 0.0], [inv, inv]], dtype=c_dtype)
        params = torch.tensor([0.3, 0.4], dtype=r_dtype)

        @qml.qnode(dev)
        def circ(p):
            ProductState(amps, wires=[0, 1])
            CompRyRy(p, wires=[0, 1])
            return qml.expval(qml.Z(0) @ qml.X(1))

        res = circ(params)
        assert dev.extended_covariance_matrix.dtype == r_dtype
        assert res.dtype == r_dtype

    def test_precision_values_agree_across_dtypes(self):
        """float32 and float64 pipelines must agree to single precision."""
        inv = 1 / np.sqrt(2)
        vals = {}
        for r_dtype, c_dtype in [(torch.float32, torch.complex64), (torch.float64, torch.complex128)]:
            dev = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype, c_dtype=c_dtype)
            amps = torch.tensor([[1.0, 0.0], [inv, inv]], dtype=c_dtype)
            params = torch.tensor([0.3, 0.4], dtype=r_dtype)

            @qml.qnode(dev)
            def circ(p):
                ProductState(amps, wires=[0, 1])
                CompRyRy(p, wires=[0, 1])
                return qml.expval(qml.Z(0) @ qml.X(1))

            vals[r_dtype] = float(circ(params))
        np.testing.assert_allclose(vals[torch.float32], vals[torch.float64], atol=1e-5)


class TestSignedPfaffian:
    """Tests for the signed_pfaffian implementation via Parlett-Reid skew-tridiagonalization."""

    def test_2x2(self):
        A = torch.tensor([[0.0, 3.0], [-3.0, 0.0]])
        pf = float(signed_pfaffian(A))
        assert abs(pf - 3.0) < ATOL

    def test_2x2_negative(self):
        A = torch.tensor([[0.0, -2.5], [2.5, 0.0]])
        pf = float(signed_pfaffian(A))
        assert abs(pf - (-2.5)) < ATOL

    def test_4x4_known(self):
        # Pf([[0,a,b,c],[-a,0,d,e],[-b,-d,0,f],[-c,-e,-f,0]]) = a*f - b*e + c*d
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        A = torch.tensor(
            [
                [0, a, b, c],
                [-a, 0, d, e],
                [-b, -d, 0, f],
                [-c, -e, -f, 0],
            ],
            dtype=torch.float64,
        )
        expected = a * f - b * e + c * d  # 1*6 - 2*5 + 3*4 = 6-10+12 = 8
        pf = float(signed_pfaffian(A))
        assert abs(pf - expected) < ATOL

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_pf_squared_equals_det(self, n):
        """Pf^2 = det for random antisymmetric matrices."""
        rng = np.random.RandomState(n * 7)
        M = rng.randn(n, n)
        A = M - M.T
        A_t = torch.tensor(A, dtype=torch.float64)
        pf = float(signed_pfaffian(A_t))
        det = float(torch.linalg.det(A_t))
        assert abs(pf**2 - det) < 1e-8, f"n={n}: pf^2={pf**2:.6f}, det={det:.6f}"


class TestMPfaffianExpvalStrategy:
    """Unit tests for MPfaffianExpvalStrategy in isolation.

    Calls the strategy directly (no QNode) on known product states and compares
    against brute-force statevector matrix multiplication. Covers can_execute
    gating, parity-breaking and parity-preserving observables, basis-state
    regression, and Hamiltonian with mixed parity sectors.
    """

    def setup_method(self):
        self.strat = MPfaffianExpvalStrategy()

    def test_can_execute_product_state(self):
        op = ProductState(np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1.0, 0.0]], dtype=complex), wires=[0, 1])
        assert self.strat.can_execute(op, qml.X(0))
        assert self.strat.can_execute(op, qml.Z(0) @ qml.Z(1))

    def test_can_execute_basis_state(self):
        bs = qml.BasisState([0, 1], wires=[0, 1])
        assert self.strat.can_execute(bs, qml.X(0))

    def test_cannot_execute_hermitian(self):
        op = ProductState(np.array([[1.0, 0.0]], dtype=complex), wires=[0])
        obs = qml.Hermitian(np.eye(2), wires=[0])
        assert not self.strat.can_execute(op, obs)

    def test_can_execute_false_for_unsupported_state_prep(self):
        state_prep = qml.StatePrep(np.array([1.0, 0.0]), wires=[0])
        assert not self.strat.can_execute(state_prep, qml.X(0))

    def test_execute_raises_on_unsupported_state_prep(self):
        state_prep = qml.StatePrep(np.array([1.0, 0.0]), wires=[0])
        tilde_L = torch.zeros(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="Cannot execute"):
            self.strat(state_prep, qml.X(0), extended_covariance_matrix=tilde_L)

    def test_hamiltonian_with_identity_term(self):
        n = 2
        psi = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1.0, 0.0]], dtype=complex)
        wires = list(range(n))
        prod_state = ProductState(psi, wires=wires)
        psi_flat = np.array(prod_state.state_vector()).reshape(-1)
        tilde_L = build_tilde_lambda(prod_state, wires)
        H = qml.Hamiltonian([2.0, 0.5], [qml.Identity(0), qml.X(0)])
        ref = 2.0 + 0.5 * brute_force_expval(psi_flat, qml.X(0), n)
        got = float(self.strat(prod_state, H, extended_covariance_matrix=tilde_L))
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (3, 2), (2, 3)])
    def test_single_qubit_xyz(self, n, seed):
        """X, Y, Z expectations match brute-force for all qubits."""
        psi = random_product_state(n, seed=seed)
        wires = list(range(n))
        prod_state = ProductState(psi, wires=wires)
        psi_flat = np.array(prod_state.state_vector()).reshape(-1)
        tilde_L = build_tilde_lambda(prod_state, wires)

        for k in range(n):
            for obs in [qml.X(k), qml.Y(k), qml.Z(k)]:
                ref = brute_force_expval(psi_flat, obs, n)
                got = float(self.strat(prod_state, obs, extended_covariance_matrix=tilde_L))
                np.testing.assert_allclose(got, ref, atol=ATOL, err_msg=f"n={n} seed={seed} obs={obs}")

    @pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (4, 2)])
    def test_two_qubit_correlators(self, n, seed):
        psi = random_product_state(n, seed=seed)
        wires = list(range(n))
        prod_state = ProductState(psi, wires=wires)
        psi_flat = np.array(prod_state.state_vector()).reshape(-1)
        tilde_L = build_tilde_lambda(prod_state, wires)

        observables = [
            qml.X(0) @ qml.X(1),
            qml.Y(0) @ qml.X(1),
            qml.X(0) @ qml.Y(1),
            qml.Y(0) @ qml.Y(1),
            qml.Z(0) @ qml.Z(1),
        ]
        for obs in observables:
            ref = brute_force_expval(psi_flat, obs, n)
            got = float(self.strat(prod_state, obs, extended_covariance_matrix=tilde_L))
            np.testing.assert_allclose(got, ref, atol=ATOL, err_msg=f"obs={obs}")

    @pytest.mark.parametrize(
        "bits,obs",
        [
            ([0, 0], qml.X(0)),
            ([0, 0], qml.Z(0) @ qml.Z(1)),
            ([1, 0], qml.X(0)),
            ([0, 1], qml.Z(0) @ qml.Z(1)),
            ([0, 0, 1], qml.X(1)),
        ],
    )
    def test_basis_state_regression(self, bits, obs):
        """On computational basis states, results must match brute-force statevector."""
        n = len(bits)
        wires = list(range(n))
        psi = np.zeros((n, 2), dtype=complex)
        for k, b in enumerate(bits):
            psi[k, b] = 1.0
        prod_state = ProductState(psi, wires=wires)
        psi_flat = np.array(prod_state.state_vector()).reshape(-1)
        tilde_L = build_tilde_lambda(prod_state, wires)

        ref = brute_force_expval(psi_flat, obs, n)
        got = float(self.strat(prod_state, obs, extended_covariance_matrix=tilde_L))
        np.testing.assert_allclose(got, ref, atol=ATOL, err_msg=f"bits={bits} obs={obs}")

    @pytest.mark.parametrize("n,seed", [(2, 5), (3, 6)])
    def test_hamiltonian_mixed_sectors(self, n, seed):
        """Hamiltonian with terms from both parity sectors."""
        psi = random_product_state(n, seed=seed)
        wires = list(range(n))
        prod_state = ProductState(psi, wires=wires)
        psi_flat = np.array(prod_state.state_vector()).reshape(-1)
        tilde_L = build_tilde_lambda(prod_state, wires)

        H = 0.5 * qml.X(0) + 0.3 * qml.Z(0) @ qml.Z(1) + 0.2 * qml.Y(0)
        ref = sum(float(c) * brute_force_expval(psi_flat, op, n) for c, op in zip(*H.terms()))
        got = float(self.strat(prod_state, H, extended_covariance_matrix=tilde_L))
        np.testing.assert_allclose(got, ref, atol=ATOL)


class TestMPfaffianEndToEnd:
    """End-to-end test of MPfaffianExpvalStrategy through the full NIF device QNode.

    Runs ProductState + matchgate circuits on NonInteractingFermionicDevice and
    compares every expval against default.qubit statevector simulation. Because
    MPfaffianExpvalStrategy is first in the NIF device dispatch chain, all expval
    measurements here go through it, including both parity-preserving (ZZ, XX)
    and parity-breaking (X, Y, XY) observables on non-trivial product states.
    """

    @staticmethod
    def _nif(n: int) -> NonInteractingFermionicDevice:
        return NonInteractingFermionicDevice(wires=n)

    @staticmethod
    def _qubit(n: int):
        return qml.device("default.qubit", wires=n)

    @pytest.mark.parametrize(
        "n,seed,obs",
        [
            (2, 0, qml.X(0)),
            (2, 1, qml.Y(0)),
            (2, 2, qml.Z(0) @ qml.Z(1)),
            (2, 3, qml.X(0) @ qml.Y(1)),
            (3, 4, qml.X(1)),
            (3, 5, qml.Y(0) @ qml.X(1)),
            (3, 6, qml.Z(0) @ qml.Z(2)),
        ],
    )
    def test_product_state_no_circuit(self, n, seed, obs):
        """ProductState alone, no matchgates: NIF must match default.qubit."""
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))

        def nif_circuit():
            ProductState(psi, wires=wires)
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "n,seed,angles,obs",
        [
            (2, 0, [0.3, 0.7], qml.X(0)),
            (2, 1, [0.3, 0.7], qml.Z(0) @ qml.Z(1)),
            (2, 2, [1.1, 0.4], qml.X(0) @ qml.Y(1)),
            (2, 3, [0.5, 1.5], qml.Y(0)),
            (2, 4, [2.0, 0.1], qml.Y(0) @ qml.Y(1)),
            (2, 5, [np.pi / 4, np.pi / 3], qml.X(0) @ qml.X(1)),
            (3, 6, [0.8, 0.2], qml.X(0)),
            (3, 7, [1.0, 1.5], qml.Z(0) @ qml.Z(1)),
        ],
    )
    def test_single_rzrz(self, n, seed, angles, obs):
        """ProductState + one CompRzRz on [0,1]: NIF must match default.qubit."""
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))
        params = np.array(angles)

        def nif_circuit():
            ProductState(psi, wires=wires)
            CompRzRz(params, wires=[0, 1])
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            CompRzRz(params, wires=[0, 1])
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "n,seed,angles,obs",
        [
            (2, 0, [0.3, 0.7], qml.X(0)),
            (2, 1, [0.3, 0.7], qml.Z(0) @ qml.Z(1)),
            (2, 2, [1.1, 0.4], qml.Y(0) @ qml.X(1)),
            (2, 3, [np.pi / 2, np.pi / 6], qml.X(0) @ qml.Y(1)),
            (2, 4, [0.9, 1.3], qml.X(0)),
            (3, 5, [0.5, 1.2], qml.Y(1)),
            (3, 6, [1.5, 0.6], qml.Z(0) @ qml.Z(1)),
        ],
    )
    def test_single_ryry(self, n, seed, angles, obs):
        """ProductState + one CompRyRy on [0,1]: NIF must match default.qubit."""
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))
        params = np.array(angles)

        def nif_circuit():
            ProductState(psi, wires=wires)
            CompRyRy(params, wires=[0, 1])
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            CompRyRy(params, wires=[0, 1])
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "n,seed,obs",
        [
            (2, 0, qml.Z(0) @ qml.Z(1)),
            (2, 1, qml.X(0) @ qml.X(1)),
            (2, 2, qml.Z(0)),
            (2, 3, qml.Y(0) @ qml.X(1)),
            (2, 4, qml.X(0)),
        ],
    )
    def test_fswap(self, n, seed, obs):
        """ProductState + fSWAP on [0,1]: NIF must match default.qubit."""
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))

        def nif_circuit():
            ProductState(psi, wires=wires)
            fSWAP(wires=[0, 1])
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            fSWAP(wires=[0, 1])
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "seed,a01,a12,obs",
        [
            (0, [0.3, 0.7], [1.1, 0.4], qml.X(0)),
            (1, [0.3, 0.7], [1.1, 0.4], qml.Z(0) @ qml.Z(1)),
            (2, [0.5, 1.2], [0.9, 0.3], qml.X(0) @ qml.Y(1)),
            (3, [np.pi / 4, np.pi / 3], [np.pi / 6, 1.0], qml.Y(1) @ qml.X(2)),
            (4, [0.8, 0.2], [1.5, 0.6], qml.Z(1) @ qml.Z(2)),
            (5, [1.0, 0.5], [0.4, 1.2], qml.X(0) @ qml.X(2)),
        ],
    )
    def test_chain_3qubit(self, seed, a01, a12, obs):
        """ProductState + CompRyRy on [0,1] then CompRzRz on [1,2], 3-qubit chain."""
        n = 3
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))
        p01 = np.array(a01)
        p12 = np.array(a12)

        def nif_circuit():
            ProductState(psi, wires=wires)
            CompRyRy(p01, wires=[0, 1])
            CompRzRz(p12, wires=[1, 2])
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            CompRyRy(p01, wires=[0, 1])
            CompRzRz(p12, wires=[1, 2])
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "n,n_gates,seed",
        [
            (2, 1, 10),
            (2, 3, 11),
            (2, 5, 12),
            (3, 2, 13),
            (3, 4, 14),
            (3, 6, 15),
            (4, 3, 16),
            (4, 6, 17),
            (4, 9, 18),
        ],
    )
    def test_random_circuit(self, n, n_gates, seed):
        """Random product state + random matchgate circuit + random Pauli observable."""
        rng = np.random.RandomState(seed)
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))
        gates = random_circuit_gates(n, n_gates, rng)
        obs = random_pauli_obs(n, rng)

        def nif_circuit():
            ProductState(psi, wires=wires)
            for gate_cls, params, gate_wires in gates:
                gate_cls(wires=gate_wires) if params is None else gate_cls(params, wires=gate_wires)
            return qml.expval(obs)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            for gate_cls, params, gate_wires in gates:
                gate_cls(wires=gate_wires) if params is None else gate_cls(params, wires=gate_wires)
            return qml.expval(obs)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)

    @pytest.mark.parametrize(
        "n,seed,angles",
        [
            (2, 0, [0.3, 0.7]),
            (2, 1, [1.1, 0.4]),
            (2, 2, [np.pi / 4, np.pi / 3]),
            (2, 3, [0.9, 0.6]),
            (3, 4, [0.5, 1.2]),
        ],
    )
    def test_hamiltonian_mixed_parity(self, n, seed, angles):
        """Hamiltonian with parity-preserving (ZZ) and parity-breaking (X, Y, XY) terms."""
        psi = random_product_state(n, seed=seed)
        psi_flat = np.array(ProductState(psi, wires=list(range(n))).state_vector()).reshape(-1)
        wires = list(range(n))
        params = np.array(angles)
        H = 0.5 * qml.X(0) + 0.3 * qml.Z(0) @ qml.Z(1) + 0.2 * qml.Y(0) @ qml.X(1) - 0.4 * qml.Y(1)

        def nif_circuit():
            ProductState(psi, wires=wires)
            CompRyRy(params, wires=[0, 1])
            return qml.expval(H)

        def qubit_circuit():
            qml.StatePrep(psi_flat, wires=wires)
            CompRyRy(params, wires=[0, 1])
            return qml.expval(H)

        got = float(qml.QNode(nif_circuit, self._nif(n))())
        ref = float(qml.QNode(qubit_circuit, self._qubit(n))())
        np.testing.assert_allclose(got, ref, atol=ATOL)
