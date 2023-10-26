from typing import NamedTuple, Union, Optional
from dataclasses import dataclass
import numpy as np

from . import utils
from . import matchgate_parameter_sets as mps


class Matchgate:
    r"""
    A matchgate is a matrix of the form

    .. math::
        \begin{pmatrix}
            a & 0 & 0 & b \\
            0 & w & x & 0 \\
            0 & y & z & 0 \\
            c & 0 & 0 & d
        \end{pmatrix}

    where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The matrix M can be decomposed as

    .. math::
        A = \begin{pmatrix}
            a & b \\
            c & d
        \end{pmatrix}

    and

    .. math::
        W = \begin{pmatrix}
            w & x \\
            y & z
        \end{pmatrix}

    The matchgate is a unitary matrix if and only if the following conditions are satisfied:

    .. math::
        M^\dagger M = \mathbb{I} \quad \text{and} \quad MM^\dagger = \mathbb{I}

    where :math:`\mathbb{I}` is the identity matrix and :math:`M^\dagger` is the conjugate transpose of :math:`M`,
    and the following condition is satisfied:

    .. math::
        \det(A) = \det(W)

    The final form of the matchgate is

    .. math::
        \begin{pmatrix}
            r_0 e^{i\theta_0} & 0 & 0 & (\sqrt{1 - r_0^2}) e^{i(\theta_2 + \theta_4 - (\theta_1+\pi))} \\
            0 & r_1 e^{i\theta_2} & (\sqrt{1 - r_1^2}) e^{i(\theta_2 + \theta_4 - (\theta_3+\pi))} & 0 \\
            0 & (\sqrt{1 - r_1^2}) e^{i\theta_3} & r_1 e^{i\theta_4} & 0 \\
            (\sqrt{1 - r_0^2}) e^{i\theta_1} & 0 & 0 & r_0 e^{i(\theta_2 + \theta_4 - \theta_0)}
        \end{pmatrix}

    with the parameters

    .. math::
        r_0, r_1 \quad \text{and} \quad  \theta_0, \theta_1, \theta_2, \theta_3, \theta_4

    with

    .. math::
         \theta_i \in [0, 2\pi) \forall i in \{0, 1, 2, 3, 4\}

    Note that by setting the parameter :math:`\theta_4` to :math:`-\theta_2`, the matchgate can be written as

    .. math::
        \begin{pmatrix}
            r_0 e^{i\theta_0} & 0 & 0 & (\sqrt{1 - r_0^2}) e^{-i(\theta_1+\pi)} \\
            0 & r_1 e^{i\theta_2} & (\sqrt{1 - r_1^2}) e^{-i(\theta_3+\pi)} & 0 \\
            0 & (\sqrt{1 - r_1^2}) e^{i\theta_3} & r_1 e^{-i\theta_2} & 0 \\
            (\sqrt{1 - r_0^2}) e^{i\theta_1} & 0 & 0 & r_0 e^{-i\theta_0}
        \end{pmatrix}

    which set

    .. math::
        \det(A) = \det(W) = 1



    """
    ZEROS_INDEXES = [
        (0, 1), (0, 2),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 1), (3, 2),
    ]
    ELEMENTS_INDEXES = [
        (0, 0), (0, 3),  # a, b
        (3, 0), (3, 3),  # c, d
        (1, 1), (1, 2),  # w, x
        (2, 1), (2, 2),  # y, z
    ]

    @staticmethod
    def random() -> 'Matchgate':
        """
        Construct a random Matchgate.

        :return: A random Matchgate.
        :rtype Matchgate:
        """
        return Matchgate(
            params=mps.MatchgatePolarParams(
                r0=np.random.uniform(),
                r1=np.random.uniform(),
                theta0=np.random.uniform(0, 2 * np.pi),
                theta1=np.random.uniform(0, 2 * np.pi),
                theta2=np.random.uniform(0, 2 * np.pi),
                theta3=np.random.uniform(0, 2 * np.pi),
                theta4=np.random.uniform(0, 2 * np.pi),
            )
        )

    @staticmethod
    def is_matchgate(matrix: np.ndarray) -> bool:
        zeros_indexes_as_array = np.asarray(Matchgate.ZEROS_INDEXES)
        check_zeros = np.allclose(matrix[zeros_indexes_as_array[:, 0], zeros_indexes_as_array[:, 1]], 0.0)
        if not check_zeros:
            return False
        elements_indexes_as_array = np.asarray(Matchgate.ELEMENTS_INDEXES)
        full_params_arr = matrix[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
        full_params = mps.MatchgateStandardParams.parse_from_full_params(full_params_arr)
        params = mps.MatchgatePolarParams.parse_from_standard_params(full_params)
        try:
            m = Matchgate(params=params)
            m.check_asserts()
            return True
        except:
            return False

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Matchgate':
        r"""
        Construct a Matchgate from a matrix if it is possible. The matrix must be a 4x4 matrix.

        :param matrix: The matrix to construct the Matchgate from.
        :type matrix: np.ndarray
        :return: The Matchgate constructed from the matrix.
        :rtype: Matchgate

        :raises ValueError: If the matrix is not a 4x4 matrix.
        :raises ValueError: If the matrix is not a matchgate.
        """
        if matrix.shape != (4, 4):
            raise ValueError("The matrix must be a 4x4 matrix.")
        if Matchgate.is_matchgate(matrix):
            elements_indexes_as_array = np.asarray(Matchgate.ELEMENTS_INDEXES)
            full_params_arr = matrix[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
            full_params = mps.MatchgateStandardParams.parse_from_full_params(full_params_arr)
            params = mps.MatchgatePolarParams.parse_from_standard_params(full_params)
            return Matchgate(params=params)
        raise ValueError("The matrix is not a matchgate.")

    @staticmethod
    def from_sub_matrices(A: np.ndarray, W: np.ndarray) -> Union['Matchgate', np.ndarray]:
        r"""
        Construct a Matchgate from the sub-matrices :math:`A` and :math:`W` defined as

        .. math::
            A = \begin{pmatrix}
                a & b \\
                c & d
            \end{pmatrix}

        and

        .. math::
            W = \begin{pmatrix}
                w & x \\
                y & z
            \end{pmatrix}

        where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The matchgate is constructed as

        .. math::
            \begin{pmatrix}
                a & 0 & 0 & b \\
                0 & w & x & 0 \\
                0 & y & z & 0 \\
                c & 0 & 0 & d
            \end{pmatrix}

        """
        if A.shape != (2, 2):
            raise ValueError("The A matrix must be a 2x2 matrix.")

        if W.shape != (2, 2):
            raise ValueError("The W matrix must be a 2x2 matrix.")

        matrix = np.zeros((4, 4), dtype=np.complex128)
        matrix[0, 0] = A[0, 0]
        matrix[0, 3] = A[0, 1]
        matrix[1, 1] = W[0, 0]
        matrix[1, 2] = W[0, 1]
        matrix[2, 1] = W[1, 0]
        matrix[2, 2] = W[1, 1]
        matrix[3, 0] = A[1, 0]
        matrix[3, 3] = A[1, 1]

        return Matchgate.from_matrix(matrix)

    @staticmethod
    def from_hamiltonian_coeffs(coeffs_vector) -> 'Matchgate':
        r"""
        Construct a Matchgate from the Hamiltonian coefficients vector. The Hamiltonian coefficients is a vector that
        represent the upper triangular part of a skew-symmetric 4x4 matrix defined as

        .. math::
            \begin{pmatrix}
                0 & h_{0} & h_{1} & h_{2} \\
                -h_{0} & 0 & h_{3} & h_{4} \\
                -h_{1} & -h_{3} & 0 & h_{5} \\
                -h_{2} & -h_{4} & -h_{5} & 0
            \end{pmatrix}

        where :math:`h_{i}` is the :math:`i`-th element of the Hamiltonian coefficients vector. The matchgate is
        constructed as

        .. math::
            M = \exp{-i\sum_{\mu\nu}^{2n} h_{\mu\nu} c_\mu c_\nu}

        where :math:`c_\mu` and :math:`c_\nu` are the Majorana operators.

        :param coeffs_vector:
        :return:
        """
        from scipy.linalg import expm
        coeffs_matrix = utils.skew_antisymmetric_vector_to_matrix(coeffs_vector)
        hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs_matrix)
        pred_matchgate = expm(-1j * hamiltonian)
        return Matchgate.from_matrix(pred_matchgate)

    @staticmethod
    def to_sympy():
        import sympy as sp
        params = mps.MatchgatePolarParams.to_sympy()
        a, b, c, d, w, x, y, z = mps.MatchgateStandardParams.parse_from_params(params, pkg='sympy')
        return sp.Matrix([
            [a, 0, 0, b],
            [0, w, x, 0],
            [0, y, z, 0],
            [c, 0, 0, d]
        ])

    def __init__(
            self,
            params: Union[mps.MatchgateParams, np.ndarray, list, tuple],
            *,
            backend='numpy',
    ):
        r"""
        Construct a Matchgate from the parameters. The parameters can be a MatchgateParams object, a list, a tuple or
        a numpy array. If the parameters are a list, tuple or numpy array, the parameters will be interpreted as
        MatchgateStandardParams if the length is 8, MatchgatePolarParams if the length is 6 or 7.

        If the parameters are a MatchgateParams object, the parameters will be interpreted as the type of the object.

        :param params: The parameters of the Matchgate.
        :type params: Union[mps.MatchgateParams, np.ndarray, list, tuple]
        :param backend: The backend to use for the computations. Can be 'numpy', 'sympy' or 'torch'.
        :type backend: str
        """
        self._backend = backend

        self._polar_params = None
        self._standard_params = None
        self._hamiltonian_params = None
        self._composed_hamiltonian_params = None

        self._initialize_params_(params)

        self.thetas = self.thetas
        self._standard_params = self._make_full_params()
        self._data = None
        self._make_data_()
        self.check_asserts()
        self._hamiltonian_coeffs = None
        self._hamiltonian_coeffs_found_order = None

    @property
    def polar_params(self) -> mps.MatchgatePolarParams:
        if self._polar_params is None:
            self._make_polar_params_()
        return self._polar_params

    @property
    def standard_params(self) -> mps.MatchgateStandardParams:
        if self._standard_params is None:
            self._make_standard_params_()
        return self._standard_params

    @property
    def hamiltonian_params(self) -> mps.MatchgateHamiltonianParams:
        if self._hamiltonian_params is None:
            self._make_hamiltonian_params_()
        return self._hamiltonian_params

    @property
    def composed_hamiltonian_params(self) -> mps.MatchgateComposedHamiltonianParams:
        if self._composed_hamiltonian_params is None:
            self._make_composed_hamiltonian_params_()
        return self._composed_hamiltonian_params

    @property
    def backend(self):
        return self._backend

    @property
    def data(self):
        return self._data

    @property
    def thetas(self):
        return np.asarray([
            self.polar_params.theta0,
            self.polar_params.theta1,
            self.polar_params.theta2,
            self.polar_params.theta3,
            self.polar_params.theta4,
        ])

    @thetas.setter
    def thetas(self, thetas):
        wrapped_thetas = np.mod(thetas, 2 * np.pi)
        self._polar_params = mps.MatchgatePolarParams(
            r0=self.polar_params.r0,
            r1=self.polar_params.r1,
            theta0=wrapped_thetas[0],
            theta1=wrapped_thetas[1],
            theta2=wrapped_thetas[2],
            theta3=wrapped_thetas[3],
            theta4=wrapped_thetas[4],
        )
        self._standard_params = self._make_full_params()

    @property
    def r(self):
        return np.asarray([
            self.polar_params.r0,
            self.polar_params.r1,
        ])

    @r.setter
    def r(self, r):
        self._polar_params = mps.MatchgatePolarParams(
            r0=r[0],
            r1=r[1],
            theta0=self.polar_params.theta0,
            theta1=self.polar_params.theta1,
            theta2=self.polar_params.theta2,
            theta3=self.polar_params.theta3,
            theta4=self.polar_params.theta4,
        )
        self._standard_params = self._make_full_params()

    @property
    def A(self):
        return np.asarray([
            [self.standard_params.a, self.standard_params.b],
            [self.standard_params.c, self.standard_params.d],
        ])

    @property
    def W(self):
        return np.asarray([
            [self.standard_params.w, self.standard_params.x],
            [self.standard_params.y, self.standard_params.z],
        ])

    @property
    def hamiltonian_coeffs(self):
        r"""
        The Hamiltonian coefficients is a vector that represent the upper triangular part of a skew-symmetric
        4x4 matrix defined as

        .. math::
            \begin{pmatrix}
                0 & h_{0} & h_{1} & h_{2} \\
                -h_{0} & 0 & h_{3} & h_{4} \\
                -h_{1} & -h_{3} & 0 & h_{5} \\
                -h_{2} & -h_{4} & -h_{5} & 0
            \end{pmatrix}

        where :math:`h_{i}` is the :math:`i`-th element of the Hamiltonian coefficients vector.

        :Note: If this quantity is not computed yet, it will be computed using the :meth:`find_hamiltonian_coefficients`
        method with default inputs values. It may take a while to compute.

        :return: The Hamiltonian coefficients vector of length 6.
        """
        if self._hamiltonian_coeffs is None:
            self._hamiltonian_coeffs = self.find_hamiltonian_coefficients()
        return self._hamiltonian_coeffs

    @property
    def hamiltonian_coeffs_found_order(self):
        return self._hamiltonian_coeffs_found_order

    @property
    def hamiltonian_coeffs_matrix(self):
        r"""

        Since the Hamiltonian coefficients is a vector that represent the upper triangular part of a skew-symmetric
        4x4 matrix, we can reconstruct the matrix using the following formula:

        .. math::
            \begin{pmatrix}
                0 & h_{0} & h_{1} & h_{2} \\
                -h_{0} & 0 & h_{3} & h_{4} \\
                -h_{1} & -h_{3} & 0 & h_{5} \\
                -h_{2} & -h_{4} & -h_{5} & 0
            \end{pmatrix}

        where :math:`h_{i}` is the :math:`i`-th element of the Hamiltonian coefficients vector.


        :return: The Hamiltonian coefficients matrix.
        """
        coeffs = self.hamiltonian_coeffs
        return np.asarray([
            [0, coeffs[0], coeffs[1], coeffs[2]],
            [-coeffs[0], 0, coeffs[3], coeffs[4]],
            [-coeffs[1], -coeffs[3], 0, coeffs[5]],
            [-coeffs[2], -coeffs[4], -coeffs[5], 0],
        ])

    def _make_full_params(self):
        return mps.MatchgateStandardParams.parse_from_params(self.polar_params)

    def _initialize_params_(
            self,
            given_params: Union[mps.MatchgateParams, np.ndarray, list, tuple]
    ) -> mps.MatchgateParams:
        r"""

        Initialize the parameters of the matchgate. The parameters can be a MatchgateParams object, a list, a tuple or
        a numpy array. If the parameters are a list, tuple or numpy array, the parameters will be interpreted as
        MatchgateStandardParams if the length is 8, MatchgatePolarParams if the length is 6 or 7.

        If the parameters are a MatchgateParams object, the parameters will be interpreted as the type of the object.

        Finally, the attribute :attr:`_polar_params`, :attr:`_standard_params`, :attr:`_hamiltonian_params` or
        :attr:`_composed_hamiltonian_params` will be set to the given parameters converted to the corresponding type.

        :param given_params: The parameters of the matchgate.
        :type given_params: Union[mps.MatchgateParams, np.ndarray, list, tuple]
        :return: The parameters of the matchgate.
        :rtype: mps.MatchgateParams
        """
        params = None
        if isinstance(given_params, mps.MatchgateParams):
            params = given_params
        elif isinstance(given_params, np.ndarray):
            if given_params.size == 6 or given_params.size == 7:
                params = mps.MatchgatePolarParams.from_numpy(given_params.flatten())
            elif given_params.size == 8:
                params = mps.MatchgateStandardParams.from_numpy(given_params.flatten())
            else:
                raise ValueError("The given params must be a 6, 7 or 8 elements array.")
        elif isinstance(given_params, (list, tuple)):
            if len(given_params) == 6 or len(given_params) == 7:
                params = mps.MatchgatePolarParams.parse_from_params(given_params)
            elif len(given_params) == 8:
                params = mps.MatchgateStandardParams.parse_from_params(given_params)
            else:
                raise ValueError("The given params must be a 6, 7 or 8 elements array.")
        else:
            raise ValueError("The given params must be a 6, 7 or 8 elements array.")

        if isinstance(params, mps.MatchgatePolarParams):
            self._polar_params = params
        elif isinstance(params, mps.MatchgateStandardParams):
            self._standard_params = params
        elif isinstance(params, mps.MatchgateHamiltonianParams):
            self._hamiltonian_params = params
        elif isinstance(params, mps.MatchgateComposedHamiltonianParams):
            self._composed_hamiltonian_params = params
        else:
            raise ValueError("The given params is not a valid MatchgateParams object.")
        return params

    def _make_polar_params_(self):
        raise NotImplementedError

    def _make_standard_params_(self):
        raise NotImplementedError

    def _make_hamiltonian_params_(self):
        raise NotImplementedError

    def _make_composed_hamiltonian_params_(self):
        raise NotImplementedError

    def _make_data_(self) -> np.ndarray:
        self._data = np.asarray([
            [self.standard_params.a, 0, 0, self.standard_params.b],
            [0, self.standard_params.w, self.standard_params.x, 0],
            [0, self.standard_params.y, self.standard_params.z, 0],
            [self.standard_params.c, 0, 0, self.standard_params.d],
        ])
        return self._data

    def compute_m_m_dagger(self):
        return np.matmul(self.data, np.conjugate(self.data.T))

    def compute_m_dagger_m(self):
        return np.matmul(np.conjugate(self.data.T), self.data)

    def get_sub_a_det(self) -> float:
        return np.linalg.det(self.A)

    def get_sub_w_det(self) -> float:
        return np.linalg.det(self.W)

    def check_m_m_dagger_constraint(self) -> bool:
        return np.allclose(self.compute_m_m_dagger(), np.eye(4))

    def check_m_dagger_m_constraint(self) -> bool:
        return np.allclose(self.compute_m_dagger_m(), np.eye(4))

    def check_det_constraint(self) -> bool:
        return np.isclose(self.get_sub_a_det(), self.get_sub_w_det())

    def check_asserts(self):
        assert self.check_m_m_dagger_constraint()
        assert self.check_m_dagger_m_constraint()
        assert self.check_det_constraint()

    def __repr__(self):
        return f"Matchgate(params={self._polar_params})"

    def __str__(self):
        return f"Matchgate(params={self._polar_params})"

    def __eq__(self, other):
        return np.allclose(self.polar_params.to_numpy(), other.polar_params.to_numpy())

    def __hash__(self):
        return hash(self.polar_params)

    def __copy__(self):
        return Matchgate(params=self.polar_params)

    def __getitem__(self, item):
        return self.data[item]

    def find_hamiltonian_coefficients(self, order: int = 1, iterations: int = 100) -> np.ndarray:
        r"""

        Find the 2n x 2n matrix :math:`h` of elements :math:`h_{\mu\nu}` such that

        .. math::
            M = \exp{-i\sum_{\mu\neq\nu = 1}^{2n} h_{\mu\nu} c_\mu c_\nu}

        where :math:`c_\mu` is the :math:`\mu`-th Majorana operator, :math:`M` is the matchgate and
        :math:`n` is the number of qubits which is equal to the number of rows and columns of the matchgate.

        :param order: The order of the taylor expansion of the exponential.
        :type order: int
        :return: The matrix of coefficients :math:`h`.
        :rtype: np.ndarray
        """
        from scipy.optimize import minimize
        from scipy.linalg import expm

        n_state = self.data.shape[0]
        n = self.data.shape[0] // 2

        def cost_function(coeffs_vector):
            coeffs_matrix = utils.skew_antisymmetric_vector_to_matrix(coeffs_vector)
            hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs_matrix)
            pred_matchgate = expm(-1j * hamiltonian)
            return np.linalg.norm(pred_matchgate - self.data)

        result = minimize(
            cost_function,
            np.random.uniform(-1.0, 1.0, size=(n_state * (n_state - 1) // 2,)),
        )
        self._hamiltonian_coeffs = result.x
        self._hamiltonian_coeffs_found_order = order
        return self._hamiltonian_coeffs

