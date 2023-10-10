from typing import NamedTuple, Union

import numpy as np


class MatchgateParams(NamedTuple):
    r0: float
    r1: float
    theta0: float
    theta1: float
    theta2: float
    theta3: float
    theta4: float

    @staticmethod
    def parse_from_params(params):
        return MatchgateParams(
            r0=params[0],
            r1=params[1],
            theta0=params[2],
            theta1=params[3],
            theta2=params[4],
            theta3=params[5],
            theta4=params[6],
        )

    @staticmethod
    def parse_from_full_params(full_params):
        full_params = MatchgateFullParams.parse_from_full_params(full_params)
        return MatchgateParams(
            r0=np.sqrt(full_params.a * np.conjugate(full_params.a)),
            r1=np.sqrt(full_params.w * np.conjugate(full_params.w)),
            theta0=np.angle(full_params.a),
            theta1=np.angle(full_params.c),
            theta2=np.angle(full_params.w),
            theta3=np.angle(full_params.y),
            theta4=np.angle(full_params.z),
        )

    @staticmethod
    def to_sympy():
        import sympy as sp
        r0, r1 = sp.symbols('r_0 r_1')
        theta0, theta1, theta2, theta3, theta4 = sp.symbols('\\theta_0 \\theta_1 \\theta_2 \\theta_3 \\theta_4')
        return sp.Matrix([r0, r1, theta0, theta1, theta2, theta3, theta4])

    @staticmethod
    def compute_r_tilde(r, pkg='numpy'):
        if pkg == 'numpy':
            _pkg = np
        elif pkg == 'sympy':
            import sympy
            _pkg = sympy
        else:
            raise ValueError("The pkg argument must be either 'numpy' or 'sympy'.")
        return _pkg.sqrt(1 - r ** 2)

    @property
    def r0_tilde(self) -> float:
        """
        Return :math:`\\Tilde{r}_0 = \\sqrt{1 - r_0^2}`

        :return: :math:`\\Tilde{r}_0`
        """
        return self.compute_r_tilde(self.r0, pkg='numpy')

    @property
    def r1_tilde(self) -> float:
        """
        Return :math:`\\Tilde{r}_1 = \\sqrt{1 - r_1^2}`

        :return: :math:`\\Tilde{r}_1`
        """
        return self.compute_r_tilde(self.r1, pkg='numpy')

    def to_numpy(self):
        return np.asarray([
            self.r0,
            self.r1,
            self.theta0,
            self.theta1,
            self.theta2,
            self.theta3,
            self.theta4,
        ])

    def to_string(self):
        return str(self)

    def __repr__(self):
        return (f"MatchgateParams("
                f"r0={self.r0}, "
                f"r1={self.r1}, "
                f"theta0={self.theta0}, "
                f"theta1={self.theta1}, "
                f"theta2={self.theta2}, "
                f"theta3={self.theta3}, "
                f"theta4={self.theta4})")

    def __str__(self):
        return f"[{self.r0}, {self.r1}, {self.theta0}, {self.theta1}, {self.theta2}, {self.theta3}, {self.theta4}]"

    def __eq__(self, other):
        return np.allclose(self.to_numpy(), other.to_numpy())

    def __hash__(self):
        return hash(self.to_string())


class MatchgateFullParams(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def parse_from_full_params(full_params):
        return MatchgateFullParams(
            a=full_params[0],
            b=full_params[1],
            c=full_params[2],
            d=full_params[3],
            w=full_params[4],
            x=full_params[5],
            y=full_params[6],
            z=full_params[7],
        )

    @staticmethod
    def parse_from_params(params, pkg='numpy'):
        if pkg == 'numpy':
            _pkg = np
        elif pkg == 'sympy':
            import sympy
            _pkg = sympy
        else:
            raise ValueError("The pkg argument must be either 'numpy' or 'sympy'.")
        params = MatchgateParams.parse_from_params(params)
        r0_tilde = MatchgateParams.compute_r_tilde(params.r0, pkg=pkg)
        r1_tilde = MatchgateParams.compute_r_tilde(params.r1, pkg=pkg)
        return MatchgateFullParams(
            a=params.r0 * _pkg.exp(1j * params.theta0),
            b=r0_tilde * _pkg.exp(1j * (params.theta2 + params.theta4 - (params.theta1 + _pkg.pi))),
            c=r0_tilde * _pkg.exp(1j * params.theta1),
            d=params.r0 * _pkg.exp(1j * (params.theta2 + params.theta4 - params.theta0)),
            w=params.r1 * _pkg.exp(1j * params.theta2),
            x=r1_tilde * _pkg.exp(1j * (params.theta2 + params.theta4 - (params.theta3 + _pkg.pi))),
            y=r1_tilde * _pkg.exp(1j * params.theta3),
            z=params.r1 * _pkg.exp(1j * params.theta4),
        )

    @staticmethod
    def to_sympy():
        import sympy as sp
        a, b, c, d = sp.symbols('a b c d')
        w, x, y, z = sp.symbols('w x y z')
        return sp.Matrix([a, b, c, d, w, x, y, z])

    def to_numpy(self):
        return np.asarray([
            self.a,
            self.b,
            self.c,
            self.d,
            self.w,
            self.x,
            self.y,
            self.z,
        ])


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

    """
    ZEROS_INDEXES = [
        (0, 1), (0, 2),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 1), (3, 2),
    ]
    ELEMENTS_INDEXES = [
        (0, 0), (0, 3),
        (1, 1), (1, 2),
        (2, 1), (2, 2),
        (3, 0), (3, 3),
    ]

    @staticmethod
    def random() -> 'Matchgate':
        """
        Construct a random Matchgate.

        :return: A random Matchgate.
        :rtype Matchgate:
        """
        return Matchgate(
            params=MatchgateParams(
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
        full_params = MatchgateFullParams.parse_from_full_params(full_params_arr)
        params = MatchgateParams.parse_from_full_params(full_params)
        try:
            m = Matchgate(params=params)
            m.check_asserts()
            return True
        except:
            return False

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Union['Matchgate', np.ndarray]:
        """
        Construct a Matchgate from a matrix.

        :param matrix: The matrix to construct the Matchgate or Matchgates from.
        :type matrix: np.ndarray
        :return: The Matchgate or Matchgates constructed from the matrix.
        :rtype: Matchgate or np.ndarray
        """
        if matrix.shape != (4, 4):
            raise ValueError("The matrix must be a 4x4 matrix.")
        if Matchgate.is_matchgate(matrix):
            elements_indexes_as_array = np.asarray(Matchgate.ELEMENTS_INDEXES)
            full_params_arr = matrix[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
            full_params = MatchgateFullParams.parse_from_full_params(full_params_arr)
            params = MatchgateParams.parse_from_full_params(full_params)
            return Matchgate(params=params)
        # check if the matrix can be decomposed into multiple matchgates
        # if so, return a list of matchgates
        # else, raise an error

    @staticmethod
    def to_sympy():
        import sympy as sp
        params = MatchgateParams.to_sympy()
        a, b, c, d, w, x, y, z = MatchgateFullParams.parse_from_params(params, pkg='sympy')
        return sp.Matrix([
            [a, 0, 0, b],
            [0, w, x, 0],
            [0, y, z, 0],
            [c, 0, 0, d]
        ])

    def __init__(
            self,
            params: Union[MatchgateParams, np.ndarray, list, tuple]
    ):
        self._params = MatchgateParams.parse_from_params(params)
        self.thetas = self.thetas
        self._full_params = self._make_full_params()
        self._data = None
        self._make_data_()
        self.check_asserts()

    @property
    def params(self):
        return self._params

    @property
    def full_params(self):
        return self._full_params

    @property
    def data(self):
        return self._data

    @property
    def thetas(self):
        return np.asarray([
            self.params.theta0,
            self.params.theta1,
            self.params.theta2,
            self.params.theta3,
            self.params.theta4,
        ])

    @thetas.setter
    def thetas(self, thetas):
        wrapped_thetas = np.mod(thetas, 2 * np.pi)
        self._params = MatchgateParams(
            r0=self.params.r0,
            r1=self.params.r1,
            theta0=wrapped_thetas[0],
            theta1=wrapped_thetas[1],
            theta2=wrapped_thetas[2],
            theta3=wrapped_thetas[3],
            theta4=wrapped_thetas[4],
        )
        self._full_params = self._make_full_params()

    @property
    def r(self):
        return np.asarray([
            self.params.r0,
            self.params.r1,
        ])

    @r.setter
    def r(self, r):
        self._params = MatchgateParams(
            r0=r[0],
            r1=r[1],
            theta0=self.params.theta0,
            theta1=self.params.theta1,
            theta2=self.params.theta2,
            theta3=self.params.theta3,
            theta4=self.params.theta4,
        )
        self._full_params = self._make_full_params()

    @property
    def A(self):
        return np.asarray([
            [self.full_params.a, self.full_params.b],
            [self.full_params.c, self.full_params.d],
        ])

    @property
    def W(self):
        return np.asarray([
            [self.full_params.w, self.full_params.x],
            [self.full_params.y, self.full_params.z],
        ])

    def _make_full_params(self):
        return MatchgateFullParams.parse_from_params(self.params)

    def _make_data_(self):
        self._data = np.asarray([
            [self.full_params.a, 0, 0, self.full_params.b],
            [0, self.full_params.w, self.full_params.x, 0],
            [0, self.full_params.y, self.full_params.z, 0],
            [self.full_params.c, 0, 0, self.full_params.d],
        ])

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
        return f"Matchgate(params={self._params})"

    def __str__(self):
        return f"Matchgate(params={self._params})"

    def __eq__(self, other):
        return np.allclose(self.params.to_numpy(), other.params.to_numpy())

    def __hash__(self):
        return hash(self.params)

    def __copy__(self):
        return Matchgate(params=self.params)

    def __getitem__(self, item):
        return self.data[item]


