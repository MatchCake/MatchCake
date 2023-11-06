from typing import Union

import numpy as np

from .matchgate_params import MatchgateParams


class MatchgateStandardParams(MatchgateParams):
    N_PARAMS = 8
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

    def __init__(
            self,
            a: Union[float, complex] = 0.0,
            b: Union[float, complex] = 0.0,
            c: Union[float, complex] = 0.0,
            d: Union[float, complex] = 0.0,
            w: Union[float, complex] = 0.0,
            x: Union[float, complex] = 0.0,
            y: Union[float, complex] = 0.0,
            z: Union[float, complex] = 0.0,
            *,
            backend='numpy',
    ):
        super().__init__(backend=backend)
        self._a = complex(a)
        self._b = complex(b)
        self._c = complex(c)
        self._d = complex(d)
        self._w = complex(w)
        self._x = complex(x)
        self._y = complex(y)
        self._z = complex(z)

    @property
    def a(self) -> Union[float, complex]:
        return self._a

    @property
    def b(self) -> Union[float, complex]:
        return self._b

    @property
    def c(self) -> Union[float, complex]:
        return self._c

    @property
    def d(self) -> Union[float, complex]:
        return self._d

    @property
    def w(self) -> Union[float, complex]:
        return self._w

    @property
    def x(self) -> Union[float, complex]:
        return self._x

    @property
    def y(self) -> Union[float, complex]:
        return self._y

    @property
    def z(self) -> Union[float, complex]:
        return self._z

    @classmethod
    def to_sympy(cls):
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

    def to_matrix(self):
        return self.backend.asarray([
            [self.a, 0, 0, self.b],
            [0, self.w, self.x, 0],
            [0, self.y, self.z, 0],
            [self.c, 0, 0, self.d],
        ])

    def adjoint(self):
        r"""
        Return the adjoint version of the parameters.

        :return: The adjoint parameters.
        """
        return MatchgateStandardParams(
            a=np.conjugate(self.a),
            b=np.conjugate(self.c),
            c=np.conjugate(self.b),
            d=np.conjugate(self.d),
            w=np.conjugate(self.w),
            x=np.conjugate(self.y),
            y=np.conjugate(self.x),
            z=np.conjugate(self.z),
            backend=self.backend,
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"a={self.a}, "
                f"b={self.b}, "
                f"c={self.c}, "
                f"d={self.d}, "
                f"w={self.w}, "
                f"x={self.x}, "
                f"y={self.y}, "
                f"z={self.z})")
