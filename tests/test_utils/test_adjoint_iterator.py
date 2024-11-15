import pytest
from matchcake.utils.operators import adjoint_generator


class DummyOp:

    @classmethod
    def iterator(cls, n):
        for i in range(n):
            yield cls(i)

    def __init__(self, value):
        self.value = value

    def adjoint(self):
        return DummyOp(-self.value)


def test_adjoint_generator():
    n = 10
    it = adjoint_generator(DummyOp.iterator(n))
    for i in reversed(range(n)):
        assert next(it).value == DummyOp(i).adjoint().value



