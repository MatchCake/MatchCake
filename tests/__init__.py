import pytest
import numpy as np
from .configs import TEST_SEED

np.random.seed(TEST_SEED)


def get_slow_test_mark():
    from .configs import RUN_SLOW_TESTS

    return pytest.mark.skipif(
        not RUN_SLOW_TESTS,
        reason=f"Only run when configs.{RUN_SLOW_TESTS} is True.",
    )
