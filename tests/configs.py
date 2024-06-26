
N_RANDOM_TESTS_PER_CASE = 10
TEST_SEED = 42

ATOL_MATRIX_COMPARISON = 1e-5
RTOL_MATRIX_COMPARISON = 1e-5

ATOL_SCALAR_COMPARISON = 1e-7
RTOL_SCALAR_COMPARISON = 1e-7

ATOL_APPROX_COMPARISON = 2e-2
RTOL_APPROX_COMPARISON = 5e-2

ATOL_SHAPE_COMPARISON = 0
RTOL_SHAPE_COMPARISON = 0

RUN_SLOW_TESTS = True


def set_seed(seed: int = TEST_SEED):
    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.random.manual_seed(TEST_SEED)
    except ImportError:
        pass
    return seed
