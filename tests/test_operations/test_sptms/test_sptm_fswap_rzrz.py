import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP,
    SptmRzRz,
    SptmFSwapRzRz,
)
from matchcake.operations.single_particle_transition_matrices import (
    SptmFSwap,
)
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "wire0, wire1, all_wires, params",
    [
        (wire0, wire1, n_wires, np.random.choice(SptmRzRz.ALLOWED_ANGLES, size=2))
        for n_wires in range(2, 16)
        for wire0 in range(n_wires - 1)
        for wire1 in range(wire0 + 1, n_wires)
    ],
)
def test_sptm_fswap_rzrz_chain_equal_to_sptm_fswap_rzrz(wire0, wire1, all_wires, params):
    all_wires = list(range(all_wires))

    def _gen():
        yield SptmFSwap(wires=[wire0, wire1])
        yield SptmRzRz(params, wires=[wire1 - 1, wire1])
        yield SptmFSwap(wires=[wire1, wire0])
        return

    device = mc.NIFDevice(wires=all_wires)
    device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
    chain_sptm = device.apply_metadata["global_sptm"]
    sptm = SptmFSwapRzRz(params, wires=[wire0, wire1]).matrix(all_wires)

    np.testing.assert_allclose(
        sptm,
        chain_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wire0, wire1, all_wires, params",
    [
        (wire0, wire1, n_wires, np.random.choice(SptmRzRz.ALLOWED_ANGLES, size=2))
        for n_wires in range(2, 16)
        for wire0 in range(n_wires - 1)
        for wire1 in range(wire0 + 1, n_wires)
    ],
)
def test_sptm_fswap_rzrz(wire0, wire1, all_wires, params):
    all_wires = list(range(all_wires))

    def _gen():
        yield SptmFSwap(wires=[wire0, wire1])
        yield SptmRzRz(params, wires=[wire1 - 1, wire1])
        yield SptmFSwap(wires=[wire1, wire0])
        return

    device = mc.NIFDevice(wires=all_wires)
    device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
    chain_sptm = device.apply_metadata["global_sptm"]
    sptm = SptmFSwapRzRz(params, wires=[wire0, wire1]).matrix(all_wires)

    np.testing.assert_allclose(
        sptm,
        chain_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
