import os
import sys
from typing import Optional

from sklearn.metrics.pairwise import pairwise_kernels

try:
    import matchcake
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    import matchcake

from matchcake.ml.kernels.ml_kernel import (
    MLKernel,
)

from .pennylane_kernels import (
    MPennylaneQuantumKernel,
    CPennylaneQuantumKernel,
    PQCKernel,
    IdentityPQCKernel,
    LightningPQCKernel,
)
from .fermi_kernels import (
    NeighboursFermionicPQCKernel,
    CudaFermionicPQCKernel,
    CpuFermionicPQCKernel,
    FastCudaFermionicPQCKernel,
    SwapCudaFermionicPQCKernel,
    IdentityCudaFermionicPQCKernel,
    HadamardCudaFermionicPQCKernel,
    SwapCpuFermionicPQCKernel,
    IdentityCpuFermionicPQCKernel,
    HadamardCpuFermionicPQCKernel,
)


class ClassicalKernel(MLKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self._metric = kwargs.get("metric", "linear")

    def single_distance(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)

    def batch_distance(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)

    def pairwise_distances(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)
