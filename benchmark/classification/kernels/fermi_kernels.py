from typing import Optional

import pennylane as qml

from . import matchcake
from matchcake.ml.ml_kernel import (
    FermionicPQCKernel,
    WideFermionicPQCKernel,
)


class NeighboursFermionicPQCKernel(FermionicPQCKernel):
    def pre_initialize(self):
        self._device = matchcake.NonInteractingFermionicDevice(wires=self.size, contraction_method="neighbours")
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)


class HFermionicPQCKernel(FermionicPQCKernel):
    pass


class CudaFermionicPQCKernel(FermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["use_cuda"] = True
        super().__init__(size=size, **kwargs)


class CpuFermionicPQCKernel(FermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["use_cuda"] = False
        super().__init__(size=size, **kwargs)


class CudaFermionicPQCKernel(FermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["use_cuda"] = True
        super().__init__(size=size, **kwargs)


class CudaWideFermionicPQCKernel(WideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["use_cuda"] = True
        super().__init__(size=size, **kwargs)


class CpuWideFermionicPQCKernel(WideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["use_cuda"] = False
        super().__init__(size=size, **kwargs)


class FastCudaWideFermionicPQCKernel(CudaWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fast_fcnot"
        super().__init__(size=size, **kwargs)


class FastCudaFermionicPQCKernel(CudaFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fast_fcnot"
        super().__init__(size=size, **kwargs)


class SwapCudaFermionicPQCKernel(CudaFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fswap"
        super().__init__(size=size, **kwargs)


class SwapCudaWideFermionicPQCKernel(CudaWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fswap"
        super().__init__(size=size, **kwargs)


class IdentityCudaWideFermionicPQCKernel(CudaWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "identity"
        super().__init__(size=size, **kwargs)


class IdentityCudaFermionicPQCKernel(CudaFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "identity"
        super().__init__(size=size, **kwargs)


class HadamardCudaFermionicPQCKernel(CudaFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "hadamard"
        super().__init__(size=size, **kwargs)


class HadamardCudaWideFermionicPQCKernel(CudaWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "hadamard"
        super().__init__(size=size, **kwargs)


class SwapCpuFermionicPQCKernel(CpuFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fswap"
        super().__init__(size=size, **kwargs)


class SwapCpuWideFermionicPQCKernel(CpuWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "fswap"
        super().__init__(size=size, **kwargs)


class IdentityCpuWideFermionicPQCKernel(CpuWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "identity"
        super().__init__(size=size, **kwargs)


class IdentityCpuFermionicPQCKernel(CpuFermionicPQCKernel):
    # TODO: Use MPS simulator
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "identity"
        super().__init__(size=size, **kwargs)


class HadamardCpuFermionicPQCKernel(CpuFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "hadamard"
        super().__init__(size=size, **kwargs)


class HadamardCpuWideFermionicPQCKernel(CpuWideFermionicPQCKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        kwargs["entangling_mth"] = "hadamard"
        super().__init__(size=size, **kwargs)
