from functools import cached_property
from typing import Optional, Union, Tuple
from dataclasses import dataclass, fields
import torch
import pennylane as qml


@dataclass
class MatchgateParams:
    def matrix(self, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError()

    def compute_batch_size(self) -> Optional[int]:
        shapes = [qml.math.shape(p) for p in fields(self) if p is not None]
        batch_sizes = list(set([s[0] for s in shapes if len(s) > 0]))
        assert len(batch_sizes) <= 1, f"Expect the same batch size for every parameters. Got: {batch_sizes}."
        batch_size = batch_sizes[0] if len(batch_sizes) > 0 else None
        return batch_size

    def compute_shape(self) -> Union[Tuple[int], Tuple[int, int]]:
        n_fields = len(fields(self))
        if self.batch_size is None:
            return n_fields,
        return self.batch_size, n_fields

    @cached_property
    def batch_size(self) -> Optional[int]:
        return self.compute_batch_size()

    @cached_property
    def shape(self) -> Union[Tuple[int], Tuple[int, int]]:
        return self.compute_shape()
