from numbers import Number
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Callable, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from matchcake.utils.torch_utils import to_numpy, to_tensor


class GramMatrix:
    def __init__(
        self,
        shape: Tuple[int, ...],
        initial_value: Number = 0.0,
        requires_grad: bool = False,
    ):
        self._shape = shape
        self._initial_value = initial_value
        self._filepath = Path(mkstemp()[1])
        self._requires_grad = requires_grad
        self._memmap = None
        self._tensor = None
        if self.requires_grad:
            self._tensor = torch.zeros(self._shape, dtype=torch.float32)
        else:
            self._memmap = np.memmap(
                filename=self._filepath,
                dtype="float32",
                mode="r+",
                shape=self._shape,
            )

    def __getitem__(self, item):
        if self.requires_grad:
            return self._tensor[item]
        return to_tensor(self._memmap[item], dtype=torch.float32)

    def __setitem__(self, key, value):
        if self.requires_grad:
            self._tensor[key] = to_tensor(value, dtype=torch.float32)
        else:
            self._memmap[key] = to_numpy(value, dtype=np.float32)

    def apply_(
        self,
        func: Callable[[NDArray], Any],
        batch_size: int = 32,
        symmetrize: bool = True,
    ):
        for indices in self.indices_batch_generator(batch_size):
            self[indices[:, 0], indices[:, 1]] = func(indices)
        if symmetrize:
            self.symmetrize_()
        return self

    def to_tensor(self):
        if self.requires_grad:
            return self._tensor
        return torch.from_numpy(self._memmap).to(dtype=torch.float32)

    def symmetrize_(self):
        if self.shape[0] < self.shape[1]:
            indices = np.tril_indices(n=self.shape[0], m=self.shape[1], k=-1)
        else:
            indices = np.triu_indices(n=self.shape[0], m=self.shape[1], k=1)
        self[indices[0], indices[1]] = self[indices[1], indices[0]]
        self._fill_diagonal_()
        return self

    def _fill_diagonal_(self):
        if self.requires_grad:
            self._tensor.fill_diagonal_(1.0)
        else:
            np.fill_diagonal(self._memmap, 1.0)

    def __del__(self):
        try:
            self._filepath.unlink(missing_ok=True)
        except:
            pass

    def indices_batch_generator(self, batch_size: int = 32):
        batch = []
        for i, j in np.ndindex(self.shape):
            if j > i and self.shape[0] < self.shape[1]:
                batch.append((i, j))
            elif i > j and self.shape[0] >= self.shape[1]:
                batch.append((i, j))
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
        if len(batch) > 0:
            yield np.stack(batch, axis=0)
        return

    @property
    def shape(self):
        return self._shape

    @property
    def requires_grad(self):
        return self._requires_grad
