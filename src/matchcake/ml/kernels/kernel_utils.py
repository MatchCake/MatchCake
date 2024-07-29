import warnings
from typing import Optional, Tuple, Union

import numpy as np

from ...operations import fRZZ


class GramMatrixKernel:
    def __init__(
            self,
            shape: Tuple[int, int],
            dtype: Optional[Union[str, np.dtype]] = np.float64,
            array_type: str = "table",
            **kwargs
    ):
        self._shape = shape
        self._dtype = dtype
        self._array_type = array_type
        self.kwargs = kwargs
        self.h5file = None
        self.data = None
        self._initiate_data_()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_square(self):
        return self.shape[0] == self.shape[1]

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def _initiate_data_(self):
        if self._array_type == "table":
            import tables as tb
            self.h5file = tb.open_file("gram_matrix.h5", mode="w")
            self.data = self.h5file.create_carray(
                self.h5file.root,
                "data",
                tb.Float64Atom(),
                shape=self.shape,
            )
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)

    def __array__(self):
        return self.data[:]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def symmetrize(self):
        if not self.is_square:
            warnings.warn("Cannot symmetrize a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = (self.data + self.data.T) / 2
        elif self._array_type == "ndarray":
            self.data = (self.data + self.data.T) / 2
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def mirror(self):
        if not self.is_square:
            warnings.warn("Cannot mirror a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def triu_reflect(self):
        if not self.is_square:
            warnings.warn("Cannot triu_reflect a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data + self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data + self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def tril_reflect(self):
        if not self.is_square:
            warnings.warn("Cannot tril_reflect a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data + self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data + self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def fill_diagonal(self, value: float):
        if not self.is_square:
            warnings.warn("Cannot fill_diagonal a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[np.diag_indices(self.shape[0])] = value
        elif self._array_type == "ndarray":
            np.fill_diagonal(self.data, value)
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def close(self):
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
            self.data = None

    def make_batches_indexes_generator(self, batch_size: int) -> Tuple[iter, int]:
        if self.is_square:
            # number of elements in the upper triangle without the diagonal elements
            length = self.size * (self.size - 1) // 2
        else:
            length = self.size - self.shape[0]
        n_batches = int(np.ceil(length / batch_size))

        def _gen():
            for i in range(n_batches):
                b_start_idx = i * batch_size
                b_end_idx = (i + 1) * batch_size
                # generate the coordinates of the matrix elements in that batch
                yield np.unravel_index(
                    np.arange(b_start_idx, b_end_idx),
                    self.shape
                )

        return _gen(), n_batches


def mrot_zz_template(param0, param1, wires):
    fRZZ([param0, param1], wires=wires)
