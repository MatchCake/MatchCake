from numbers import Number
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Callable, Iterator, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from matchcake.utils.torch_utils import to_numpy, to_tensor


class GramMatrix:
    """
    Represents a Gram matrix structure for storing symmetric matrix data. The class provides
    utilities for managing memory efficiently, either on disk via `numpy.memmap` or in memory
    using PyTorch tensors, depending on whether gradient computation is required. It supports
    in-place operations, symmetric updates, and provides tools for batch processing.

    :ivar shape: Shape of the Gram matrix.
    :type shape: Tuple[int, ...]
    :ivar requires_grad: Whether the matrix requires gradient computation.
    :type requires_grad: bool
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        initial_value: float = 0.0,
        requires_grad: bool = False,
    ):
        self._shape = shape
        self._initial_value = initial_value
        self._filepath = Path(mkstemp()[1])
        self._requires_grad = requires_grad
        self._memmap = None
        self._tensor = None
        if self.requires_grad:
            self._tensor = torch.full(self._shape, initial_value, dtype=torch.float32)
        else:
            self._memmap = np.memmap(
                filename=self._filepath,
                dtype="float32",
                mode="r+",
                shape=self._shape,
            )
            self._memmap.fill(initial_value)
            self._memmap.flush()

    def __getitem__(self, item) -> torch.Tensor:
        """
        Retrieves a tensor or an item from the given index.

        If gradients are required, the method fetches the item from the in-memory tensor.
        Otherwise, it fetches the item from the memory-mapped file and converts it to a
        Tensor with the specified data type.

        :param item: The index or slice to access the desired element(s).
        :type item: Any
        :return: The tensor or data corresponding to the given index or slice.
        :rtype: torch.Tensor
        """
        if self.requires_grad:
            return self._tensor[item]  # type: ignore
        return to_tensor(self._memmap[item], dtype=torch.float32)  # type: ignore

    def __setitem__(self, key, value):
        """
        Sets the value at the specified key in the corresponding storage. If `requires_grad`
        is True, the value is converted to a tensor and stored. Otherwise, the value is
        converted to a NumPy array and saved in the in-memory map, ensuring it is flushed
        to persist changes.

        :param key: The key at which the value should be stored.
        :type key: Any
        :param value: The value to be stored at the specified key.
        :type value: Any
        """
        if self.requires_grad:
            self._tensor[key] = to_tensor(value, dtype=torch.float32)  # type: ignore
        else:
            self._memmap[key] = to_numpy(value, dtype=np.float32)  # type: ignore
            self._memmap.flush()

    def apply_(
        self,
        func: Callable[[NDArray], Any],
        batch_size: int = 32,
        symmetrize: bool = True,
    ) -> "GramMatrix":
        """
        Apply a given function to a batch of indices in the Gram matrix and optionally
        symmetrize the matrix.

        The `apply_` method processes the Gram matrix in batches of indices and applies
        the provided function to update the matrix values at those indices. After processing
        the batches, the matrix can be symmetrized if specified.

        :param func: A callable that accepts an ndarray of indices and performs an operation
            on the corresponding elements in the matrix.
        :param batch_size: The size of the batches in which the Gram matrix is processed.
            Default is 32.
        :param symmetrize: Whether to symmetrize the matrix after applying `func`.
            Default is True.
        :return: Returns the updated instance of the Gram matrix.
        """
        for indices in self.indices_batch_generator(batch_size):
            self[indices[:, 0], indices[:, 1]] = func(indices)
        if symmetrize:
            self.symmetrize_()
        return self

    def to_tensor(self) -> torch.Tensor:
        """
        Converts the contained data to a PyTorch tensor.

        If `requires_grad` is True, the method returns the original tensor associated
        with the object. Otherwise, it converts the `_memmap` data to a PyTorch tensor
        with `float32` dtype.

        :return: The PyTorch tensor representation of the contained data.
        :rtype: torch.Tensor
        """
        if self.requires_grad:
            return self._tensor  # type: ignore
        return torch.from_numpy(self._memmap).to(dtype=torch.float32)  # type: ignore

    def symmetrize_(self) -> "GramMatrix":
        """
        Creates a symmetric version of the matrix by mirroring its values across the main
        diagonal. Entries below the diagonal are updated to match their corresponding
        entries above the diagonal for matrices with more rows than columns.
        Conversely, entries above the diagonal are updated to match their counterparts
        below the diagonal for matrices with more columns than rows. The diagonal
        values of the matrix are filled with ones.

        :return: A symmetric version of the current matrix.
        :rtype: GramMatrix
        """
        if self.shape[0] < self.shape[1]:
            indices = np.tril_indices(n=self.shape[0], m=self.shape[1], k=-1)
        else:
            indices = np.triu_indices(n=self.shape[0], m=self.shape[1], k=1)
        self[indices[0], indices[1]] = self[indices[1], indices[0]]
        self._fill_diagonal_()
        return self

    def _fill_diagonal_(self) -> "GramMatrix":
        """
        Fills the diagonal of the matrix with a value of 1.0. This operation is
        performed in-place, altering the internal tensor or memmap based on whether
        gradient tracking is enabled. Returns the updated matrix instance.

        :param self: The object instance of the GramMatrix.
        :return: The updated GramMatrix instance with the diagonal filled.
        """
        if self.requires_grad:
            self._tensor.fill_diagonal_(1.0)  # type: ignore
        else:
            np.fill_diagonal(self._memmap, 1.0)  # type: ignore
            self._memmap.flush()  # type: ignore
        return self

    def __del__(self):
        """
        Handles the deletion of the file associated with the current object upon its destruction.

        This method is automatically invoked when the object is about to be destroyed. It attempts
        to remove the file used by the memmap. If the operation is unsuccessful,
        errors are silently ignored.
        """
        try:
            self._memmap = None
            self._filepath.unlink(missing_ok=True)
        except:  # pragma: no cover
            pass  # pragma: no cover

    def indices_batch_generator(self, batch_size: int = 32) -> Iterator[NDArray[np.int64]]:
        """
        Generates batches of indices in the form of tuples within the constraints of a given
        shape and selection logic. The generator iterates over all pairwise index combinations
        (i, j) of an array defined by the object's shape attribute. The generator yields these
        indices batch by batch, depending on the specified batch size.

        :param batch_size: The maximum number of (i, j) index pairs to be included in each batch.
            Defaults to 32.
        :return: Iterator that yields batches of index pairs as numpy arrays of shape
            (batch_size, 2). Each batch contains tuples of indices in the form (i, j),
            respecting the constraints based on the dimensions of the object's shape.
        """
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
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
