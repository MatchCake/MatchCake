import datetime
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pennylane as qml
import pythonbasictools as pbt
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm

from ...utils import torch_utils
from ..std_estimator import StdEstimator


class MLKernel(StdEstimator):
    _TO_NUMPY_ON_PICKLE = ["_parameters"]

    def __init__(self, size: Optional[int] = None, **kwargs):
        self._size = size
        self.nb_workers = kwargs.get("nb_workers", 0)
        self.batch_size = kwargs.get("batch_size", 32)
        self._assume_symmetric = kwargs.get("assume_symmetric", True)
        self._assume_diag_one = kwargs.get("assume_diag_one", True)
        self._batch_size_try_counter = 0
        self._gram_type = kwargs.get("gram_type", "ndarray")
        self._parameters = kwargs.get("parameters", None)
        self.seed = kwargs.get("seed", 0)
        self.parameters_rng = np.random.default_rng(seed=self.seed)
        self.use_cuda = kwargs.get("use_cuda", False)
        if self._gram_type not in {"ndarray", "hdf5"}:
            raise ValueError(f"Unknown gram type: {self._gram_type}.")
        super().__init__(**kwargs)

    @property
    def size(self):
        return self._size

    @property
    def parameters(self):
        if self.use_cuda:
            return self.cast_tensor_to_interface(self._parameters)
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        if self.use_cuda:
            self._parameters = self.cast_tensor_to_interface(parameters)
        # elif self.qnode.interface == "torch":
        #     import torch
        #     self.parameters = torch.from_numpy(self.parameters).float().requires_grad_(True)
        # elif not isinstance(parameters, np.ndarray):
        #     self._parameters = torch_utils.to_numpy(parameters)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.pre_initialize()

    def cast_tensor_to_interface(self, tensor):
        if self.use_cuda:
            return torch_utils.to_cuda(tensor)
        return tensor

    def _compute_default_size(self):
        return self.X_.shape[-1]

    def pre_initialize(self):
        pass

    def initialize_parameters(self):
        self.parameters_rng = np.random.default_rng(seed=self.seed)

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y)

        if self._size is None:
            self._size = self._compute_default_size()
        self.pre_initialize()
        self.initialize_parameters()
        return self

    def transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return x

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def single_distance(self, x0, x1, **kwargs):
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")

    def batch_distance_in_sequence(self, x0, x1, **kwargs):
        assert qml.math.ndim(x0) > 1, f"Expected x0 to be a batch of vectors, got {qml.math.shape(x0)}."
        if qml.math.ndim(x1) == qml.math.ndim(x0):
            distances = [self.single_distance(_x0, _x1, **kwargs) for _x0, _x1 in zip(x0, x1)]
        else:
            distances = [self.single_distance(x, x1, **kwargs) for x in x0]
        return qml.math.asarray(distances)

    def batch_distance(self, x0, x1, **kwargs):
        return self.batch_distance_in_sequence(x0, x1, **kwargs)

    def get_batch_size_for(self, length: int):
        if self.batch_size == "try":
            return length // 2**self._batch_size_try_counter
        elif self.batch_size == "sqrt":
            return int(np.sqrt(length))
        elif self.batch_size == 0:
            return 1
        elif self.batch_size < 0:
            return length
        elif self.batch_size > length:
            return length
        return self.batch_size

    def make_batches_generator(self, x, **kwargs) -> Tuple[iter, int]:
        length = qml.math.shape(x)[0]
        batch_size = self.get_batch_size_for(length)
        n_batches = int(np.ceil(length / batch_size))

        def _gen():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                yield x[start_idx:end_idx]

        return _gen(), n_batches

    def pairwise_distances_in_sequence(self, x0, x1, **kwargs):
        if kwargs.get("check_array", True):
            x0 = check_array(x0)
            x1 = check_array(x1)
        self.check_is_fitted()
        verbose = kwargs.pop("verbose", False)
        desc = kwargs.pop(
            "desc",
            f"{self.__class__.__name__}: pairwise_distances(x0:{qml.math.shape(x0)}, x1:{qml.math.shape(x1)})",
        )
        _list_results = pbt.apply_func_multiprocess(
            func=self.batch_distance,
            iterable_of_args=[(x0, b) for b in x1],
            iterable_of_kwargs=[kwargs for _ in range(len(x1))],
            nb_workers=self.nb_workers,
            verbose=verbose,
            desc=desc,
        )
        _result = np.stack(_list_results, axis=-1)
        return _result

    def pairwise_distances_in_batch(self, x0, x1, **kwargs):
        r"""
        Compute pairwise distances between two sets of vectors in batch.

        :param x0: First set of vectors of shape (n_samples_0, n_features)
        :param x1: Second set of vectors of shape (n_samples_1, n_features)
        :param kwargs: Additional keyword arguments

        :keyword check_array: Whether to check the input arrays. Default is True.
        :keyword p_bar: Progress bar. Default is None.

        :return: Pairwise distances between x0 and x1 of shape (n_samples_0, n_samples_1)
        """

        if kwargs.get("check_array", True):
            x0 = check_array(x0)
            x1 = check_array(x1)
        self.check_is_fitted()
        p_bar: Optional[tqdm] = kwargs.pop("p_bar", None)
        p_bar_postfix_str = p_bar.postfix if p_bar is not None else ""
        if p_bar is not None:
            p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: ?, ?%)")
        gram = np.zeros((qml.math.shape(x0)[0], qml.math.shape(x1)[0]))
        gram = self.cast_tensor_to_interface(gram)
        is_square = gram.shape[0] == gram.shape[1]
        triu_indices = np.stack(np.triu_indices(n=gram.shape[0], m=gram.shape[1], k=1), axis=-1)
        if is_square:
            indices = triu_indices
        else:
            tril_indices = np.stack(np.tril_indices(n=gram.shape[0], m=gram.shape[1], k=-1), axis=-1)
            indices = np.concatenate([triu_indices, tril_indices], axis=0)
        start_time = time.perf_counter()
        n_data = qml.math.shape(indices)[0]
        n_done = 0
        batch_gen, n_batches = self.make_batches_generator(indices, **kwargs)
        for i, b_idx in enumerate(batch_gen):
            b_x0, b_x1 = x0[b_idx[:, 0]], x1[b_idx[:, 1]]
            batched_distances = self.batch_distance(b_x0, b_x1, **kwargs)
            gram[b_idx[:, 0], b_idx[:, 1]] += self.cast_tensor_to_interface(batched_distances)
            if p_bar is not None:
                n_done += qml.math.shape(b_x0)[0]
                curr_time = time.perf_counter()
                eta = (curr_time - start_time) / n_done * (n_data - n_done)
                eta_fmt = datetime.timedelta(seconds=eta)
                p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: {eta_fmt}, {100 * n_done / n_data:.2f}%)")
        if is_square:
            gram = gram + gram.T
        gram = self.gram_diagonal_fill(gram)
        if p_bar is not None:
            p_bar.set_postfix_str(p_bar_postfix_str)
        return gram

    def gram_diagonal_fill(self, gram):
        if isinstance(gram, np.ndarray):
            np.fill_diagonal(gram, 1.0)
        elif qml.math.get_interface(gram) == "torch":
            gram.fill_diagonal_(1.0)
        else:
            qml.math.fill_diagonal(gram, 1.0)
        return gram

    def pairwise_distances(self, x0, x1, **kwargs):
        r"""
        Compute pairwise distances between two sets of vectors.

        :param x0: First set of vectors of shape (n_samples_0, n_features)
        :param x1: Second set of vectors of shape (n_samples_1, n_features)
        :param kwargs: Additional keyword arguments

        :keyword check_array: Whether to check the input arrays. Default is True.

        :return: Pairwise distances between x0 and x1 of shape (n_samples_0, n_samples_1)
        """
        if kwargs.get("check_array", True):
            x0 = check_array(x0)
            x1 = check_array(x1)
        self.check_is_fitted()
        if self.batch_size == 0:
            return self.pairwise_distances_in_sequence(x0, x1, **kwargs)
        # TODO: add support for batch_size = try
        return self.pairwise_distances_in_batch(x0, x1, **kwargs)

    def compute_gram_matrix(self, x, **kwargs):
        kwargs.setdefault(
            "desc",
            f"{self.__class__.__name__}: "
            f"compute_gram_matrix("
            f"x:{qml.math.shape(x)}"
            f", batch_size={self.batch_size}"
            f")",
        )
        return self.pairwise_distances(x, x, **kwargs)

    def predict(self, x, **kwargs):
        return self.pairwise_distances(x, self.X_, **kwargs)
