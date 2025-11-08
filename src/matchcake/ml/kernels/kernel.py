from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from matchcake.utils.torch_utils import to_tensor, to_numpy


class Kernel(torch.nn.Module, TransformerMixin, BaseEstimator):
    DEFAULT_GRAM_BATCH_SIZE = 10_000

    def __init__(
            self,
            *,
            gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
            random_state: int = 0,
    ):
        super().__init__()
        self.gram_batch_size = gram_batch_size
        self.random_state = random_state
        self.np_rn_gen = np.random.RandomState(seed=random_state)
        self.x_train_ = None
        self.y_train_ = None
        self.is_fitted_ = False

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def transform(self, x: Union[NDArray, torch.Tensor]) -> Union[NDArray, torch.Tensor]:
        is_torch = isinstance(x, torch.Tensor)
        x = to_tensor(x).to(device=self.device)
        x = self.forward(x, self.x_train_)
        if is_torch:
            return x
        return to_numpy(x, dtype=np.float32)

    def fit(self, x_train: Union[NDArray, torch.Tensor], y_train: Optional[Union[NDArray, torch.Tensor]] = None):
        self.x_train_ = x_train
        self.y_train_ = y_train
        self.is_fitted_ = True
        return self

    def predict(self, x: Union[NDArray, torch.Tensor]) -> Union[NDArray, torch.Tensor]:
        return self.transform(x)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self

    @property
    def device(self):
        if len(list(self.parameters())) == 0:
            return torch.device("cpu")
        return list(self.parameters())[0].device
