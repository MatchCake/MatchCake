from typing import Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from matchcake.utils.torch_utils import to_numpy, to_tensor


class Kernel(torch.nn.Module, TransformerMixin, BaseEstimator):
    """
    Kernel class that extends ``torch.nn.Module``, ``TransformerMixin``, and ``BaseEstimator``.

    This class provides a base structure for implementing a kernel model with features like
    forward computation, data transformation, and model fitting. It is designed to handle
    both NumPy arrays and PyTorch tensors, offering functionality for model freezing and easy
    access to the device where the model parameters are stored. It is intended to be subclassed
    for specific kernel functionality.

    :ivar gram_batch_size: Batch size for gram matrix computation.
    :type gram_batch_size: int
    :ivar random_state: Random state seed for reproducibility.
    :type random_state: int
    """

    DEFAULT_GRAM_BATCH_SIZE = 10_000

    def __init__(
        self,
        *,
        gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = 0,
    ):
        """
        Initializes the class with configurations related to batch sizes and random state.

        :param gram_batch_size: The batch size to be used for processing gram matrices.
        :param random_state: The seed value for ensuring reproducible random number generation.
        """
        super().__init__()
        self.gram_batch_size = gram_batch_size
        self.random_state = random_state
        self.np_rn_gen = np.random.RandomState(seed=random_state)
        self.x_train_: Optional[Union[NDArray, torch.Tensor]] = None
        self.y_train_: Optional[Union[NDArray, torch.Tensor]] = None
        self.is_fitted_ = False

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the forward pass of the model with the given inputs.

        The forward method defines the computation performed at every call of the
        model. This typically involves applying a sequence of operations on input
        tensors to produce the output tensor(s). The specific implementation details
        are to be defined in subclasses.

        :param x0: The primary input tensor for the forward
            pass. This tensor is expected to have a shape suitable for the compute
            module being used in the forward method.
        :param x1: An optional secondary input tensor for the forward
            pass. If provided, it is used in conjunction with the primary input tensor
            depending on the specific implementation of the forward pass logic.
        :return: The output tensor resulting from the computation of the forward
            pass.

        """
        raise NotImplementedError

    def transform(self, x: Union[NDArray, torch.Tensor]) -> Union[NDArray, torch.Tensor]:
        """
        Transforms the input using a specified transformation process. The transformation
        is performed on the provided input data, converting it into a tensor if necessary
        and applying the forward operation with the trained data. If the original input
        was not a tensor, it converts the transformed data back to a numpy array.

        :param x: Input data to be transformed. Can be either a NumPy array or a PyTorch
                  tensor.
        :type x: Union[NDArray, torch.Tensor]
        :return: The transformed data. The format matches the input type (a NumPy array
                 or a PyTorch tensor).
        :rtype: Union[NDArray, torch.Tensor]
        """
        is_torch = isinstance(x, torch.Tensor)
        x: torch.Tensor = to_tensor(x).to(device=self.device)  # type: ignore
        x: torch.Tensor = self.forward(x, self.x_train_)  # type: ignore
        if is_torch:
            return x
        return to_numpy(x, dtype=np.float32)  # pragma: no cover

    def fit(self, x_train: Union[NDArray, torch.Tensor], y_train: Optional[Union[NDArray, torch.Tensor]] = None):
        """
        Fits the model using the provided training data. This method stores the
        training data and sets the model's state to fitted.

        :param x_train: Training input data. It can be a NumPy array
            or a Torch tensor.
        :type x_train: Union[NDArray, torch.Tensor]
        :param y_train: Training target data. It can be a NumPy array,
            a Torch tensor, or None. Defaults to None.
        :type y_train: Optional[Union[NDArray, torch.Tensor]]
        :return: The fitted model instance.
        :rtype: self
        """
        self.x_train_ = x_train
        self.y_train_ = y_train
        self.is_fitted_ = True
        return self

    def predict(self, x: Union[NDArray, torch.Tensor]) -> Union[NDArray, torch.Tensor]:
        """
        Predict transformed data using the model.

        This function applies a transformation to the input data and returns the
        resulting output. It supports both NumPy arrays and PyTorch tensors as input.
        The return type will match the input type.

        :param x: The input data to be transformed. Can be either a NumPy array
            or a PyTorch tensor.
        :return: The transformed input data. The returned type will match the
            type of the input data (NumPy array or PyTorch tensor).
        """
        return self.transform(x)

    def freeze(self):
        """
        Freezes the model by setting it to evaluation mode and disabling gradient computations.

        This method is typically used for inference to ensure the model does not update its
        parameters during forward passes. It switches the model to evaluation mode and sets
        all parameters to not require gradients.

        :return: The model instance with evaluation mode enabled and gradient computations
                 disabled.
        :rtype: self
        """
        self.eval()
        self.requires_grad_(False)
        return self

    @property
    def device(self):
        """
        Provides the current device of the model parameters. If the model has no parameters,
        it defaults to the CPU device. This property is useful for determining where the
        model is currently located, either on CPU or a specific GPU.

        :return: The device on which the model's parameters reside, or 'cpu' if there are
                 no parameters.
        :rtype: torch.device
        """
        if len(list(self.parameters())) == 0:
            return torch.device("cpu")
        return list(self.parameters())[0].device
