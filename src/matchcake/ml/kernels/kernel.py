from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from matchcake.typing import TensorLike
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
        alignment: bool = False,
        alignment_iterations: int = 100,
        alignment_learning_rate: float = 1e-3,
        alignment_early_stopping_patience: int = 10,
        alignment_early_stopping_threshold: float = 1e-5,
    ):
        """
        Initializes the class with configurations related to batch sizes and random state.

        :param gram_batch_size: The batch size to be used for processing gram matrices.
        :param random_state: The seed value for ensuring reproducible random number generation.
        """
        super().__init__()
        self.gram_batch_size = gram_batch_size
        self.random_state = random_state
        self.alignment = alignment
        self.alignment_iterations = alignment_iterations
        self.alignment_learning_rate = alignment_learning_rate
        self.alignment_early_stopping_patience = alignment_early_stopping_patience
        self.alignment_early_stopping_threshold = alignment_early_stopping_threshold
        self.np_rn_gen = np.random.RandomState(seed=random_state)
        self.x_train_: Optional[Union[NDArray, TensorLike]] = None
        self.y_train_: Optional[Union[NDArray, TensorLike]] = None
        self.opt_: Optional[OptimizeResult] = None
        self.is_fitted_: bool = False
        self.device_tracker_param = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, x0: TensorLike, x1: Optional[TensorLike] = None, **kwargs) -> torch.Tensor:
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

    def transform(self, x: Union[NDArray, TensorLike]) -> Union[NDArray, torch.Tensor]:
        """
        Transforms the input using a specified transformation process. The transformation
        is performed on the provided input data, converting it into a tensor if necessary
        and applying the forward operation with the trained data. If the original input
        was not a tensor, it converts the transformed data back to a numpy array.

        The model is set to evaluation mode during prediction to ensure that layers
        like dropout and batch normalization behave appropriately for inference.
        The transformation is performed without tracking gradients to optimize performance.

        :param x: Input data to be transformed. Can be either a NumPy array or a PyTorch
                  tensor.
        :type x: Union[NDArray, TensorLike]
        :return: The transformed data. The format matches the input type (a NumPy array
                 or a PyTorch tensor).
        :rtype: Union[NDArray, torch.Tensor]
        """
        is_torch = isinstance(x, torch.Tensor)
        x: torch.Tensor = to_tensor(x)  # type: ignore
        self.eval()
        with torch.no_grad():
            x: torch.Tensor = self(x, self.x_train_, x1_train=True)  # type: ignore
        if is_torch:
            return x
        return to_numpy(x, dtype=np.float32)  # pragma: no cover

    def fit(self, x_train: Union[NDArray, TensorLike], y_train: Optional[Union[NDArray, TensorLike]] = None):
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
        self.build_model()
        if self.alignment:
            self._align_kernel()
        self.is_fitted_ = True
        return self

    def predict(self, x: Union[NDArray, TensorLike]) -> Union[NDArray, torch.Tensor]:
        """
        Predict transformed data using the model.

        This function applies a transformation to the input data and returns the
        resulting output. It supports both NumPy arrays and PyTorch tensors as input.
        The return type will match the input type.

        The model is set to evaluation mode during prediction to ensure that layers
        like dropout and batch normalization behave appropriately for inference.
        The transformation is performed without tracking gradients to optimize performance.

        :param x: The input data to be transformed. Can be either a NumPy array
            or a PyTorch tensor.
        :return: The transformed input data. The returned type will match the
            type of the input data (NumPy array or PyTorch tensor).
        """
        self.eval()
        with torch.no_grad():
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

    def build_model(self) -> "Kernel":
        """
        Execute a forward pass with the training data to make sure the model's internal structures are properly
        initialized.

        :return: The model instance after building the internal structures.
        :rtype: self
        """
        assert self.x_train_ is not None, "Training data must be provided to build the model."
        with torch.no_grad():
            x_dummy = self.x_train_[0:3] if len(self.x_train_) >= 3 else self.x_train_
            self(x_dummy)
        return self

    def _align_kernel(
        self,
        verbose: bool = False,
    ):
        r"""
        Aligns the kernel matrix using the training data.

        This method adjusts the kernel matrix based on the training data to ensure
        that it is properly aligned using gradient-based optimization. The kernel
        alignment score measures the similarity between the kernel matrix and the
        ideal kernel derived from the target labels.

        The alignment is computed as:

        .. :math::
            \text{alignment} = \frac{\langle K_c, Y_c \rangle}{||K_c|| * ||Y_c||}

        where

        .. :math::
            K_c = C K C,

        .. :math::
            Y_c = C Y C,

        and :math:`C` is the centering matrix. Here, :math:`\langle\cdot,\cdot\rangle` denotes the Frobenius inner product,
        and ||.|| denotes the Frobenius norm, with :math:`K` being the kernel matrix and :math:`Y` being
        the ideal kernel from labels.

        where K is the kernel matrix and Y is the ideal kernel from labels.

        :return: The aligned kernel matrix.
        :rtype: torch.Tensor
        """
        if self.x_train_ is None or self.y_train_ is None:
            raise ValueError("Training data must be provided before kernel alignment.")
        self.train()
        centerer = self._create_kernel_centerer()
        y_kernel = self._create_y_kernel().to(self.device)
        centered_y_kernel = torch.einsum("ij,jk,kl->il", centerer, y_kernel, centerer)

        self.opt_ = OptimizeResult()
        self.opt_.fun = np.inf
        self.opt_.history = []
        self.to(device=self.device)
        best_params = torch.nn.utils.parameters_to_vector(self.parameters()).cpu().detach().clone()
        best_alignment = -np.inf
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.alignment_learning_rate)
        p_bar = tqdm(np.arange(self.alignment_iterations), desc="Aligning kernel", disable=not verbose)
        for _ in p_bar:
            self.train()
            optimizer.zero_grad()
            kernel_matrix = self(self.x_train_, self.x_train_)
            centered_kernel = torch.einsum("ij,jk,kl->il", centerer, kernel_matrix, centerer)
            alignment = torch.einsum("ij,ij->", centered_kernel, centered_y_kernel.detach()) / (
                torch.norm(centered_kernel) * torch.norm(centered_y_kernel.detach()) + 1e-8
            )
            loss = -alignment
            self.opt_.fun = alignment.detach().cpu().item()
            self.opt_.history.append(self.opt_.fun)
            if np.isclose(self.opt_.fun, np.max(self.opt_.history)):
                best_params = torch.nn.utils.parameters_to_vector(self.parameters()).cpu().detach().clone()
                best_alignment = self.opt_.fun
            if all(
                [
                    len(self.opt_.history) > self.alignment_early_stopping_patience,
                    np.all(
                        np.abs(np.diff(self.opt_.history[-self.alignment_early_stopping_patience :]))
                        <= self.alignment_early_stopping_threshold
                    ),
                ]
            ):
                break
            loss.backward()
            optimizer.step()
            p_bar.set_postfix_str(f"Alignment score: {self.opt_.fun:.4f}")
        torch.nn.utils.vector_to_parameters(to_tensor(best_params, device=self.device), self.parameters())
        self.opt_.fun = best_alignment
        p_bar.close()
        self.eval()
        return self

    def _create_y_kernel(self) -> torch.Tensor:
        """
        Creates a kernel matrix for the target variable `y_train`.

        This method calculates the kernel matrix, which is used for measuring the similarity or
        correlation structure of the target variable `y_train`. It handles classification tasks by
        converting `y_train` to one-hot encoding when needed and supports binary or regression tasks.
        For multi-dimensional target data (e.g., multi-variate regression), the method computes
        the kernel via generalized Einstein summation.

        :raises ValueError: Raised if `y_train` contains invalid or unexpected values that cannot be processed.

        :return: A kernel matrix computed from the target variable `y_train`.
        :rtype: torch.Tensor
        """
        if not isinstance(self.y_train_, torch.Tensor):
            y_train = to_tensor(np.asarray(self.y_train_), device=self.device)
        else:
            y_train = to_tensor(self.y_train_, device=self.device)
        is_classification = all(
            [
                y_train.dim() == 1,  # single-dimensional
                torch.allclose(y_train.float(), y_train.long().float()),  # integer values
            ]
        )
        if is_classification:
            # For classification: convert to one-hot encoding
            n_classes = int(y_train.max().item()) + 1
            y_one_hot = F.one_hot(y_train.long(), num_classes=n_classes).float()
            y_kernel = torch.einsum("ki,bi->kb", y_one_hot, y_one_hot)
        else:  # Regression case
            y_kernel = torch.einsum("i...,k...->ik", y_train, y_train)
        return y_kernel

    def _create_kernel_centerer(self):
        """
        Creates and returns a kernel centerer matrix for training data.

        The kernel centerer is a matrix used to center the kernel matrix in kernel
        methods, which involves transforming the kernel matrix such that its mean in
        both rows and columns is zero. This is accomplished using the training data
        sample size.

        :return: A kernel centerer matrix for the training data.
        :rtype: torch.Tensor
        """
        n_samples = len(self.x_train_)
        centerer = (
            torch.eye(n_samples, device=self.device)
            - torch.ones((n_samples, n_samples), device=self.device) / n_samples
        )
        return centerer

    @property
    def device(self):
        """
        Provides the current device of the model parameters.

        :return: The device on which the model's parameters reside.
        :rtype: torch.device
        """
        return self.device_tracker_param.device
