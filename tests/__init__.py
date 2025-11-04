import numpy as np
import torch

from .configs import TEST_SEED

np.random.seed(TEST_SEED)
torch.manual_seed(TEST_SEED)

