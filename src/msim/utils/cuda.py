import warnings


def is_cuda_available(enable_warnings: bool = False) -> bool:
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Cuda not available.", ImportWarning)
    except ImportError:
        warnings.warn("Pytorch not installed.", ImportWarning)
        use_cuda = False
    return use_cuda



