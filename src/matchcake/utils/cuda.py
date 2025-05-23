import warnings


def is_cuda_available(enable_warnings: bool = False, throw_error: bool = False) -> bool:
    try:
        import torch

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            if throw_error:
                raise RuntimeError("Cuda not available.")
            if enable_warnings:
                warnings.warn("Cuda not available.", ImportWarning)
    except ImportError:
        if throw_error:
            raise ImportError("Pytorch not installed.")
        if enable_warnings:
            warnings.warn("Pytorch not installed.", ImportWarning)
        use_cuda = False
    return use_cuda
