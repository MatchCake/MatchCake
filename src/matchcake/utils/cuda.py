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
    except ImportError:  # pragma: no cover
        if throw_error:  # pragma: no cover
            raise ImportError("Pytorch not installed.")  # pragma: no cover
        if enable_warnings:  # pragma: no cover
            warnings.warn("Pytorch not installed.", ImportWarning)  # pragma: no cover
        use_cuda = False  # pragma: no cover
    return use_cuda
