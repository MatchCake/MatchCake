import pytest

from matchcake.utils.cuda import is_cuda_available

from ..configs import TEST_SEED, set_seed


class TestIsCudaAvailable:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_returns_bool(self):
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_default_no_exception(self):
        is_cuda_available()

    def test_enable_warnings_when_unavailable(self):
        if is_cuda_available():
            pytest.skip("CUDA is available; cannot test the warning path.")
        with pytest.warns(ImportWarning):
            is_cuda_available(enable_warnings=True)

    def test_throw_error_when_unavailable(self):
        if is_cuda_available():
            pytest.skip("CUDA is available; cannot test the error path.")
        with pytest.raises(RuntimeError):
            is_cuda_available(throw_error=True)
