# Code style

## Files
- No file-level docstrings. Move any documentation into the class docstring instead.
- For files with only module-level functions, omit the module docstring entirely.

## Comments
- No separator comments between methods or functions (e.g. no `# --- Public interface ---`).
- Inline comments inside function bodies are fine when they clarify non-obvious things (e.g. tensor shape annotations).
- Use em dashes (—) sparingly in docstrings and comments. Prefer plain sentence structure over dashes.

## Type hints and docstrings
- All public functions and methods must have full type annotations on every parameter and the return type.
- Docstrings use Sphinx style: `:param name: description`, `:return: description`, `:rtype: type`.

## Tensors and backends
- Any function that accepts a `TensorLike` argument must return the same backend and dtype as its primary input, using `convert_and_cast_like(result, reference_input)` before returning.
- Tensor functions must support arbitrary leading batch dimensions (expressed as `...` in shape comments, e.g. `(..., 2n, 2n)`).
- Hardware-specific branches (CUDA, platform-only code) that cannot run in CI must be marked `# pragma: no cover`.

## Testing
- Every new class or module-level function must have an associated test class added in the same PR/commit.
- Tests must achieve >98% code coverage for the new code. Numerical comparisons use tolerance constants from `tests/configs.py` (`ATOL_SCALAR_COMPARISON`, `ATOL_MATRIX_COMPARISON`, etc.) — never hardcode tolerance values.
- The `tests/` folder must mirror the package structure (replacing `src/matchcake/` with `tests/` and prefixing each path segment with `test_`). For example, `src/matchcake/utils/jordan_wigner.py` → `tests/test_utils/test_jordan_wigner.py`, and `src/matchcake/devices/expval_strategies/m_pfaffian/m_pfaffian_expval_strategy.py` → `tests/test_devices/test_expval_strategies/test_m_pfaffian/test_m_pfaffian_expval_strategy.py`.

## Class method ordering
Methods within a class must appear in this order:
1. `@staticmethod`
2. `@classmethod`
3. Constructor (`__init__`)
4. Operator / dunder methods (`__call__`, `__repr__`, …)
5. Public methods
6. Protected methods (`_method`)
7. Private methods (`__method`)
8. Properties (`@property`)
