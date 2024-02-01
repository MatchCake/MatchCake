import pennylane as qml


def cast_to_complex(__inputs):
    r"""

    Cast the inputs to complex numbers.

    :param __inputs: Inputs to cast
    :return: Inputs casted to complex numbers
    """
    return type(__inputs)(qml.math.asarray(__inputs).astype(complex))


@qml.math.multi_dispatch()
def logm(tensor, like=None):
    """Compute the matrix exponential of an array :math:`\\ln{X}`.

    ..note::
        This function is not differentiable with Autograd, as it
        relies on the scipy implementation.
    """
    if like == "torch":
        return tensor.matrix_log()
    if like == "jax":
        from jax.scipy.linalg import logm as jax_logm
        
        return jax_logm(tensor)
    if like == "tensorflow":
        import tensorflow as tf
        
        return tf.linalg.logm(tensor)
    from scipy.linalg import logm as scipy_logm
    
    as_arr = qml.math.array(tensor, dtype=complex)
    tensor_shape = qml.math.shape(as_arr)
    batched_tensor = qml.math.reshape(as_arr, (-1, *tensor_shape[-2:]))
    stacked_tensor = qml.math.stack([scipy_logm(m) for m in batched_tensor], axis=0)
    return qml.math.reshape(stacked_tensor, tensor_shape)


def shape(tensor):
    """Get the shape of a tensor.

    :param tensor: The tensor.
    :type tensor: Any
    :return: The shape of the tensor.
    :rtype: tuple
    """
    _shape = []
    if isinstance(tensor, (list, tuple)) and len(tensor) > 0:
        _shape.append(len(tensor))
        _shape.extend(qml.math.shape(tensor[0]))
    else:
        _shape.extend(qml.math.shape(tensor))
    return tuple(_shape)


def convert_and_cast_like(tensor1, tensor2):
    return qml.math.cast_like(qml.math.convert_like(tensor1, tensor2), tensor2)


def astensor(tensor, like=None, **kwargs):
    """Convert input to a tensor.

    :param tensor: Input tensor.
    :type tensor: Any
    :param like: The desired tensor framework to use.
    :type like: str, optional
    :param kwargs: Additional keyword arguments that are passed to the tensor framework.
    :return: The tensor.
    :rtype: Any
    """
    from ..templates.tensor_like import TensorLike
    if isinstance(tensor, TensorLike):
        return tensor
    return qml.math.array(tensor, like=like, **kwargs)
