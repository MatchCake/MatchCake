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
    
    as_arr = qml.math.array(tensor)
    tensor_shape = qml.math.shape(as_arr)
    batched_tensor = qml.math.reshape(as_arr, (-1, *tensor_shape[-2:]))
    stacked_tensor = qml.math.stack([scipy_logm(m) for m in batched_tensor], axis=0)
    return qml.math.reshape(stacked_tensor, tensor_shape)
