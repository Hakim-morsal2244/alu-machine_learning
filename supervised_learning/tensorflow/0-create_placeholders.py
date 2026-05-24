import tensorflow as tf  # type: ignore


def create_placeholders(nx, classes):
    """Create TF placeholders for input data and one-hot labels.
    nx: number of features
    classes: number of classes
    Returns: x, y placeholders
    """
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
