import tensorflow as tf
from tensorflow.keras.layers import Layer


class AngleComparison(Layer):
    def __init__(self, filters=32, kernel_size=3):
        super(AngleComparison, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same"
        )
        super(AngleComparison, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        dot = tf.matmul(x1, x2, transpose_b=True)
        norm1 = tf.norm(x1, ord=2, axis=-1, keepdims=True)
        norm2 = tf.norm(x2, ord=2, axis=-1, keepdims=True)
        norm = tf.multiply(norm1, norm2)
        angle = dot / norm
        output = tf.concat([angle, x], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1)
