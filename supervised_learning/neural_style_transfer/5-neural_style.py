#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor for Neural Style Transfer."""
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image to 512 pixels on its largest side."""
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0),
            size=(h_new, w_new)
        )
        resized = tf.clip_by_value(resized, 0, 255)
        rescaled = resized / 255

        return rescaled

    def load_model(self):
        """Creates the VGG19 model used to calculate costs."""
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        VGG19_model.save("VGG19_base_model")

        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }

        vgg = tf.keras.models.load_model(
            "VGG19_base_model",
            custom_objects=custom_objects
        )

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        product = int(h * w)

        features = tf.reshape(input_layer, (product, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram = gram / tf.cast(product, tf.float32)

        return gram

    def generate_features(self):
        """Extracts the features used to calculate neural style cost."""
        vgg19 = tf.keras.applications.vgg19

        preprocess_style = vgg19.preprocess_input(
            self.style_image * 255
        )
        preprocess_content = vgg19.preprocess_input(
            self.content_image * 255
        )

        style_features = self.model(preprocess_style)[:-1]
        self.content_feature = self.model(preprocess_content)[-1]

        self.gram_style_features = [
            self.gram_matrix(feature) for feature in style_features
        ]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer."""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        _, h, w, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                len(gram_target.shape) != 3 or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(c, c)
            )

        gram_style = self.gram_matrix(style_output)
        diff = tf.reduce_mean(tf.square(gram_style - gram_target))

        return diff

    def style_cost(self, style_outputs):
        """Calculates the style cost for a generated image."""
        length = len(self.style_layers)

        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(length)
            )

        weight = 1 / length
        style_cost = 0

        for i in range(length):
            style_cost += (
                self.layer_style_cost(
                    style_outputs[i],
                    self.gram_style_features[i]
                ) * weight
            )

        return style_cost
