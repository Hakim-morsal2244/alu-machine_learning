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
        """Initialize an NST instance."""
        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescale an image so its largest side is 512 pixels."""
        if not isinstance(image, np.ndarray) or \
                len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        height, width = image.shape[:2]

        if height >= width:
            new_height = 512
            new_width = int(width * 512 / height)
        else:
            new_width = 512
            new_height = int(height * 512 / width)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        image = tf.image.resize_images(
            image,
            (new_height, new_width),
            method=tf.image.ResizeMethod.BICUBIC
        )

        image = tf.clip_by_value(image, 0.0, 255.0)
        image = image / 255.0

        return image

    def load_model(self):
        """Create the VGG19 model used to calculate costs."""
        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg19.save('VGG19_base_model')

        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }

        vgg = tf.keras.models.load_model(
            'VGG19_base_model',
            custom_objects=custom_objects
        )

        outputs = []

        for layer in vgg.layers:
            layer.trainable = False

            if layer.name in self.style_layers:
                outputs.append(layer.output)

            if layer.name == self.content_layer:
                outputs.append(layer.output)

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a layer output."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError('input_layer must be a tensor of rank 4')

        if len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        _, height, width, channels = input_layer.shape
        size = int(height * width)

        features = tf.reshape(
            input_layer,
            (size, channels)
        )

        gram = tf.matmul(
            features,
            features,
            transpose_a=True
        )

        gram = tf.expand_dims(gram, axis=0)

        gram = gram / tf.cast(size, tf.float32)

        return gram

    def generate_features(self):
        """Extract the style and content features."""
        vgg19 = tf.keras.applications.vgg19

        style_input = vgg19.preprocess_input(
            self.style_image * 255
        )

        content_input = vgg19.preprocess_input(
            self.content_image * 255
        )

        features_style = self.model(style_input)
        features_content = self.model(content_input)

        style_outputs = features_style[:-1]
        content_output = features_content[-1]

        self.gram_style_features = []

        for style_output in style_outputs:
            self.gram_style_features.append(
                self.gram_matrix(style_output)
            )

        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single layer."""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError(
                'style_output must be a tensor of rank 4'
            )

        _, height, width, channels = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                len(gram_target.shape) != 3 or \
                gram_target.shape != (1, channels, channels):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'
                .format(channels, channels)
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """Calculate the style cost for a generated image."""
        length = len(self.style_layers)

        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                'style_outputs must be a list with a length of {}'
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
