#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Builds a convolutional autoencoder
    """

    # ======================
    # ENCODER
    # ======================
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            f,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.MaxPooling2D(
            (2, 2)
        )(x)

    encoder = keras.Model(inputs, x)

    # ======================
    # DECODER
    # ======================
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    rev_filters = list(reversed(filters))

    x = keras.layers.Conv2D(
        rev_filters[0],
        (3, 3),
        padding='same',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        (2, 2)
    )(x)

    x = keras.layers.Conv2D(
        rev_filters[1],
        (3, 3),
        padding='same',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        (2, 2)
    )(x)

    x = keras.layers.Conv2D(
        rev_filters[2],
        (3, 3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        (2, 2)
    )(x)

    x = keras.layers.Conv2D(
        input_dims[2],
        (3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, x)

    # ======================
    # AUTOENCODER
    # ======================
    auto_input = inputs
    auto_output = decoder(encoder(auto_input))

    auto = keras.Model(auto_input, auto_output)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
