#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Builds a convolutional autoencoder
    """

    # ===== Encoder =====
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    encoder = keras.Model(inputs, x)

    # ===== Decoder =====
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    rev_filters = list(reversed(filters))

    # First layers of decoder (all except last two convs)
    for f in rev_filters[:-2]:
        x = keras.layers.Conv2D(
            f, (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last conv (VALID padding)
    x = keras.layers.Conv2D(
        rev_filters[-2],
        (3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Last conv (NO upsampling)
    x = keras.layers.Conv2D(
        input_dims[-1],
        (3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, x)

    # ===== Autoencoder =====
    auto_input = inputs
    auto_output = decoder(encoder(auto_input))

    auto = keras.Model(auto_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
