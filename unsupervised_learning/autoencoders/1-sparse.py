#!/usr/bin/env python3
"""
Sparse Autoencoder module
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """

    # Input layer
    input_layer = keras.Input(shape=(input_dims,))

    # ===== Encoder =====
    encoder = input_layer
    for nodes in hidden_layers:
        encoder = keras.layers.Dense(nodes, activation='relu')(encoder)

    # Latent layer with L1 regularization (SPARSITY)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoder)

    # Encoder model
    encoder_model = keras.Model(inputs=input_layer, outputs=latent)

    # ===== Decoder =====
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder = decoder_input

    for nodes in reversed(hidden_layers):
        decoder = keras.layers.Dense(nodes, activation='relu')(decoder)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)

    # Decoder model
    decoder_model = keras.Model(inputs=decoder_input, outputs=output)

    # ===== Autoencoder =====
    auto_input = input_layer
    auto_output = decoder_model(encoder_model(auto_input))

    auto_model = keras.Model(inputs=auto_input, outputs=auto_output)

    auto_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto_model
