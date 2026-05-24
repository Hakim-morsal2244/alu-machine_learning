#!/usr/bin/env python3
"""
Creates a vanilla autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims: int, input size
    hidden_layers: list of encoder layer sizes
    latent_dims: int, latent space size

    Returns: encoder, decoder, auto
    """

    # Input layer
    input_layer = keras.Input(shape=(input_dims,))

    # -------- Encoder --------
    encoded = input_layer

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    encoder = keras.Model(inputs=input_layer, outputs=latent)

    # -------- Decoder --------
    decoder_input = keras.Input(shape=(latent_dims,))

    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(inputs=decoder_input, outputs=output)

    # -------- Autoencoder --------
    auto_input = input_layer
    encoded_output = encoder(auto_input)
    reconstructed = decoder(encoded_output)

    auto = keras.Model(inputs=auto_input, outputs=reconstructed)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
