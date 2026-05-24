#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """

    # ======================
    # Encoder
    # ======================
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var])

    # ======================
    # Decoder
    # ======================
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # ======================
    # Autoencoder
    # ======================
    auto_inputs = inputs
    z_enc, mu_enc, log_var_enc = encoder(auto_inputs)

    reconstructed = decoder(z_enc)

    auto = keras.Model(auto_inputs, reconstructed)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
