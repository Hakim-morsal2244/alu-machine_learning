#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """

    # ======================
    # ENCODER
    # ======================
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    mu = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sample(args):
        """
        Samples from normal distribution
        """
        mu, log_var = args

        epsilon = K.random_normal(
            shape=K.shape(mu)
        )

        return mu + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sample
    )([mu, log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, mu, log_var]
    )

    # ======================
    # DECODER
    # ======================
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # ======================
    # AUTOENCODER
    # ======================
    outputs = decoder(z)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        outputs
    )

    reconstruction_loss *= input_dims

    kl_loss = 1 + log_var - K.square(mu) - K.exp(log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(
        reconstruction_loss + kl_loss
    )

    auto.add_loss(vae_loss)

    auto.compile(
        optimizer='adam'
    )

    return encoder, decoder, auto
