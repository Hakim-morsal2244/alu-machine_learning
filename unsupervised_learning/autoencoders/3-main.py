#!/usr/bin/env python3

"""Main entry point for variational autoencoder tasks."""

import tensorflow.compat.v1 as tf

autoencoder = __import__('3-variational').autoencoder

tf.disable_v2_behavior()