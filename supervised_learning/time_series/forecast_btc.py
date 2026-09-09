#!/usr/bin/env python3
"""Forecast the next hour of Bitcoin prices using an RNN."""

import os

import numpy as np
import pandas as pd
import tensorflow as tf


def create_dataset(data, window=24, batch_size=32):
    """Create a TensorFlow dataset using sliding windows."""
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data[:-window],
        targets=data[window:],
        sequence_length=window,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size
    )
    return dataset


def build_model(input_shape):
    """Build and compile the RNN forecasting model."""
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(
            64,
            activation="tanh",
            input_shape=input_shape
        ),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def main():
    """Load data, train the model, and save the results."""
    data = pd.read_csv("preprocessed.csv")

    values = data["coinbase"].values.astype(np.float32)

    split = int(len(values) * 0.8)

    train_values = values[:split]
    test_values = values[split - 24:]

    mean = train_values.mean()
    std = train_values.std()

    train_values = (train_values - mean) / std
    test_values = (test_values - mean) / std

    train_values = train_values.reshape(-1, 1)
    test_values = test_values.reshape(-1, 1)

    train_dataset = create_dataset(
        train_values,
        window=24,
        batch_size=32
    )

    test_dataset = create_dataset(
        test_values,
        window=24,
        batch_size=32
    )

    model = build_model((24, 1))

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=20
    )

    os.makedirs("results", exist_ok=True)

    model.save("results/btc_forecasting_model.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        "results/training_history.csv",
        index=False
    )

    print("Model training complete.")
    print("Model saved to results/btc_forecasting_model.h5")


if __name__ == "__main__":
    main()