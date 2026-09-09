#!/usr/bin/env python3
"""Preprocess Bitcoin data for time series forecasting."""

import pandas as pd


def preprocess_data():
    """Load, clean, resample, and save Bitcoin data."""
    coinbase = pd.read_csv("coinbase.csv")
    bitstamp = pd.read_csv("bitstamp.csv")

    coinbase["Timestamp"] = pd.to_datetime(
        coinbase["Timestamp"], unit="s"
    )
    bitstamp["Timestamp"] = pd.to_datetime(
        bitstamp["Timestamp"], unit="s"
    )

    coinbase = coinbase.set_index("Timestamp")
    bitstamp = bitstamp.set_index("Timestamp")

    coinbase["Close"] = pd.to_numeric(
        coinbase["Close"], errors="coerce"
    )
    bitstamp["Close"] = pd.to_numeric(
        bitstamp["Close"], errors="coerce"
    )

    coinbase = coinbase["Close"].resample("1h").mean()
    bitstamp = bitstamp["Close"].resample("1h").mean()

    data = pd.DataFrame({
        "coinbase": coinbase,
        "bitstamp": bitstamp
    })

    data = data.ffill().bfill()
    data = data.dropna()

    data.to_csv("preprocessed.csv")


if __name__ == "__main__":
    preprocess_data()