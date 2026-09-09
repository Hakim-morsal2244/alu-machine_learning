# Time Series Forecasting

This project uses Recurrent Neural Networks (RNNs) and TensorFlow
to perform time series forecasting.

## Learning Objectives

- Understand time series forecasting.
- Understand stationary processes.
- Understand sliding windows.
- Preprocess time series data.
- Create TensorFlow data pipelines.
- Perform time series forecasting with RNNs.

## Task 0: When to Invest

The goal is to predict the Bitcoin closing price for the following
hour using the previous 24 hours of Bitcoin data.

The original datasets contain Bitcoin data recorded in 60-second
windows from Coinbase and Bitstamp.

## Preprocessing

The `preprocess_data.py` script:

- Loads the Coinbase and Bitstamp datasets.
- Converts timestamps to datetime values.
- Uses the Bitcoin closing price.
- Resamples the 60-second data into hourly data.
- Handles missing values.
- Saves the processed data to `preprocessed.csv`.

## Model

The `forecast_btc.py` script creates a TensorFlow `tf.data.Dataset`
using a sliding window of 24 hours.

The model contains:

1. A SimpleRNN layer with 64 units.
2. A Dense layer with 32 units.
3. A final Dense layer that predicts the next closing price.

The model uses the Adam optimizer and Mean Squared Error (MSE)
as the loss function.

## Files

- `README.md` - Project documentation.
- `preprocess_data.py` - Preprocesses the Bitcoin datasets.
- `forecast_btc.py` - Trains the RNN forecasting model.
- `preprocessed.csv` - Processed hourly Bitcoin data.
- `results/` - Contains the trained model and training history.

## Requirements

- Python 3
- NumPy
- pandas
- TensorFlow

## Usage

First, place `coinbase.csv` and `bitstamp.csv` in this directory.

Run the preprocessing script:

```bash
./preprocess_data.py