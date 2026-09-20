"""Stacked Bidirectional LSTM for next-day price forecasting.

Clean reference implementation of the *price model* from the CNN+LSTM trading bot.
It consumes a window of MinMax-scaled OHLCV features (flattened into a single
sequence, as in the original pipeline) and regresses the next day's scaled opening
price. Rewritten from the 2022 original for clarity and modern ``tf.keras``.

Run ``python price_lstm.py`` to print the model summary.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_price_lstm(
    sequence_len: int,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout: float = 0.0,
    optimizer: str = "adam",
) -> keras.Model:
    """Build and compile the Bidirectional-LSTM price regressor.

    Args:
        sequence_len: length of the flattened input sequence — in the original
            pipeline this is ``window_days * n_features`` (e.g. 27 trading days
            of OHLCV).
        lstm_units_1: units in the first (sequence-returning) BiLSTM layer.
        lstm_units_2: units in the second BiLSTM layer.
        dropout: dropout applied inside the LSTM cells.
        optimizer: keras optimizer name.

    Returns:
        A compiled regressor: linear output, ``mean_absolute_error`` loss.
    """
    model = keras.Sequential(name="price_lstm")
    model.add(keras.Input(shape=(sequence_len, 1)))
    model.add(layers.Bidirectional(
        layers.LSTM(lstm_units_1, dropout=dropout, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(lstm_units_2, dropout=dropout)))
    model.add(layers.Dense(1))  # regress the next-day (scaled) open
    model.compile(optimizer=optimizer, loss="mean_absolute_error")
    return model


if __name__ == "__main__":
    # 27-day window over 6 OHLCV columns, flattened -> length-162 sequence.
    build_price_lstm(sequence_len=27 * 6).summary()
