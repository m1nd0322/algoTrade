import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_data(path, window=30):
    df = pd.read_csv(path)
    close = df['Close'].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(len(close_scaled) - window):
        X.append(close_scaled[i:i + window])
        y.append(close_scaled[i + window])
    return np.array(X), np.array(y), scaler


def split_data(X, y, ratio=0.8):
    split = int(len(X) * ratio)
    return X[:split], y[:split], X[split:], y[split:]


def build_model(window, d_model=16, num_heads=2):
    inputs = layers.Input(shape=(window, 1))
    positions = tf.range(start=0, limit=window, delta=1)
    pos_embed = layers.Embedding(input_dim=window, output_dim=d_model)(positions)
    x = layers.Dense(d_model)(inputs)
    x = x + pos_embed
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=0.1
    )(x, x, use_causal_mask=True)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def main(args):
    X, y, scaler = load_data('data/us_etf_data/AAPL.csv', window=args.window)
    X_train, y_train, X_test, y_test = split_data(X, y)
    model = build_model(args.window)
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=32,
              validation_split=0.1, verbose=2)
    preds = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds)
    for i, p in enumerate(preds[:5]):
        print(f"Predicted close {i}: {p[0]:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)
