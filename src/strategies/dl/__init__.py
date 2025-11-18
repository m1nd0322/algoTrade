"""
Deep learning based trading strategies.
"""

from .cnn_candlestick import CNNCandlestickStrategy
from .lstm_direction import LSTMDirectionStrategy
from .gru import GRUStrategy
from .autoencoder import AutoencoderStrategy
from .transformer import TransformerStrategy

__all__ = [
    'CNNCandlestickStrategy',
    'LSTMDirectionStrategy',
    'GRUStrategy',
    'AutoencoderStrategy',
    'TransformerStrategy'
]
