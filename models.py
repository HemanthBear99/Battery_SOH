import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Layer, GRU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class Attention(Layer):
    """
    Custom Attention Layer for weighted feature extraction across time steps.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

def build_attention_cnn_bilstm(input_shape):
    """
    Builds the primary proposed custom model: Conv1D -> BiLSTM -> Attention -> Dense.
    """
    inp = Input(shape=input_shape)
    x = Conv1D(32, 1, activation='relu')(inp)
    x = MaxPooling1D(1)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Attention()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_dnn_model(input_shape):
    """
    Standard Deep Neural Network baseline. Input should be flattened.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_gru_model(input_shape):
    """
    GRU baseline model. Input shape should be (1, Features).
    """
    model = Sequential([
        GRU(32, return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def build_lstm_model(input_shape):
    """
    Standard LSTM baseline model. Input shape should be (1, Features).
    """
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model
