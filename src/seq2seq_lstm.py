from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

def build_seq2seq(n_features, seq_len, horizon):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, n_features)))
    model.add(RepeatVector(horizon))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer="adam", loss="mse")
    return model
