from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

def create_model(classes=6):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(45,258)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(256, return_sequences=True, activation='tanh'))
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model