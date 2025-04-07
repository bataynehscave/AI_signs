import numpy as np
import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_PATH = Path(r'MP_Data')

sequence_length = 45

# Actions that we try to detect
actions = np.array(['null','besm allah' , 'alsalam alekom' , 'alekom salam' , 'aslan w shlan' , 'me',
                    'age','alhamdulilah' , 'bad' , 'how are you' , 'friend' ,
                    'good' , 'happy' , 'you' , 'my name is' , 'no' , 
                    'or' , 'taaban' , 'what' , 'where' , 'yes' ,
                    'look' , 'said' , 'walking' , 'did not hear' , 'remind me',
                    'eat' , 'bayt' , 'hospital' , 'run' , 'sleep',
                    'think' , 'tomorrow' , 'yesterday' , 'today' , 'when',
                    'dhuhr' , 'sabah' , 'university' , 'kuliyah' ,'night',
                    'a3ooth bellah' , 'danger' , 'enough' , 'hot' , 'mosque' , 'surprise' , 'tard' , 
                    'big' , 'clean' , 'dirty' , 'fire' , 'give me' , 'sho dakhalak' , 'small' , 
                    'help' , 'same' , 'hour' , 'important' , 'ok' , 'please' , 'want' ,
                    'riyadah' , 'sallah' , 'telephone' , 'hamam' , 'water' , 'eid'
                   ])

label_map = {label:num for num, label in enumerate(actions)}

def normalize_landmarks(landmarks, epsilon=1e-6):
    num_points = len(landmarks) // 3
    landmarks = landmarks.reshape(num_points, 3)

    # Use wrist as anchor for hands and midpoint of hips for pose
    if num_points == 33:  # Pose
        anchor = landmarks[23]  # Left hip
        reference_dist = np.linalg.norm(landmarks[11] - landmarks[12])  # Shoulder width
    elif num_points == 21:  # Hands
        anchor = landmarks[0]  # Wrist
        reference_dist = np.linalg.norm(landmarks[5] - landmarks[17])  # Palm width
    else:
        return landmarks.flatten()  # Return unchanged for unexpected data

    # Handle potential zero or small reference distance
    if reference_dist < epsilon:
        return np.zeros_like(landmarks.flatten())  # Return zeroed-out array for invalid frames

    # Translate (center around anchor)
    landmarks -= anchor

    # Scale (normalize distances)
    landmarks /= reference_dist

    return landmarks.flatten()


sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))):
        try:
            sequence = sequence.astype(int)
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                normalized_res = np.concatenate([
                    normalize_landmarks(res[:33*4]),  # Pose landmarks
                    normalize_landmarks(res[33*4:33*4 + 21*3]),  # Left hand landmarks
                    normalize_landmarks(res[33*4 + 21*3:])  # Right hand landmarks
                ])
                window.append(normalized_res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Error in sequence {sequence}: {e}")


X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)



log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(45,258)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=30, callbacks=[tb_callback])


model.save('action.h5')