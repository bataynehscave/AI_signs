import numpy as np
import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==============================
# Configuration
# ==============================
DATA_PATH = Path('MP_Data')
SEQUENCE_LENGTH = 45

# List of actions
actions = np.array([
    'null', 'besm allah', 'alsalam alekom', 'alekom salam', 'aslan w shlan', 'me',
    'age', 'alhamdulilah', 'bad', 'how are you', 'friend', 'good', 'happy',
    'you', 'my name is', 'no', 'or', 'taaban', 'what', 'where', 'yes', 'look',
    'said', 'walking', 'did not hear', 'remind me', 'eat', 'bayt', 'hospital',
    'run', 'sleep', 'think', 'tomorrow', 'yesterday', 'today', 'when', 'dhuhr',
    'sabah', 'university', 'kuliyah', 'night', 'a3ooth bellah', 'danger', 'enough',
    'hot', 'mosque', 'surprise', 'tard', 'big', 'clean', 'dirty', 'fire',
    'give me', 'sho dakhalak', 'small', 'help', 'same', 'hour', 'important',
    'ok', 'please', 'want', 'riyadah', 'sallah', 'telephone', 'hamam', 'water', 'eid'
])
label_map = {label: idx for idx, label in enumerate(actions)}

# ==============================
# Landmark indices
# ==============================
selected_pose_indices = np.array([0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
pose_x = selected_pose_indices
pose_y = pose_x + 33
pose_z = pose_x + 2 * 33
pose_indices = np.concatenate([pose_x, pose_y, pose_z])  # (63,)

LEFT_HAND_START = 33 * 4
RIGHT_HAND_START = LEFT_HAND_START + 63
frame_shape = (189,)

# ==============================
# Sequence Loading Function
# ==============================
def load_sequence(action_path, seq_idx, seq_len, fallback_shape):
    sequence = []
    last_valid = np.zeros(fallback_shape)

    for frame_num in range(seq_len):
        try:
            res = np.load(os.path.join(action_path, str(seq_idx), f"{frame_num}.npy"))
            pose = res[pose_indices]
            left = res[LEFT_HAND_START:LEFT_HAND_START + 63]
            right = res[RIGHT_HAND_START:RIGHT_HAND_START + 63]
            if left.shape[0] != 63:
                left = np.zeros(63)
            if right.shape[0] != 63:
                right = np.zeros(63)
            frame = np.concatenate([pose, left, right])
            last_valid = frame
        except Exception:
            frame = last_valid
        sequence.append(frame)
    return sequence

# ==============================
# Load All Sequences
# ==============================
sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    for sequence_dir in os.listdir(action_path):
        try:
            seq_idx = int(sequence_dir)
            seq = load_sequence(action_path, seq_idx, SEQUENCE_LENGTH, fallback_shape=frame_shape)
            sequences.append(np.array(seq))
            labels.append(label_map[action])
        except Exception as e:
            print(f"Skipping {action}/{sequence_dir} due to error: {e}")

# ==============================
# Prepare for Training
# ==============================
X = np.array(sequences)
X = X.reshape(X.shape[0], SEQUENCE_LENGTH, frame_shape[0])  # Shape: (samples, timesteps, features)

y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15)

# ==============================
# Callbacks
# ==============================
log_dir = os.path.join('Logs')
callbacks = [
    TensorBoard(log_dir=log_dir),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint('best_bilstm_heavy_model.h5', monitor='val_loss', save_best_only=True)
]

# ==============================
# Build BiLSTM Model
# ==============================
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=(SEQUENCE_LENGTH, frame_shape[0])),
    Dropout(0.3),
    Bidirectional(LSTM(256, return_sequences=True, activation='tanh')),
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
    Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
    Bidirectional(LSTM(32, return_sequences=False, activation='tanh')),
    Dense(128, activation='tanh'),
    Dropout(0.4),
    Dense(64, activation='tanh'),
    Dropout(0.4),
    Dense(32, activation='tanh'),
    Dense(actions.shape[0], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==============================
# Train Model
# ==============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

model.save('bilstm_heavy_gesture_model_final.h5')
