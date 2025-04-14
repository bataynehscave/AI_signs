import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
import math

# Load trained model
model = load_model('bilstm_gesture_model_final.h5')

# Configuration
SEQUENCE_LENGTH = 45
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

# Landmark indices used during training
selected_pose_indices = np.array([0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
pose_x = selected_pose_indices
pose_y = pose_x + 33
pose_z = pose_x + 2 * 33
pose_indices = np.concatenate([pose_x, pose_y, pose_z])  # (63,)

LEFT_HAND_START = 33 * 4
RIGHT_HAND_START = LEFT_HAND_START + 63

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Sequence for storing frames
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=10)

def preprocess_landmarks(results):
    try:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        pose = pose[selected_pose_indices]
    except:
        pose = np.zeros((21, 3))

    try:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    except:
        left_hand = np.zeros((21, 3))

    try:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    except:
        right_hand = np.zeros((21, 3))

    return np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])

def draw_prediction_bar(image, prediction, prob):
    bar_length = int(prob * 300)
    cv2.rectangle(image, (20, 50), (20 + bar_length, 90), (0, 255, 0), -1)
    cv2.putText(image, f'{prediction.upper()} - {prob*100:.1f}%', (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def draw_instruction_overlay(image):
    cv2.putText(image, 'Perform a gesture...', (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(image, (10, 10), (630, 470), (255, 255, 255), 2)

# =====================
# Start Video Capture
# =====================
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True

    # Draw instructions and prediction box
    draw_instruction_overlay(image)

    # Extract and preprocess
    keypoints = preprocess_landmarks(results)
    sequence.append(keypoints)

    # Only predict when sequence is ready
    if len(sequence) == SEQUENCE_LENGTH:
        input_seq = np.array(sequence).reshape( 1,SEQUENCE_LENGTH, 189)
        res = model.predict(input_seq, verbose=0)[0]
        pred_class = actions[np.argmax(res)]
        pred_prob = np.max(res)

        if pred_prob > 0.7:
            predictions.append(pred_class)

        # Voting from last 10 predictions
        if predictions:
            most_common = Counter(predictions).most_common(1)[0]
            draw_prediction_bar(image, most_common[0], pred_prob)

    # Show window
    cv2.imshow("Real-Time Gesture Recognition", image)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
