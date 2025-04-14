import cv2
import os
import numpy as np
from pathlib import Path
import mediapipe as mp
import time

# -------- CONFIGURATION --------
actions = ['hello', 'thanks', 'iloveyou']
sequence_length = 45  # Frames per sequence
num_sequences = 15  # Videos per action
data_path = Path('MP_Data')
start_folder = 1  # Start index for folder numbering
fps = 30  # Target camera FPS

# -------- SETUP MEDIAPIPE --------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------- FUNCTIONS --------
def create_folders():
    for action in actions:
        for sequence in range(start_folder, start_folder + num_sequences):
            dir_path = data_path / action / str(sequence)
            dir_path.mkdir(parents=True, exist_ok=True)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# -------- MAIN COLLECTION --------
def collect_data():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, fps)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:

            
            for sequence in range(start_folder, start_folder + num_sequences):
                print(f'\nCollecting for {action}, video {sequence}')
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (160, 160))
                    if not ret:
                        print("Failed to read frame from camera.")
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    image = cv2.resize(image, (640*2, 480*2))
                    draw_styled_landmarks(image, results)

                    # Countdown on first frame
                    if frame_num == 0:
                        if sequence == start_folder:
                            cv2.putText(image, f'STARTING COLLECTION FOR ACTION {action}', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(3500)
                        else:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                            cv2.putText(image, f'Action: {action}, Video: {sequence}', (15, 12), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(3500)
                    else:
                        cv2.putText(image, f'Action: {action}, Video: {sequence}', (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = data_path / action / str(sequence) / f"{frame_num}.npy"
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

# -------- RUN --------
if __name__ == "__main__":
    create_folders()
    collect_data()
