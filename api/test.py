import cv2
import mediapipe as mp
import numpy as np
import os
import traceback
from time import time
import tensorflow as tf




mp_drawing = mp.solutions.drawing_utils 
def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                             ) 




# Replace with actual model and actions list
# Load your trained model (Keras in this example)
model = tf.keras.models.load_model("action_normed.h5")

# Define the actions list (update these with your actual sign language labels)
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
threshold = 0.5

# Initialize Mediapipe holistic module
mp_holistic = mp.solutions.holistic

# ---------------------------
# Helper Functions
# ---------------------------
def mediapipe_detection(frame, holistic):
    """Convert frame color, run Mediapipe detection, and convert back."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def normalize_landmarks(landmarks, epsilon=1e-6):
    num_points = len(landmarks) // 3
    landmarks = landmarks.reshape(num_points, 3)
    if num_points == 33:  # Pose landmarks
        anchor = landmarks[23]  # Left hip
        reference_dist = np.linalg.norm(landmarks[11] - landmarks[12])
    elif num_points == 21:  # Hand landmarks
        anchor = landmarks[0]  # Wrist
        reference_dist = np.linalg.norm(landmarks[5] - landmarks[17])
    else:
        return landmarks.flatten()
    if reference_dist < epsilon:
        return np.zeros_like(landmarks.flatten())
    landmarks -= anchor
    landmarks /= reference_dist
    return landmarks.flatten()

def extract_keypoints(results):
    """
    Extract and normalize keypoints from Mediapipe detection results.
    This example extracts pose and both hands.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    normalized_pose = normalize_landmarks(pose.reshape(-1, 3).flatten())
    normalized_lh = normalize_landmarks(lh)
    normalized_rh = normalize_landmarks(rh)
    return np.concatenate([normalized_pose, normalized_lh, normalized_rh])


def process_video(video_path):
    print('started visualization')
    try:
        # 1) Open video and get FPS
        cap = cv2.VideoCapture(video_path)
        print(f"Video opened: {cap.isOpened()}")
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # 2) Define virtual FPS for 45 frames every 1.5s
        desired_frames = 45
        block_seconds = 1.5
        virtual_fps = desired_frames / block_seconds  # 30.0
        sample_ratio = virtual_fps / original_fps
        accum = 0.0

        keypoints_buffer = []
        frame_buffer = []

        # 3) Extract keypoints and frames
        with mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while True:
                ret, frame = cap.read()
                print(frame)
                if not ret:
                    print('ha')
                    break
                # frame = cv2.flip(frame, 1)
                # frame = cv2.resize(frame, (320*2, 240*2))
                accum += sample_ratio
                num_virtual = int(accum)
                accum -= num_virtual

                for _ in range(num_virtual):
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    keypoints_buffer.append(keypoints)
                    frame_buffer.append(image.copy())

        cap.release()

        # 4) Sliding window predictions every 45 frames
        predictions_list = []
        sentence = []
        frame_labels = [""] * len(frame_buffer)

        for start in range(0, len(keypoints_buffer) - desired_frames + 1, desired_frames):
            window = keypoints_buffer[start:start + desired_frames]
            if len(window) == desired_frames:
                res = model.predict(np.expand_dims(window, axis=0))[0]
                idx = np.argmax(res)
                action = actions[idx]
                predictions_list.append(action)

                # Label all 45 frames in the window
                if res[idx] > threshold:
                    for i in range(start, start + desired_frames):
                        if i < len(frame_labels):
                            frame_labels[i] = action

                # Sentence logic with confidence + consistency
                if (len(predictions_list) >= 3 and
                    predictions_list[-1] == predictions_list[-2] == predictions_list[-3]):
                    if not sentence or sentence[-1] != action:
                        sentence.append(action)
                    sentence = sentence[-5:]

        # 5) Display the video with predictions
        for i, frame in enumerate(frame_buffer):
            label = frame_labels[i]
            if label:
                cv2.putText(frame, f'Prediction: {label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Prediction Playback', frame)
            # Slowed down playback (150 ms between frames = ~6.7 FPS)
            if cv2.waitKey(150) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        return {
            "predictions": predictions_list,
            "sentence": " ".join(sentence)
        }

    except Exception as ex:
        traceback.print_exc()
        return {"error": "Server error", "details": str(ex)}


# Example usage:
video_path = r"bis.mp4"
result = process_video(video_path)
print(result)
