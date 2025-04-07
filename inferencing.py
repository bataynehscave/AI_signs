import numpy as np
import cv2
import mediapipe as mp
from model import create_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def normalize_landmarks(landmarks, epsilon=1e-6):
    num_points = len(landmarks) // 3
    landmarks = landmarks.reshape(num_points, 3)

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
        return np.zeros_like(landmarks.flatten())

    landmarks -= anchor
    landmarks /= reference_dist

    return landmarks.flatten()

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Normalize keypoints
    normalized_pose = normalize_landmarks(pose[:33*4])
    normalized_lh = normalize_landmarks(lh)
    normalized_rh = normalize_landmarks(rh)
    
    return np.concatenate([normalized_pose, normalized_lh, normalized_rh])

colors = [(245,117,16), (117,245,16), (16,117,245),(16,137,205), (16,17,245), (122,147,45), (156,117,240)]*10
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame



# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5
model = create_model(actions.shape[0])
model.load_weights('action.h5')


cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-45:]

        if len(sequence) == 45:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()