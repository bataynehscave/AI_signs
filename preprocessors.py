import numpy as np


def normalize_landmarks(landmarks, epsilon=1e-6, pose=False):
    num_points = len(landmarks) // 3
    landmarks = landmarks.reshape(num_points, 3)

    if pose:
        if num_points <= 12:
            return np.zeros_like(landmarks.flatten())
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        anchor = (left_shoulder + right_shoulder) / 2
        reference_dist = np.linalg.norm(left_shoulder - right_shoulder)
    else:
        anchor = landmarks[0]  # wrist
        reference_dist = np.linalg.norm(landmarks[5] - landmarks[17])  # palm width

    if reference_dist < epsilon:
        return np.zeros_like(landmarks.flatten())

    normalized = (landmarks - anchor) / reference_dist
    return normalized.flatten()