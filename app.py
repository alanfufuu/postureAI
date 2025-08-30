import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from playsound import playsound
import os
import math
import joblib

model = tf.keras.models.load_model("posture_classifier.h5")
scaler = joblib.load("scaler.pkl")

class_names = ["Good", "Poor"]

def compute_features(row):
    nose_x, nose_y, ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y = row
    shoulder_width = abs(ls_x - rs_x) + 1e-6
    shoulder_slope = (ls_y - rs_y) / shoulder_width
    le_dist = np.sqrt((le_x - ls_x) ** 2 + (le_y - ls_y) ** 2) / shoulder_width
    re_dist = np.sqrt((re_x - rs_x) ** 2 + (re_y - rs_y) ** 2) / shoulder_width
    ear_line_slope = (le_y - re_y) / (le_x - re_x + 1e-6)
    shoulder_line_slope = (ls_y - rs_y) / (ls_x - rs_x + 1e-6)
    neck_tilt = math.atan(abs(ear_line_slope - shoulder_line_slope))
    coords = np.array([
        nose_x, nose_y, ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y
    ]) / shoulder_width
    return np.concatenate([coords, [shoulder_slope, le_dist, re_dist, neck_tilt]])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # change index if wrong camera
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

script_dir = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(script_dir, "bell.wav")

bad_posture_start_time = None
notification_interval = 15  # ping every x seconds

print("Starting live posture detection...")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    slouch_status = "Unknown"
    shoulder_status = "Unknown"
    is_bad_posture = False

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            raw_features = np.array(nose + l_shoulder + r_shoulder + l_ear + r_ear)
            engineered = compute_features(raw_features)
            engineered_scaled = scaler.transform([engineered])

            prediction = model.predict(engineered_scaled, verbose=0)
            slouch_status = class_names[np.argmax(prediction)]

            left_y, right_y = l_shoulder[1], r_shoulder[1]
            shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            threshold = shoulder_width * 0.08
            shoulder_status = "Uneven" if abs(left_y - right_y) > threshold else "Even"

            if slouch_status == "Poor" or shoulder_status == "Uneven":
                is_bad_posture = True

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            slouch_status = "Error"

    if is_bad_posture:
        if bad_posture_start_time is None:
            bad_posture_start_time = time.time()

        elapsed_time = time.time() - bad_posture_start_time
        if elapsed_time >= notification_interval:
            playsound(sound_path)
            bad_posture_start_time = None
    else:
        bad_posture_start_time = None

    slouch_color = (0, 0, 255) if slouch_status == "Poor" else (0, 255, 0)
    cv2.putText(image, f"Posture: {slouch_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, slouch_color, 2, cv2.LINE_AA)

    shoulder_color = (0, 0, 255) if shoulder_status == "Uneven" else (0, 255, 0)
    cv2.putText(image, f"Shoulders: {shoulder_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, shoulder_color, 2, cv2.LINE_AA)

    cv2.imshow("Live Posture Detection", image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

