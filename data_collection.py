#capture keypoints of good posture and poor posture and save to csv

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

csv_file = 'posture_data.csv'
file_exists = os.path.isfile(csv_file)
csv_header = [
    'l_shoulder_x',
    'l_shoulder_y',
    'r_shoulder_x',
    'r_shoulder_y',
    'l_hip_x',
    'l_hip_y',
    'r_hip_x',
    'r_hip_y',
    'l_ear_x',
    'l_ear_y',
    'r_ear_x',
    'r_ear_y',
    'class'
]

print(f"Starting data collection process: Press 'g' for good posture, 'b' for bad posture, and 'q' to quit")

with open(csv_file,'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(csv_header)
    
    current_class = 'unknown'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writable = False

        results = pose.process(image)

        image.flags.writable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('g'):
            current_class = 'good'
            print("Current class: good posture")
        elif key == ord('b'):
            current_class = 'bad'
            print("Current class: bad posture")
        elif key == ord('q'):
            break

        if results.pose_landmarks and current_class in ['good', 'bad']:
            landmarks = results.pose_landmarks.landmark
            try:
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                row = l_shoulder + r_shoulder + l_hip + r_hip + l_ear + r_ear + [current_class]
                writer.writerow(row)
            except Exception as e:
                print(f'Error extracting landmarks: {e}')
                
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(image, f'Class: {current_class.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('Data Collection', image)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data has been saved to csv file")




