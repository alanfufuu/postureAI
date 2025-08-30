#capture keypoints of good posture and poor posture and save to csv
import cv2
import mediapipe as mp
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #make sure this camera index is correct for you
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

csv_file = 'posture_data.csv' 
file_exists = os.path.isfile(csv_file)
csv_header = [
    'nose_x', 'nose_y',
    'l_shoulder_x', 'l_shoulder_y', 'r_shoulder_x', 'r_shoulder_y',
    'l_ear_x', 'l_ear_y', 'r_ear_x', 'r_ear_y', 'class'
]

print("Starting data collection...")
print("Press 'g' for good posture, 'b' for bad posture, and 'q' to quit.")

with open(csv_file, 'w', newline='') as f: # 'w' to write instead of append
    writer = csv.writer(f)
    writer.writerow(csv_header)
    
    current_class = 'unknown'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('g'):
            current_class = 'good'
        elif key == ord('b'):
            current_class = 'bad'
        elif key == ord('q'):
            break

        if results.pose_landmarks and current_class in ['good', 'bad']:
            try:
                landmarks = results.pose_landmarks.landmark
                
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                row = nose + l_shoulder + r_shoulder + l_ear + r_ear + [current_class]
                writer.writerow(row)
                
            except Exception as e:
                print(f'Error extracting landmarks: {e}')

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(image, f'CLASS: {current_class.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', image)

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to {csv_file}")







