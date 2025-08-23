import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('posture_classifier.h5')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()
    
class_names = ['Good', 'Poor']

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

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- 1. SLOUCH DETECTION (using the new model) ---
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            
            input_data = np.array([nose + l_shoulder + r_shoulder + l_ear + r_ear])
            prediction = model.predict(input_data)
            slouch_status = class_names[np.argmax(prediction)]

            # --- 2. SHOULDER IMBALANCE CHECK (new rule-based check) ---
            left_y = l_shoulder[1]
            right_y = r_shoulder[1]
            
            # Use shoulder width as a reference for the threshold
            shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            threshold = shoulder_width * 0.08  # 8% of shoulder width as threshold

            if abs(left_y - right_y) > threshold:
                shoulder_status = "Uneven"
            else:
                shoulder_status = "Even"

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            slouch_status = "Error"
            print(f'Classification error: {e}')

    # Display Slouch Status
    slouch_color = (0, 0, 255) if slouch_status == 'Poor' else (0, 255, 0)
    cv2.putText(image, f'Slouch: {slouch_status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, slouch_color, 2, cv2.LINE_AA)
    
    # Display Shoulder Status
    shoulder_color = (0, 0, 255) if shoulder_status == 'Uneven' else (0, 255, 0)
    cv2.putText(image, f'Shoulders: {shoulder_status}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, shoulder_color, 2, cv2.LINE_AA)
    
    cv2.imshow('Live Posture Detection', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()