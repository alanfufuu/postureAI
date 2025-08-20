#start webcam, process each frame, use model to classify posture, stream to frontend through websockets
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app)

try:
    model = tf.keras.models.load_model('posture_classifier.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frames():
    last_status = ""

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writable = False

        results = pose.process(image)

        image.flags.writable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status = "Unknown"

        if results.pose_landmarks and model:
            landmarks = results.pose_landmarks.landmark
            try:
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                input_data = np.array([l_shoulder + r_shoulder + l_hip + r_hip + l_ear + r_ear])

                prediction = model.predict(input_data)
                predicted_class_index = np.argmax(prediction)

                status = 'Good' if predicted_class_index == 0 else 'Poor'

                if status == 'Poor' and last_status != 'Poor':
                    socketio.emit('posture_alert', {'status': 'Poor'})
                last_status = status
            except Exception as e:
                status = "error"
                print(f'Classification error: {e}')

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(image, 
                        f'Status: {status}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255) if status == 'Poor' else (0, 255, 0), 
                        2, 
                        cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()

        socketio.emit('video_frame', {'image': frame_bytes, 'status' : status})
        socketio.sleep(0.05)


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("connected")
    socketio.start_background_task(target=process_frames)

if __name__ == '__main__':
    socketio.run(app, debug=True)






















