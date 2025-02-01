
import json
from flask import Flask, request, jsonify, Response
import cv2
import threading
import tensorflow as tf
import numpy as np
import mediapipe as mp
import os
import requests
import time

app = Flask(__name__)

# Initialize global variables for thread-safe operation
output_data = {}
lock = threading.Lock()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Download and load the emotion detection model if it's not already downloaded
MODEL_PATH = "emotion_model.hdf5"
MODEL_URL = "https://github.com/vjgpt/Face-and-Emotion-Recognition/raw/master/models/emotion_model.hdf5"

if not os.path.exists(MODEL_PATH):
    print("Downloading emotion detection model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define a function to process video frames and update global output_data
def process_video():
    global output_data, lock

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    max_frames = 30
    landmark_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and resize frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert frame for different detections
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_behaviors = []
        detected_emotions = []

        # Emotion Detection
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.equalizeHist(gray_face)
                    resized_face = cv2.resize(gray_face, (64, 64))
                    normalized_face = resized_face.astype("float32") / 255.0
                    normalized_face = np.expand_dims(normalized_face, axis=(0, -1))

                    predictions = emotion_model.predict(normalized_face, verbose=0)
                    emotion_index = np.argmax(predictions)
                    confidence = np.max(predictions)

                    if confidence > 0.6:
                        detected_emotions.append({
                            "emotion": emotion_labels[emotion_index],
                            "confidence": float(confidence)
                        })

        # Gesture Detection (omitted for brevity)

        # Fight Detection (omitted for brevity)

        # Update global output data
        with lock:
            output_data = {
                "emotions": detected_emotions,
                "behaviors": list(set(detected_behaviors)),
                "fight_detection": 1  # replace with your actual fight status logic
            }

        time.sleep(0.1)  # Sleep to avoid excessive CPU usage

    cap.release()
    hands.close()
    pose.close()

# Run the video processing in a separate thread
video_thread = threading.Thread(target=process_video, daemon=True)
video_thread.start()

# Streaming function to handle continuous output
def generate_continuous_output():
    while True:
        with lock:
            response = output_data.copy()  # Ensure thread safety
        yield f"data: {json.dumps(response)}\n\n"
        time.sleep(0.1)  # Sleep to reduce load, adjust as needed

@app.route("/detect", methods=["GET"])
def detect():
    return Response(generate_continuous_output(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
