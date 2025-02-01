import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os
import datetime
import requests

# ===============================
# Emotion Detection Setup
# ===============================
# Download the pre-trained model for emotion detection
MODEL_URL = "https://github.com/vjgpt/Face-and-Emotion-Recognition/raw/master/models/emotion_model.hdf5"
MODEL_PATH = "emotion_model.hdf5"

if not os.path.exists(MODEL_PATH):
    print("Downloading emotion detection model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Load the emotion detection model
emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ===============================
# Gesture Detection and Fight Detection Setup
# ===============================
# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the fight detection model
fight_model = tf.keras.models.load_model('/Users/ayushshetty/Desktop/FinalYrCrime/fight_detection_model_mediapipe.h5')

# Define behaviors
BEHAVIORS = {
    "fist_clench": "Agression Detected",
    "pointing": "Agression Detected",
}

# Function to calculate Euclidean distance
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# Gesture detection: fist clench
def detect_fist_clench(hand_landmarks):
    if hand_landmarks:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        finger_tips = [
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        ]
        clenched = all(calculate_distance(finger_tip, wrist) < 0.15 for finger_tip in finger_tips)
        return clenched
    return False

# Gesture detection: pointing
def detect_pointing(hand_landmarks):
    if hand_landmarks:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        others = [
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        ]
        index_extended = calculate_distance(index_tip, wrist) > 0.25
        others_folded = all(calculate_distance(finger_tip, wrist) < 0.20 for finger_tip in others)
        return index_extended and others_folded
    return False

# Extract pose landmarks for fight detection
def extract_landmarks_from_frame(frame):
    landmarks = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        frame_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        landmarks = frame_landmarks
    else:
        landmarks = [0.0] * 99

    return np.array(landmarks), results.pose_landmarks

# ===============================
# Real-Time Video Processing
# ===============================
cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)  # Adjusted model selection

max_frames = 30
landmark_sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting...")
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame for different detections
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected_behaviors = []

    # ===============================
    # Emotion Detection
    # ===============================
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:  # Check if face region is valid
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.equalizeHist(gray_face)
                resized_face = cv2.resize(gray_face, (64, 64))
                normalized_face = resized_face.astype("float32") / 255.0
                normalized_face = np.expand_dims(normalized_face, axis=(0, -1))

                predictions = emotion_model.predict(normalized_face, verbose=0)
                emotion_index = np.argmax(predictions)
                confidence = np.max(predictions)

                emotion = emotion_labels[emotion_index] if confidence > 0.6 else "Uncertain"
                # Display emotion label above the detected face region
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ===============================
    # Gesture Detection
    # ===============================
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if detect_fist_clench(hand_landmarks):
                detected_behaviors.append(BEHAVIORS["fist_clench"])
            if detect_pointing(hand_landmarks):
                detected_behaviors.append(BEHAVIORS["pointing"])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ===============================
    # Fight Detection
    # ===============================
    landmarks, pose_landmarks = extract_landmarks_from_frame(frame)
    landmark_sequence.append(landmarks)

    if len(landmark_sequence) > max_frames:
        landmark_sequence.pop(0)

    if len(landmark_sequence) == max_frames:
        input_data = np.expand_dims(np.array(landmark_sequence), axis=0)
        prediction = fight_model.predict(input_data)
        predicted_class = np.argmax(prediction)
        label = "Fight" if predicted_class == 0 else "No Fight"
        confidence = np.max(prediction)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw pose landmarks
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # ===============================
    # Display Behaviors and Timestamp
    # ===============================
    for i, behavior in enumerate(set(detected_behaviors)):
        cv2.putText(frame, behavior, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    timestamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p")
    cv2.putText(frame, timestamp, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("Integrated Real-Time Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
