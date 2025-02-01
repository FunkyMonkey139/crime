import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract landmarks using MediaPipe
def extract_landmarks_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    landmarks = []
    frame_count = 0


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            # Extract landmark coordinates
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks.append(frame_landmarks)
        else:
            # Append a zero vector if no landmarks are detected
            landmarks.append([0.0] * 99)

        frame_count += 1

    cap.release()

    # Pad with empty frames if fewer frames are extracted
    if len(landmarks) < max_frames:
        landmarks.extend([[0.0] * 99] * (max_frames - len(landmarks)))

    return np.array(landmarks[:max_frames])

# Function to prepare the dataset
def prepare_dataset_with_mediapipe(dataset_path, max_frames=30):
    features = []
    labels = []

    for label, folder in enumerate(['fight', 'nofight']):
        folder_path = os.path.join(dataset_path, folder)
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            if video_file.endswith('.mp4'):
                video_landmarks = extract_landmarks_from_video(video_path, max_frames=max_frames)
                features.append(video_landmarks)
                labels.append(label)

    return np.array(features), np.array(labels)

# Build the LSTM model
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to train the model
def train_model_with_mediapipe(dataset_path, max_frames=30, epochs=15, batch_size=8):
    features, labels = prepare_dataset_with_mediapipe(dataset_path, max_frames=max_frames)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_model(input_shape=(features.shape[1], features.shape[2]), num_classes=2)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    model.save('fight_detection_model_mediapipe.h5')
    print("Model trained and saved as 'fight_detection_model_mediapipe.h5'.")

# Specify the dataset path and train the model
dataset_path = '/Users/ayushshetty/Desktop/FinalYrCrime/data'
train_model_with_mediapipe(dataset_path)