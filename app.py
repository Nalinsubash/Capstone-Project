import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os

# 🔹 Load Model from Google Drive
FILE_ID = "1mrbEbFCQIk1MxZ1QghISipWfMQCNocf7"
MODEL_PATH = "emotion_CNN_FInal_model.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


if not os.path.exists(MODEL_PATH):
    st.write("⏳ Downloading Model from Google Drive...")
    response = requests.get(GDRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    st.write("✅ Model Downloaded Successfully!")

model = load_model(MODEL_PATH)
st.write("✅ Model Loaded Successfully!")

# Emotion Labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 🎭 App Title
st.title("🎭 Emotion Detection")

# 🔹 Image Preprocessing Function
def preprocess_image(image):
    """Preprocess image for emotion detection model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized = cv2.resize(image, (100, 100))  # Resize to (100, 100)
    img_array = np.array(resized, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model input
    return img_array

# 🔹 Emotion Prediction Function
def predict_emotion(image):
    """Predict emotion from processed image."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label


# -------------------------------
# 📂 **Option 1: Upload Image (Default)**
# -------------------------------
st.subheader("📂 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)  # Convert to OpenCV image
    st.image(frame, channels="BGR", caption="Uploaded Image")
    
    # Predict Emotion
    emotion = predict_emotion(frame)
    st.subheader(f"Predicted Emotion: **{emotion}**")


# -------------------------------
# 📷 **Option 2: Real-time Webcam Detection**
# -------------------------------
st.subheader("📷 Use Webcam for Real-time Detection")
use_webcam = st.checkbox("Enable Webcam")

# Webcam Processing Class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        """Process live video frame and predict emotion."""
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV BGR format
        emotion = predict_emotion(img)
        
        # Draw emotion label on frame
        cv2.putText(
            img, f"Emotion: {emotion}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )
        return img

if use_webcam:
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=VideoTransformer,
        desired_playing_state=True,
    )
