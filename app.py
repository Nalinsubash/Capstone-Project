import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os

# 🔹 New Model File from Google Drive
FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"
MODEL_PATH = "emotion_CNN_FInal_model.keras"

# 🔹 Google Drive Direct Download Link
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# 🔹 Download Model if Not Exists or is Empty
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    print("⏳ Downloading New Model from Google Drive...")
    response = requests.get(GDRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("✅ Model Downloaded Successfully!")

# 🔹 Load the Updated Model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    st.error(f"Error loading model: {e}")

# Emotion Labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------------------------
# 🔹 Image Preprocessing
# ---------------------------
def preprocess_image(image):
    """Preprocess image for model prediction."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized = cv2.resize(image, (100, 100))  # Resize to 100x100
    img_array = np.array(resized, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
    return img_array

# ---------------------------
# 🔹 Predict Emotion
# ---------------------------
def predict_emotion(image):
    """Make a prediction using the trained model."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# ---------------------------
# 🔹 Streamlit UI
# ---------------------------
st.title("🎭 Emotion Detection from Facial Images")
st.write("Upload an image, and the AI will predict the emotion.")

# -----------------------
# 📂 Upload Image (First Option)
# -----------------------
st.subheader("📂 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")
    emotion = predict_emotion(frame)
    st.subheader(f"Predicted Emotion: **{emotion}**")

# -----------------------
# 📷 Webcam Option
# -----------------------
st.subheader("📷 Use Webcam for Real-time Detection")

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class EmotionVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        emotion = predict_emotion(img)  # Predict emotion
        return img  # Return the original frame (we can overlay emotion text if needed)

# Stream webcam
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionVideoTransformer,
)

