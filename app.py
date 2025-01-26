import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av  # Required for processing video frames
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# 🔹 Load Model from Google Drive
FILE_ID = "1mrbEbFCQIk1MxZ1QghISipWfMQCNocf7"
MODEL_PATH = "emotion_CNN_FInal_model.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("⏳ Downloading Model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model Downloaded Successfully!")

# Load the model
model = load_model(MODEL_PATH)
st.write("✅ Model Loaded Successfully!")

# Emotion labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

st.title("🎭 Emotion Detection")

# 📂 Image Upload Option
st.subheader("📂 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    """Convert image to correct format for the model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (100, 100))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(image):
    """Predict emotion from processed image."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")
    emotion = predict_emotion(frame)
    st.subheader(f"Predicted Emotion: **{emotion}**")

# 📷 Webcam Detection
st.subheader("📷 Use Webcam for Real-time Detection")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    """Processes video frames and overlays predicted emotion."""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        emotion = predict_emotion(img)
        img = cv2.putText(
            img,
            f"Emotion: {emotion}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if not webrtc_ctx.state.playing:
    st.warning("⚠️ No available webcams detected. Check your device settings.")

