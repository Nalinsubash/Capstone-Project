import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os
import av

# -------------------------------
# ✅ Model Download from Google Drive
# -------------------------------
FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"
MODEL_PATH = "emotion_CNN_Final_model.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("⏳ Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model Downloaded Successfully!")

# -------------------------------
# ✅ Load Model
# -------------------------------
try:
    model = load_model(MODEL_PATH)
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Emotion Labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# -------------------------------
# ✅ Image Preprocessing Function
# -------------------------------
def preprocess_image(image):
    """Preprocess image for emotion prediction."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (100, 100))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# ✅ Predict Emotion Function
# -------------------------------
def predict_emotion(image):
    """Predict emotion from image using trained model."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🎭 Emotion Detection App")
st.write("Upload an image or use your webcam for real-time emotion detection.")

# -------------------------------
# ✅ Image Upload Option
# -------------------------------
st.subheader("📂 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    st.image(frame, channels="BGR", caption="Uploaded Image")
    
    emotion = predict_emotion(frame)
    st.subheader(f"🎭 Predicted Emotion: **{emotion}**")

# -------------------------------
# ✅ WebRTC Configuration (Fix Applied)
# -------------------------------
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        {"urls": "stun:stun3.l.google.com:19302"},
        {"urls": "stun:stun4.l.google.com:19302"},
        {"urls": "turn:relay.metered.ca:80", "username": "open", "credential": "open"},
        {"urls": "turn:relay.metered.ca:443", "username": "open", "credential": "open"},
        {"urls": "turn:relay.metered.ca:443?transport=tcp", "username": "open", "credential": "open"},
        {"urls": "turn:openrelay.metered.ca:80", "username": "open", "credential": "open"},
        {"urls": "turn:openrelay.metered.ca:443", "username": "open", "credential": "open"},
        {"urls": "turn:openrelay.metered.ca:443?transport=tcp", "username": "open", "credential": "open"},
    ]
}

# -------------------------------
# ✅ Real-time Webcam Video Processing 
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        image = frame.to_ndarray(format="bgr24")
        emotion = predict_emotion(image)  # Assume this function is defined

        # ✅ Overlay text on the video frame
        cv2.putText(
            image, f"Emotion: {emotion}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")
# -------------------------------
# ✅ Webcam Integration (Fix Applied)
# -------------------------------
st.subheader("📸 Use Webcam for Real-time Detection")

webrtc_ctx = webrtc_streamer(
    key="example",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor, 
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # ✅ Prevents Freezing
)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("🔵 **Webcam is running!**")
else:
    st.write("🔴 **Webcam failed to start! Check network or browser settings.**")
