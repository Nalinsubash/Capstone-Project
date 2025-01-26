import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import gdown
import os

# 🔹 New Model File from Google Drive
FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"
MODEL_PATH = "emotion_CNN_FInal_model.keras"

# 🔹 Google Drive Direct Download Link
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# 🔹 Download Model if Not Exists or is Empty
# Ensure the model file is downloaded
if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading Model from Google Drive...")
    try:
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("✅ Model Downloaded Successfully!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
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
