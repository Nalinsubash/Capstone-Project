import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os
from aiortc import RTCConfiguration
from streamlit_webrtc import webrtc_streamer

# 🔹 Disable GPU (if CUDA error is happening)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔹 Load Model from Google Drive
FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"
MODEL_PATH = "emotion_CNN_FInal_model.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("⏳ Downloading Model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model Downloaded Successfully!")

# 🔹 Load the model without compiling
try:
    model = load_model(MODEL_PATH, compile=False)
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# 🔹 Emotion Labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 🔹 Image Preprocessing
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (100, 100))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 🔹 Predict Emotion
def predict_emotion(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# 🔹 Streamlit UI
st.title("🎭 Emotion Detection")

# -----------------------
# 📂 Upload Image First
# -----------------------
st.subheader("📤 Upload an Image for Emotion Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")

    emotion = predict_emotion(frame)
    st.subheader(f"Predicted Emotion: **{emotion}**")

# -----------------------
# 📷 Use Webcam
# -----------------------
st.subheader("📷 Use Webcam for Real-time Detection")

RTC_CONFIGURATION = RTCConfiguration(
    iceServers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": "turn:relay.metered.ca:80", "username": "open", "credential": "open"},
    ]
)

webrtc_ctx = webrtc_streamer(
    key="example",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_receiver:
    try:
        frame = webrtc_ctx.video_receiver.get_frame()
        emotion = predict_emotion(frame)
        st.write(f"Predicted Emotion: **{emotion}**")
    except Exception as e:
        st.error(f"Error processing webcam feed: {str(e)}")

