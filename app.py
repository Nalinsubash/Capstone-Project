import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os


# 🔹 Replace with your actual Google Drive FILE ID
FILE_ID = "1-y0icKArgc0EvDuvfTaZQM4sk-EiWZzP"
MODEL_PATH = "emotion_CNN_Final_model.h5"

# 🔹 Google Drive Direct Download Link (gdown format)
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# 🔹 Download Model if Not Exists
if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading Model from Google Drive...")
    response = requests.get(GDRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("✅ Model Downloaded Successfully!")

# 🔹 Load Model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")



# Class labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Streamlit UI
st.title("😃 Emotion Detection")
st.write("Upload an image or turn on your webcam for real-time emotion detection!")

# Select mode
mode = st.radio("Choose an option:", ("📷 Use Webcam", "📂 Upload an Image"))

# Function to predict emotion
def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))  # Resize to model input size
    img_array = np.expand_dims(resized, axis=(0, -1))  # Add batch & channel dimensions
    img_array = img_array / 255.0  # Normalize
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -----------------------
# 📂 Option 1: Upload Image
# -----------------------
if mode == "📂 Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Show original image
        st.image(frame, channels="BGR", caption="Uploaded Image")

        # Predict emotion
        emotion = predict_emotion(frame)
        st.subheader(f"Predicted Emotion: **{emotion}**")

# -----------------------
# 📷 Option 2: Use Webcam
# -----------------------
elif mode == "📷 Use Webcam":
    st.write("Click 'Start' to capture from your webcam.")

    cam = cv2.VideoCapture(0)  # Open webcam

    if st.button("Start Webcam"):
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Predict emotion
            emotion = predict_emotion(frame)

            # Show webcam feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB", caption=f"Predicted Emotion: {emotion}")

            # Stop the webcam when user clicks "Stop"
            if st.button("Stop Webcam"):
                cam.release()
                break
