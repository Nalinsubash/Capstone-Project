import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import requests
import os

# ---------------------------
# 🔹 Google Drive Model Download
# ---------------------------
FILE_ID = "1-y0icKArgc0EvDuvfTaZQM4sk-EiWZzP"
MODEL_PATH = "emotion_CNN_Final_model.h5"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already exists
if not os.path.exists(MODEL_PATH):
    st.info("⏳ Downloading Model from Google Drive...")
    response = requests.get(GDRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    st.success("✅ Model Downloaded Successfully!")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
st.success("✅ Model Loaded Successfully!")

# Define emotion classes
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------------------------
# 🔹 Image Preprocessing
# ---------------------------
def preprocess_image(image):
    """Preprocess an image to match model input shape (100,100,3)."""
    # Convert image to RGB (in case it's in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to (100, 100)
    resized = cv2.resize(image, (100, 100))

    # Normalize pixel values
    img_array = np.array(resized, dtype=np.float32) / 255.0

    # Expand dimensions to match (1, 100, 100, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------------------------
# 🔹 Predict Emotion
# ---------------------------
def predict_emotion(image):
    """Predict emotion from the preprocessed image."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# ---------------------------
# 🔹 Streamlit UI
# ---------------------------
st.title("🎭 Emotion Detection System")
st.write("Upload an image or use your webcam to detect emotions!")

# Select mode
# mode = st.radio("Choose an option:", ("📷 Use Webcam", "📂 Upload an Image"))
mode = st.radio("Choose an option:", ("📂 Upload an Image","📷 Use Webcam"))

# -----------------------
# 📂 Option 1: Upload Image
# -----------------------
if mode == "📂 Upload an Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Display uploaded image
        st.image(frame, channels="BGR", caption="📸 Uploaded Image")

        # Predict emotion
        emotion = predict_emotion(frame)
        st.subheader(f"🎭 Predicted Emotion: **{emotion}**")

# -----------------------
# 📷 Option 2: Use Webcam (Fixed)
# -----------------------
elif mode == "📷 Use Webcam":
    st.write("Click 'Start Webcam' to begin capturing.")

    # **Fix: Detect available webcams**
    def get_available_cameras():
        available_cameras = []
        for i in range(3):  # Check first 3 camera indexes
            cam = cv2.VideoCapture(i)
            if cam.read()[0]:
                available_cameras.append(i)
            cam.release()
        return available_cameras

    available_cameras = get_available_cameras()

    if not available_cameras:
        st.error("❌ No available webcams detected. Check your device settings.")
        st.stop()

    cam_index = available_cameras[0]  # Use the first detected camera
    cam = cv2.VideoCapture(cam_index)

    start_webcam = st.button("🎥 Start Webcam")

    if start_webcam:
        st.info(f"✅ Using Camera Index: {cam_index}")

        FRAME_WINDOW = st.image([])
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                st.error("❌ Failed to capture image from webcam.")
                break

            emotion = predict_emotion(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB", caption=f"🎭 Predicted Emotion: {emotion}")

            time.sleep(0.1)  # Prevent freezing

        cam.release()
