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
# 📷 Option 2: Use Webcam
# -----------------------
elif mode == "📷 Use Webcam":
    st.write("Click 'Start Webcam' to begin capturing.")

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False

    start_webcam = st.button("🎥 Start Webcam")

    if start_webcam:
        st.session_state.webcam_active = True

    stop_webcam = st.button("🛑 Stop Webcam")

    if stop_webcam:
        st.session_state.webcam_active = False

    if st.session_state.webcam_active:
        cam = cv2.VideoCapture(0)

        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("❌ Failed to capture image from webcam.")
                break

            # Predict emotion
            emotion = predict_emotion(frame)

            # Convert to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB", caption=f"🎭 Predicted Emotion: {emotion}")

            # Stop the webcam if the button is clicked
            if stop_webcam:
                cam.release()
                break

        cam.release()

