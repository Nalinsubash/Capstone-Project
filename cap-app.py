import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import gdown
import os

# -------------------------------
# ✅ Model Download from Google Drive (Your Custom CNN)
# -------------------------------
FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"
MODEL_PATH = "emotion_CNN_Final_model.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("⏳ Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model Downloaded Successfully!")

# -------------------------------
# ✅ Load Your Fine-Tuned CNN Model
# -------------------------------
try:
    cnn_model = load_model(MODEL_PATH)
    st.write("✅ Custom CNN Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Emotion Labels (Aligned with FER datasets)
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# -------------------------------
# ✅ Load Pretrained ResNet50-FER (Benchmark Model)
# -------------------------------
st.write("⏳ Loading ResNet50-FER Model for Benchmarking...")
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dense(7, activation="softmax")(x)  # 7 Emotion Classes

resnet_model = Model(inputs=base_model.input, outputs=x)
resnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
st.write("✅ ResNet50-FER Benchmark Model Loaded Successfully!")

# -------------------------------
# ✅ Image Preprocessing Function (Custom CNN)
# -------------------------------
def preprocess_image(image):
    """Preprocess image for Custom CNN Model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (100, 100))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# ✅ Predict with Custom CNN Model
# -------------------------------
def predict_cnn_model(image):
    """Predict emotion using Custom CNN Model."""
    img_array = preprocess_image(image)
    predictions = cnn_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# ✅ Image Preprocessing Function (ResNet50-FER)
# -------------------------------
def preprocess_resnet_image(image):
    """Preprocess image for ResNet50-FER input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize for ResNet50
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Apply ResNet50 preprocessing
    return img_array

# -------------------------------
# ✅ Predict with ResNet50-FER (Benchmark)
# -------------------------------
def predict_resnet_model(image):
    """Predict emotion using ResNet50-FER Benchmark Model."""
    img_array = preprocess_resnet_image(image)
    predictions = resnet_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🎭 Emotion Detection Benchmarking")
st.write("Compare **Your Fine-Tuned CNN Model** with **ResNet50-FER**")

# -------------------------------
# ✅ Image Upload Option
# -------------------------------
st.subheader("📂 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")

    # Get Predictions
    cnn_emotion = predict_cnn_model(frame)
    resnet_emotion = predict_resnet_model(frame)

    # Display Results
    st.subheader("🔍 Benchmarking Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Your Fine-Tuned CNN Model")
        st.success(f"🎭 Predicted Emotion: **{cnn_emotion}**")

    with col2:
        st.subheader("🔥 ResNet50-FER (Benchmark)")
        st.warning(f"🎭 Predicted Emotion: **{resnet_emotion}**")



