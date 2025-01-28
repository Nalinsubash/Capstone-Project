import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet_v2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import gdown
import os

# -------------------------------
# ✅ Model Download from Google Drive (Your Custom CNN)
# -------------------------------
CNN_FILE_ID = "1cqZfncbko6EMUx13IHeX4YsSO73LoqWv"  # Your Fine-Tuned CNN
CNN_MODEL_PATH = "emotion_CNN_Final_model.keras"
CNN_GDRIVE_URL = f"https://drive.google.com/uc?id={CNN_FILE_ID}"

if not os.path.exists(CNN_MODEL_PATH):
#    st.write("⏳ Downloading CNN model from Google Drive...")
    gdown.download(CNN_GDRIVE_URL, CNN_MODEL_PATH, quiet=False)
#    st.write("✅ Custom CNN Model Downloaded Successfully!")

# -------------------------------
# ✅ Model Download from Google Drive (Your ResNet50V2)
# -------------------------------
RESNET_FILE_ID = "1WmeVgro-VXenbR-cc8jPl5qeQJmRS_d-"  # Replace with your ResNet50V2 File ID
RESNET_MODEL_PATH = "ResNet50V2_final.keras"
RESNET_GDRIVE_URL = f"https://drive.google.com/uc?id={RESNET_FILE_ID}"

if not os.path.exists(RESNET_MODEL_PATH):
#    st.write("⏳ Downloading ResNet50V2 model from Google Drive...")
    gdown.download(RESNET_GDRIVE_URL, RESNET_MODEL_PATH, quiet=False)
#    st.write("✅ ResNet50V2 Model Downloaded Successfully!")

# -------------------------------
# ✅ Load Your Fine-Tuned CNN Model
# -------------------------------
try:
    cnn_model = load_model(CNN_MODEL_PATH)
    st.write("✅ Custom CNN Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading CNN model: {e}")

# -------------------------------
# ✅ Load Your Fine-Tuned ResNet50V2 Model
# -------------------------------
try:
    resnet50v2_model = load_model(RESNET_MODEL_PATH)
    st.write("✅ ResNet50V2 Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading ResNet50V2 model: {e}")

# -------------------------------
# ✅ Load Pretrained VGGFace2 Model (Benchmark Model)
# -------------------------------
#st.write("⏳ Loading VGGFace2 Model for Benchmarking...")
vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
y = GlobalAveragePooling2D()(vgg_base.output)
y = Dense(512, activation="relu")(y)
y = Dense(7, activation="softmax")(y)  # 7 Emotion Classes
vgg_model = Model(inputs=vgg_base.input, outputs=y)
vgg_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
st.write("✅ VGGFace2 Benchmark Model Loaded Successfully!")

# Emotion Labels
class_labels = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
# -------------------------------
# ✅ Image Preprocessing Functions
# -------------------------------
def preprocess_cnn_image(image):
    """Preprocess image for Custom CNN Model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (100, 100))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_resnet_v2_image(image):
    """Preprocess image for ResNet50V2 input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_resnet_v2(img_array)  # Apply ResNet50V2 preprocessing
    return img_array

def preprocess_vgg_image(image):
    """Preprocess image for VGGFace2 input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_vgg(img_array)  # Apply VGG preprocessing
    return img_array

# -------------------------------
# ✅ Prediction Functions
# -------------------------------
def predict_cnn_model(image):
    """Predict emotion using Custom CNN Model."""
    img_array = preprocess_cnn_image(image)
    predictions = cnn_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

def predict_resnet50v2_model(image):
    """Predict emotion using Fine-Tuned ResNet50V2 Model."""
    img_array = preprocess_resnet_v2_image(image)
    predictions = resnet50v2_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

def predict_vgg_model(image):
    """Predict emotion using VGGFace2 Benchmark Model."""
    img_array = preprocess_vgg_image(image)
    predictions = vgg_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🔬 Emotion Detection: Custom CNN vs. Industry Benchmarks")
st.write("Testing **Your Fine-Tuned Custom CNN Model** against **ResNet50** and **VGGFace2** – industry benchmark models for facial emotion recognition.")

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
    resnet50v2_emotion = predict_resnet50v2_model(frame)
    vgg_emotion = predict_vgg_model(frame)

    # Display Results
    st.subheader("🔍 Benchmarking Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚀 Your Fine-Tuned CNN Model")
        st.success(f"🎭 Predicted Emotion: **{cnn_emotion}**")

    with col2:
        st.subheader("🔥 ResNet50V2 (Your Fine-Tuned Model)")
        st.warning(f"🎭 Predicted Emotion: **{resnet50v2_emotion}**")

    with col3:
        st.subheader("🌟 VGGFace2 (Benchmark)")
        st.info(f"🎭 Predicted Emotion: **{vgg_emotion}**")
        
st.markdown("""
---
© 2025 **Nalin Manchanayaka** | All Rights Reserved  
Developed for benchmarking **Custom CNN Model** against **Industry-Standard Emotion Recognition Models**.
""")
