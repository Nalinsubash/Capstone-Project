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

# Constants
CLASS_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# -------------------------------
# ✅ Model Download and Loading
# -------------------------------
@st.cache(allow_output_mutation=True)
def download_and_load_model(file_id, model_path, model_name):
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    try:
        model = load_model(model_path)
        st.success(f"✅ {model_name} Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading {model_name} model: {e}")
        return None

cnn_model = download_and_load_model("1cqZfncbko6EMUx13IHeX4YsSO73LoqWv", "emotion_CNN_Final_model.keras", "Custom CNN")
resnet50v2_model = download_and_load_model("1WmeVgro-VXenbR-cc8jPl5qeQJmRS_d-", "ResNet50V2_final.keras", "ResNet50V2")

# Load VGGFace2 Model
@st.cache(allow_output_mutation=True)
def load_vgg_model():
    vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    y = GlobalAveragePooling2D()(vgg_base.output)
    y = Dense(512, activation="relu")(y)
    y = Dense(7, activation="softmax")(y)  # 7 Emotion Classes
    model = Model(inputs=vgg_base.input, outputs=y)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    st.success("✅ VGGFace2 Benchmark Model Loaded Successfully!")
    return model

vgg_model = load_vgg_model()

# -------------------------------
# ✅ Image Preprocessing Functions
# -------------------------------
def preprocess_image(image, model_type):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model_type == "cnn":
        resized = cv2.resize(image, (100, 100))
        img_array = np.array(resized, dtype=np.float32) / 255.0
    elif model_type == "resnet":
        resized = cv2.resize(image, (224, 224))
        img_array = np.array(resized, dtype=np.float32)
        img_array = preprocess_resnet_v2(img_array)
    elif model_type == "vgg":
        resized = cv2.resize(image, (224, 224))
        img_array = np.array(resized, dtype=np.float32)
        img_array = preprocess_vgg(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# ✅ Prediction Functions
# -------------------------------
@st.cache
def predict_emotion(image, model, model_type):
    img_array = preprocess_image(image, model_type)
    predictions = model.predict(img_array)
    return CLASS_LABELS[np.argmax(predictions)]

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🔬 Emotion Detection: Custom CNN vs. Industry Benchmarks")
st.write("Testing **Your Fine-Tuned Custom CNN Model** against **ResNet50** and **VGGFace2** – industry benchmark models for facial emotion recognition.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")

    # Get Predictions
    with st.spinner("Predicting emotions..."):
        cnn_emotion = predict_emotion(frame, cnn_model, "cnn")
        resnet50v2_emotion = predict_emotion(frame, resnet50v2_model, "resnet")
        vgg_emotion = predict_emotion(frame, vgg_model, "vgg")

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
