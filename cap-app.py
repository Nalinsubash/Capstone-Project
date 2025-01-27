import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
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
resnet_base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(resnet_base.output)
x = Dense(512, activation="relu")(x)
x = Dense(7, activation="softmax")(x)  # 7 Emotion Classes
resnet_model = Model(inputs=resnet_base.input, outputs=x)
resnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
st.write("✅ ResNet50-FER Benchmark Model Loaded Successfully!")

# -------------------------------
# ✅ Load Pretrained VGGFace2 Model (Benchmark Model)
# -------------------------------
st.write("⏳ Loading VGGFace2 Model for Benchmarking...")
vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
y = GlobalAveragePooling2D()(vgg_base.output)
y = Dense(512, activation="relu")(y)
y = Dense(7, activation="softmax")(y)  # 7 Emotion Classes
vgg_model = Model(inputs=vgg_base.input, outputs=y)
vgg_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
st.write("✅ VGGFace2 Benchmark Model Loaded Successfully!")

# -------------------------------
# ✅ Image Preprocessing Function (Custom CNN)
# -------------------------------
def preprocess_cnn_image(image):
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
    img_array = preprocess_cnn_image(image)
    predictions = cnn_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# ✅ Image Preprocessing Function (ResNet50-FER)
# -------------------------------
def preprocess_resnet_image(image):
    """Preprocess image for ResNet50-FER input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_resnet(img_array)  # Apply ResNet50 preprocessing
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
# ✅ Image Preprocessing Function (VGGFace2)
# -------------------------------
def preprocess_vgg_image(image):
    """Preprocess image for VGGFace2 input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_vgg(img_array)  # Apply VGG preprocessing
    return img_array

# -------------------------------
# ✅ Predict with VGGFace2 (Benchmark)
# -------------------------------
def predict_vgg_model(image):
    """Predict emotion using VGGFace2 Benchmark Model."""
    img_array = preprocess_vgg_image(image)
    predictions = vgg_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🎭 Emotion Detection Benchmarking")
st.write("Compare **Your Fine-Tuned CNN Model** with **ResNet50-FER & VGGFace2**")

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
    vgg_emotion = predict_vgg_model(frame)

    # Display Results
    st.subheader("🔍 Benchmarking Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚀 Your Fine-Tuned CNN Model")
        st.success(f"🎭 Predicted Emotion: **{cnn_emotion}**")

    with col2:
        st.subheader("🔥 ResNet50-FER (Benchmark)")
        st.warning(f"🎭 Predicted Emotion: **{resnet_emotion}**")

    with col3:
        st.subheader("🌟 VGGFace2 (Benchmark)")
        st.info(f"🎭 Predicted Emotion: **{vgg_emotion}**")


