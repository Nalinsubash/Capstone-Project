import streamlit as st
import os
import gdown
import torch
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from transformers import AutoFeatureExtractor, SwinForImageClassification

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
# ✅ Load Your Fine-Tuned Model
# -------------------------------
try:
    cnn_model = load_model(MODEL_PATH)
    st.write("✅ Custom CNN Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Emotion Labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# -------------------------------
# ✅ Load Pretrained Swin Transformer Model
# -------------------------------
extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")
swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

# -------------------------------
# ✅ Image Preprocessing Function
# -------------------------------
def preprocess_image(image):
    """Preprocess image for emotion prediction (Ensure correct shape)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized = cv2.resize(image, (100, 100))  # ✅ Resize to (100,100,3)
    img_array = np.array(resized, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match batch input
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
# ✅ Predict with Swin Transformer (Benchmark Model)
# -------------------------------
def predict_swin_transformer(image):
    """Predict emotion using Swin Transformer."""
    inputs = extractor(images=image, return_tensors="pt")
    outputs = swin_model(**inputs)
    preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(preds, dim=-1).item()
    return class_labels[predicted_class]

# -------------------------------
# 🎭 Streamlit UI
# -------------------------------
st.title("🎭 Emotion Detection Benchmarking")
st.write("Compare **Your Fine-Tuned CNN Model** with **Swin Transformer**")

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
    swin_emotion = predict_swin_transformer(frame)

    # Display Results
    st.subheader("🔍 Benchmarking Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Your Fine-Tuned CNN Model")
        st.success(f"🎭 Predicted Emotion: **{cnn_emotion}**")

    with col2:
        st.subheader("🔥 Swin Transformer (Benchmark)")
        st.warning(f"🎭 Predicted Emotion: **{swin_emotion}**")

# -------------------------------
# ❌ Commented Out Webcam Integration (For Now)
# -------------------------------
# st.subheader("📸 Use Webcam for Real-time Detection")
# webrtc_ctx = webrtc_streamer(
#     key="example",
#     rtc_configuration=RTC_CONFIGURATION,
#     video_processor_factory=VideoProcessor, 
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,  # ✅ Prevents Freezing
# )
# if webrtc_ctx and webrtc_ctx.state.playing:
#     st.write("🔵 **Webcam is running!**")
# else:
#     st.write("🔴 **Webcam failed to start! Check network or browser settings.**")

