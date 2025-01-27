import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os
import av
from transformers import ViTImageProcessor, SwinForImageClassification
import torch

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

from transformers import AutoImageProcessor, SwinForImageClassification
import torch

# ✅ Load the pre-trained feature extractor (Fix ImportError)
transform = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# ✅ Ensure Swin model is properly loaded
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
swin_model.to(device)
swin_model.eval()



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
def preprocess_swin_image(image):
    """Preprocess image for Swin Transformer input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
    image = cv2.resize(image, (224, 224))  # Resize to Swin input size
    inputs = transform(images=image, return_tensors="pt")  # Transform input
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available
    return inputs

def predict_swin_transformer(image):
    """Predict emotion using Swin Transformer Model."""
    inputs = preprocess_swin_image(image)
    with torch.no_grad():  # ✅ Disable gradient computation for inference
        outputs = swin_model(**inputs)
        preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class_idx = torch.argmax(preds, dim=-1).item()
    
    if predicted_class_idx < len(class_labels):  # ✅ Ensure valid index
        return class_labels[predicted_class_idx]
    else:
        return "Unknown"  # Fallback in case of index mismatch

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

