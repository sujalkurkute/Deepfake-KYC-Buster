import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/deepfake_model.h5")

model = load_model()
IMG_SIZE = 224

st.set_page_config(page_title="Deepfake KYC Buster", layout="centered")

st.markdown("<h1 style='text-align:center;'>🔍 Deepfake KYC Buster</h1>", unsafe_allow_html=True)
st.write("Upload a face image to detect whether it is Real or Fake.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)[0][0]
    prediction = float(prediction)  # convert numpy.float32 → python float

    if prediction > 0.5:
        label = "🚨 FAKE DETECTED"
        confidence = prediction
        st.error(label)
    else:
        label = "✅ REAL IMAGE"
        confidence = 1 - prediction
        st.success(label)

    confidence_percent = round(confidence * 100, 2)

    # Progress bar expects int 0-100
    st.progress(int(confidence * 100))

    st.write(f"### Confidence: {confidence_percent}%")

st.markdown("---")
st.markdown(
    "<center>Developed using MobileNetV2 + FaceForensics++ Dataset</center>",
    unsafe_allow_html=True
)