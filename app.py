import streamlit as st
import tensorflow as tf
import numpy as np
import os

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("🩺 Skin Cancer Detection using CNN")
st.write("Upload a skin lesion image to check if it's **benign** or **malignant**.")

# ---------------------------
# Load Pretrained Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.keras")  # Load saved model
        return model
    except Exception as e:
        st.error("❌ Model file not found! Please train and save the model as 'model.keras'.")
        return None

model = load_model()

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload a skin lesion image", type=["jpg","jpeg","png"])

if uploaded_file and model:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=200)

    # Preprocess image
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(128,128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "⚠️ Malignant" if prediction > 0.5 else "✅ Benign"

    # Show result
    st.subheader("Prediction Result")
    st.write(f"Model Confidence: {prediction:.2f}")
    if prediction > 0.5:
        st.error(label)
    else:
        st.success(label)
