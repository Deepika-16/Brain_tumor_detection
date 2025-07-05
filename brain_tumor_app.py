import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "C:/Users/deepi/OneDrive/Desktop/Brain Tumor Detection using CNN-20250331T150036Z-001/Brain Tumor Detection using CNN/brain_tumor_model.h5"  # Update with your model path
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Function to preprocess image
def load_and_preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain scan image to check for tumors.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess the image
    img_array = load_and_preprocess_image(image)

    # Predict
    prediction = model.predict(img_array)

    # Display the result
    result = "🔴 TUMOR DETECTED" if prediction[0][0] < 0.5 else "🟢 NO TUMOR DETECTED"
    st.subheader(result)