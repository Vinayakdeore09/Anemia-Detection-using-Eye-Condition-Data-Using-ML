import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_preprocess

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):
    img = Image.open(image)
    img = img.resize((180, 180))  # Resize the image
    img = img_preprocess.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
# Function to load the pre-trained model and make predictions
def predict_anemia(image):
    model = tf.keras.models.load_model('anemia.h5')
    prediction = model.predict(image)
    return prediction

# Streamlit app
def main():
    st.title("Anemia Detection")
    st.write("Upload an image to detect anemia.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image = load_and_preprocess_image(uploaded_file)
        prediction = predict_anemia(image)
        
        if prediction > 1.5:
            st.write("Prediction: Person has anemia.")
        else:
            st.write("Prediction: Person does not have anemia.")

if _name_ == '_main_':
    main()