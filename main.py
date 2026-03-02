import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE ----------------
def get_base64_image(image_file):
    with open(image_file, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_image = get_base64_image("Diseases.png")

st.markdown(f"""
<style>

.stApp {{
    background-image: url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Dark overlay for readability */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    z-index: -1;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ---------------- HEADER CARD ----------------
st.markdown("""
<div style="
    background: rgba(0, 0, 0, 0.65);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 30px;
">

<h1 style="color:#A5D6A7; font-size:42px; margin-bottom:15px;">
🌿 Plant Disease Detection System
</h1>

<p style="color:white; font-size:18px; margin-bottom:10px;">
This system uses Deep Learning (CNN) to detect plant diseases 
from leaf images for sustainable agriculture.
</p>

<p style="color:#E8F5E9; font-size:17px; font-weight:600;">
📌 Upload a leaf image below to predict the disease.
</p>

</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- UPLOAD SECTION ----------------
# Hide default ugly uploader UI parts
st.markdown("""
<style>

/* Hide drag & drop instructions */
div[data-testid="stFileUploaderDropzone"] > div {
    display: none;
}

/* Style uploader as button */
div[data-testid="stFileUploader"] {
    background-color: transparent;
}

div[data-testid="stFileUploader"] section {
    border: none;
}

/* Make upload button themed */
button[kind="secondary"] {
    background: linear-gradient(to right, #43A047, #2E7D32);
    color: white !important;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

button[kind="secondary"]:hover {
    background: linear-gradient(to right, #66BB6A, #388E3C);
}

</style>
""", unsafe_allow_html=True)

test_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")


if test_image is not None:
    st.markdown("<h3 style='color:white;'>Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(test_image, use_container_width=True)


# ---------------- PREDICTION BUTTON ----------------
if st.button("🚀 Predict Disease"):

    if test_image is None:
        st.markdown("""
<div style='
    background-color:#B71C1C;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:18px;
    font-weight:700;
    color:white;
    margin-top:15px;
'>
⚠ Please upload an image first.
</div>
""", unsafe_allow_html=True)

    else:
        with st.spinner("Analyzing image..."):
            result_index = model_prediction(test_image)

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        prediction = class_name[result_index]

        st.markdown(f"""
        <div style='
            background-color:#1B5E20;
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            color:white;
            margin-top:20px;
        '>
        🌱 Predicted Disease: {prediction}
        </div>
        """, unsafe_allow_html=True)
