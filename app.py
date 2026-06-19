import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("dog_cat_model.keras")
    return model

model = load_my_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))

    img_array = np.array(image, dtype=np.float32)

    # Match training preprocessing
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ----------------------------
# UI
# ----------------------------
st.title("🐶 Dog vs 🐱 Cat Classifier")
st.write("Upload an image and the model will predict whether it contains a dog or a cat.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            processed_image = preprocess_image(image)

            prediction = model.predict(
                processed_image,
                verbose=0
            )[0][0]

            confidence = float(prediction)

        st.subheader("Prediction Result")

        # Threshold = 0.3
        if confidence > 0.3:

            st.success("🐶 DOG")

            st.metric(
                label="Dog Confidence",
                value=f"{confidence*100:.2f}%"
            )

        else:

            st.success("🐱 CAT")

            st.metric(
                label="Cat Confidence",
                value=f"{(1-confidence)*100:.2f}%"
            )

        st.write("Brought To You By @KrishRathod")
        