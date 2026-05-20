import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page Config
st.set_page_config(
    page_title="CNN AI Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# Load Model
model = load_model("final_image_classifier.keras")

# Class Names
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# Sidebar
st.sidebar.title("⚙️ Settings")

show_probabilities = st.sidebar.checkbox(
    "Show Probabilities",
    value=True
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0,
    100,
    20
)

# Main Title
st.title("🧠 CNN AI Image Classifier")

st.markdown("""
Upload an image and let the CNN model classify it.

### Supported Classes
- ✈️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐶 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚚 truck
""")

# File Upload
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# Prediction Section
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🖼 Uploaded Image")

        img = Image.open(uploaded_file)

        st.image(img, use_container_width=True)

    with col2:

        st.subheader("🤖 Prediction Result")

        # Convert RGB
        img = img.convert("RGB")

        # Resize
        img = img.resize((96, 96))

        # Convert to Array
        img_array = image.img_to_array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add Batch Dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict Button
        if st.button("🔍 Predict"):

            with st.spinner("Analyzing Image..."):

                prediction = model.predict(img_array)

                predicted_index = np.argmax(prediction)

                predicted_class = class_names[predicted_index]

                confidence = float(np.max(prediction) * 100)

                if confidence >= confidence_threshold:

                    st.success(
                        f"✅ Prediction: {predicted_class.upper()}"
                    )

                    st.info(
                        f"📊 Confidence: {confidence:.2f}%"
                    )

                else:

                    st.warning(
                        "⚠️ Confidence below threshold"
                    )

                # Probability Display
                if show_probabilities:

                    st.subheader("📈 Prediction Probabilities")

                    probability_dict = {}

                    for i in range(len(class_names)):
                        probability_dict[class_names[i]] = float(prediction[0][i])

                    st.bar_chart(probability_dict)

# Footer
st.markdown("---")

st.markdown(
    "Made with ❤️ using Streamlit and TensorFlow"
)

st.markdown(
    "Created by Devanshi Baraskar"
)
