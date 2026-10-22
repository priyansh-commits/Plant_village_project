
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# --------------------------------
# Configuration
# --------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "plant_disease_finetuned.keras"
CLASSES_PATH = APP_DIR / "class_names.json"

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------
# Load saved model and class names
# --------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------
# Disease name formatting
# --------------------------------
def format_class_name(name):
    name = name.replace("___", " - ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")
    return " ".join(name.split())


# --------------------------------
# Prediction
# --------------------------------
def predict_image(image, model, class_names):
    # The model expects RGB images of size 224x224.
    # MobileNetV2 preprocessing is already inside the model.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returns {len(probabilities)} outputs, "
            f"but {len(class_names)} class names were loaded."
        )

    top_indices = np.argsort(probabilities)[::-1][:3]

    return [
        {
            "class_name": class_names[int(i)],
            "confidence": float(probabilities[i])
        }
        for i in top_indices
    ]


# --------------------------------
# UI
# --------------------------------
st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear photo of a plant leaf to predict "
    "which of the 15 PlantVillage categories it most resembles."
)

st.info(
    "This model was trained on PlantVillage images. "
    "Predictions on real-world photos may be less accurate."
)

# Check that the files exist
if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Model files not found. Ensure these files are in "
        "the same folder as app.py:\n\n"
        "- plant_disease_finetuned.keras\n"
        "- class_names.json"
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Could not load the model or class names: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button("Detect Plant Disease", type="primary",
                     use_container_width=True):
            with st.spinner("Analyzing the image..."):
                results = predict_image(
                    image,
                    model,
                    class_names
                )

            best = results[0]

            st.subheader("Prediction")
            st.success(format_class_name(best["class_name"]))
            st.metric(
                "Model confidence",
                f"{best['confidence']:.2%}"
            )

            st.subheader("Top 3 predictions")

            for rank, result in enumerate(results, start=1):
                label = format_class_name(result["class_name"])
                confidence = result["confidence"]

                st.write(f"**{rank}. {label}**")
                st.progress(
                    max(0.0, min(1.0, confidence)),
                    text=f"{confidence:.2%}"
                )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the diagnosis is correct."
            )

    except (UnidentifiedImageError, OSError):
        st.error("This image could not be read. Try another image.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()
st.caption(
    "Built with TensorFlow, MobileNetV2 and Streamlit. "
    "For educational purposes; not a substitute for expert diagnosis."
)
