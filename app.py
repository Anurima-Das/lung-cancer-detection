import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page config ──
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="centered"
)

# ── Load model ──
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

model = load_model()

CLASS_NAMES = [
    'Adenocarcinoma',
    'Large Cell Carcinoma',
    'Normal',
    'Squamous Cell Carcinoma'
]

# ── Header ──
st.title("🫁 Lung Cancer Detection System")
st.markdown("### AI-powered CT Scan Analysis using VGG16 Deep Learning")
st.markdown("**Accuracy: 92.96% | Dataset: 1000 CT Scan Images**")
st.divider()

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This system uses VGG16 Transfer Learning to detect lung cancer from CT scan images.")
    st.divider()
    st.header("📊 Model Performance")
    st.metric("Accuracy", "92.96%")
    st.metric("Precision", "92.05%")
    st.metric("Recall", "92.17%")
    st.metric("F1-Score", "91.83%")
    st.divider()
    st.header("🔬 Classes")
    st.write("• Adenocarcinoma")
    st.write("• Large Cell Carcinoma")
    st.write("• Squamous Cell Carcinoma")
    st.write("• Normal")

# ── Main UI ──
uploaded_file = st.file_uploader(
    "📤 Upload a CT Scan Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a chest CT scan image for analysis"
)

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded CT Scan")
        st.image(image, use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing CT scan..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    with col2:
        st.subheader("🔬 Analysis Result")

        if predicted_class == 'Normal':
            st.success(f"🟢 **{predicted_class}**")
            st.markdown("**No cancer detected** in this CT scan.")
            st.markdown("The lung appears normal and healthy.")
        else:
            st.error(f"🔴 **{predicted_class} Detected**")
            st.markdown("**Abnormality detected.** Please consult a specialist immediately.")

        st.divider()
        st.metric("Confidence Score", f"{confidence:.2f}%")

        # Progress bars for all classes
        st.subheader("📊 Class Probabilities")
        for cls, prob in zip(CLASS_NAMES, predictions[0]):
            st.progress(
                float(prob),
                text=f"{cls}: {prob*100:.2f}%"
            )

    st.divider()

    # Results summary box
    st.subheader("📋 Prediction Summary")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Prediction", predicted_class)
    with col4:
        st.metric("Confidence", f"{confidence:.2f}%")
    with col5:
        status = "Healthy" if predicted_class == "Normal" else "Abnormal"
        st.metric("Status", status)

    st.divider()
    st.caption("⚠️ Disclaimer: This tool is for research and educational purposes only. Always consult a qualified medical professional for diagnosis.")

else:
    # Instructions when no image uploaded
    st.info("👆 Please upload a chest CT scan image to begin analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload any chest CT scan image in JPG or PNG format")
    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.write("AI model analyzes the image automatically")
    with col3:
        st.markdown("### 3️⃣ Result")
        st.write("Get instant prediction with confidence score")