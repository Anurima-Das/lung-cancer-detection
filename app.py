import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide"
)

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

# Build Grad-CAM model
@st.cache_resource
def build_gradcam_model(_model):
    vgg16 = _model.get_layer('vgg16')
    gradcam = tf.keras.Model(
        inputs=vgg16.input,
        outputs=[
            vgg16.get_layer('block5_conv3').output,
            vgg16.output
        ]
    )
    return gradcam

gradcam_model = build_gradcam_model(model)

def get_gradcam_heatmap(img_array, model, gradcam_model):
    dense1 = model.get_layer('dense_12')
    drop1  = model.get_layer('dropout_8')
    dense2 = model.get_layer('dense_13')
    drop2  = model.get_layer('dropout_9')
    dense3 = model.get_layer('dense_14')

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, vgg_out = gradcam_model(img_tensor)
        tape.watch(conv_out)
        x = tf.reshape(vgg_out, [1, -1])
        x = dense1(x); x = drop1(x, training=False)
        x = dense2(x); x = drop2(x, training=False)
        preds = dense3(x)
        pred_idx = tf.argmax(preds[0])
        score = preds[:, pred_idx]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.5):
    img = np.array(img_pil.resize((224, 224)))
    h = cv2.resize(heatmap, (224, 224))
    h = np.uint8(255 * h)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img, 1-alpha, colored, alpha, 0)
    return overlaid

# ── UI ──
st.title("🫁 Lung Cancer Detection System")
st.markdown("### AI-powered CT Scan Analysis using VGG16 + Grad-CAM")
st.markdown("**Accuracy: 92.96% | Dataset: 1000 CT Scan Images**")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("VGG16 Transfer Learning with Grad-CAM visualization")
    st.divider()
    st.header("📊 Model Performance")
    st.metric("Accuracy",  "92.96%")
    st.metric("Precision", "92.05%")
    st.metric("Recall",    "92.17%")
    st.metric("F1-Score",  "91.83%")
    st.divider()
    st.header("🔬 Classes")
    st.write("• Adenocarcinoma")
    st.write("• Large Cell Carcinoma")
    st.write("• Squamous Cell Carcinoma")
    st.write("• Normal")

uploaded_file = st.file_uploader(
    "📤 Upload a CT Scan Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing CT scan..."):
        preds = model.predict(img_array, verbose=0)
        heatmap = get_gradcam_heatmap(img_array, model, gradcam_model)
        overlaid = overlay_heatmap(image, heatmap)

    predicted = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show results in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original CT Scan")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        ax.set_title('Red = Cancer Region', fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("🎯 AI Detection Overlay")
        st.image(overlaid, use_container_width=True)

    st.divider()

    # Result
    if predicted == 'Normal':
        st.success(f"🟢 **Result: {predicted}**")
        st.markdown("No cancer detected in this CT scan.")
    else:
        st.error(f"🔴 **Result: {predicted} Detected**")
        st.markdown("Abnormality detected. Please consult a specialist.")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Prediction", predicted)
    with col5:
        st.metric("Confidence", f"{confidence:.2f}%")
    with col6:
        status = "Healthy" if predicted == "Normal" else "Abnormal"
        st.metric("Status", status)

    st.divider()
    st.subheader("📊 All Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, preds[0]):
        st.progress(float(prob), text=f"{cls}: {prob*100:.2f}%")

    st.divider()
    st.caption("⚠️ For research purposes only. Consult a medical professional.")

else:
    st.info("👆 Upload a chest CT scan image to begin analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload CT scan image")
    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.write("AI analyzes + Grad-CAM highlights cancer region")
    with col3:
        st.markdown("### 3️⃣ Result")
        st.write("Get prediction + visual explanation")
