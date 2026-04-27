import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as ReportlabImage
from reportlab.lib.units import inch

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
        x = dense1(x)
        x = drop1(x, training=False)
        x = dense2(x)
        x = drop2(x, training=False)
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

def generate_pdf_report(original_img, heatmap, overlaid_img,
                        predicted, confidence, preds):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#1B4F72'),
        spaceAfter=10
    )
    story.append(Paragraph("Lung Cancer Detection AI", title_style))
    story.append(Paragraph("Medical Analysis Report", styles['Title']))
    story.append(Spacer(1, 10))

    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    story.append(Paragraph("Date: " + now, styles['Normal']))
    story.append(Paragraph("Model: VGG16 Transfer Learning", styles['Normal']))
    story.append(Paragraph("Accuracy: 92.96%", styles['Normal']))
    story.append(Spacer(1, 15))

    result_color = colors.red if predicted != 'Normal' else colors.green
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=result_color
    )
    status = "ABNORMAL - Cancer Detected" if predicted != 'Normal' else "NORMAL - No Cancer Detected"
    story.append(Paragraph("DIAGNOSIS RESULT", styles['Heading1']))
    story.append(Paragraph("Prediction: " + predicted, result_style))
    story.append(Paragraph("Confidence: " + str(round(confidence, 2)) + "%", styles['Heading2']))
    story.append(Paragraph("Status: " + status, styles['Heading2']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Class Probabilities", styles['Heading2']))
    story.append(Spacer(1, 5))

    cls_names = ['Adenocarcinoma', 'Large Cell Carcinoma',
                 'Normal', 'Squamous Cell Carcinoma']
    table_data = [['Cancer Type', 'Probability', 'Status']]
    for cls, prob in zip(cls_names, preds[0]):
        s = 'DETECTED' if prob == max(preds[0]) else '-'
        table_data.append([cls, str(round(prob*100, 2)) + '%', s])

    table = Table(table_data, colWidths=[3*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B4F72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#EBF5FB'), colors.white]),
    ]))
    story.append(table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("CT Scan Analysis Images", styles['Heading2']))
    story.append(Spacer(1, 10))

    orig_buffer = io.BytesIO()
    original_img.save(orig_buffer, format='PNG')
    orig_buffer.seek(0)

    overlay_pil = PILImage.fromarray(overlaid_img)
    overlay_buffer = io.BytesIO()
    overlay_pil.save(overlay_buffer, format='PNG')
    overlay_buffer.seek(0)

    img_table_data = [[
        ReportlabImage(orig_buffer, width=2.5*inch, height=2.5*inch),
        ReportlabImage(overlay_buffer, width=2.5*inch, height=2.5*inch)
    ], [
        Paragraph('Original CT Scan', styles['Normal']),
        Paragraph('Grad-CAM Overlay', styles['Normal'])
    ]]

    img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey
    )
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for "
        "research and educational purposes only. This is NOT a medical "
        "diagnosis. Always consult a qualified medical professional.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── SIDEBAR ──
with st.sidebar:
    st.title("🫁 Lung Cancer AI")
    st.divider()
    page = st.radio("Navigation", [
        "🔬 CT Scan Analysis",
        "📊 Model Comparison",
        "ℹ️ About"
    ])
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


# ════════════════════════════════════
# PAGE 1 — CT SCAN ANALYSIS
# ════════════════════════════════════
if page == "🔬 CT Scan Analysis":
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("### AI-powered CT Scan Analysis using VGG16 + Grad-CAM")
    st.markdown("**Accuracy: 92.96% | Dataset: 1000 CT Scan Images**")
    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload a CT Scan Image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = PILImage.open(uploaded_file).convert('RGB')

        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Analyzing CT scan..."):
            preds = model.predict(img_array, verbose=0)
            heatmap = get_gradcam_heatmap(img_array, model, gradcam_model)
            overlaid = overlay_heatmap(image, heatmap)

        predicted = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100

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

        if predicted == 'Normal':
            st.success("🟢 **Result: " + predicted + "**")
            st.markdown("No cancer detected in this CT scan.")
        else:
            st.error("🔴 **Result: " + predicted + " Detected**")
            st.markdown("Abnormality detected. Please consult a specialist.")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Prediction", predicted)
        with col5:
            st.metric("Confidence", str(round(confidence, 2)) + "%")
        with col6:
            status = "Healthy" if predicted == "Normal" else "Abnormal"
            st.metric("Status", status)

        st.divider()
        st.subheader("📊 All Class Probabilities")
        for cls, prob in zip(CLASS_NAMES, preds[0]):
            st.progress(
                float(prob),
                text=cls + ": " + str(round(prob*100, 2)) + "%"
            )

        st.divider()
        st.subheader("📄 Download Medical Report")
        if st.button("📥 Generate & Download PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_pdf_report(
                    image, heatmap, overlaid,
                    predicted, confidence, preds
                )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="lung_cancer_report_" +
                          datetime.datetime.now().strftime(
                              "%Y%m%d_%H%M%S") + ".pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF Report ready for download!")

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
            st.write("Get prediction + visual explanation + PDF report")


# ════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison Dashboard")
    st.markdown("### Traditional ML vs Deep Learning Performance")
    st.divider()

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SVM Accuracy", "56.00%", "-36.96%")
    with col2:
        st.metric("Random Forest", "68.50%", "-24.46%")
    with col3:
        st.metric("Baseline CNN", "20.83%", "-72.13%")
    with col4:
        st.metric("VGG16 (Ours)", "92.96%", "+92.96%")

    st.divider()

    tab1, tab2 = st.tabs(["📈 Chart", "📋 Table"])

    with tab1:
        try:
            comparison_img = PILImage.open("model_comparison.png")
            st.image(comparison_img, use_container_width=True)
        except:
            # Generate chart on the fly
            models = ['SVM', 'Random Forest', 'Baseline CNN', 'VGG16 (Ours)']
            accuracy  = [56.00, 68.50, 20.83, 92.96]
            precision = [46.93, 67.98, 15.00, 92.05]
            recall    = [56.00, 68.50, 20.83, 92.17]
            f1        = [48.67, 68.13, 18.00, 91.83]

            x = np.arange(len(models))
            width = 0.2

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',  color='#2E86AB')
            ax.bar(x - 0.5*width, precision, width, label='Precision', color='#A23B72')
            ax.bar(x + 0.5*width, recall,    width, label='Recall',    color='#F18F01')
            ax.bar(x + 1.5*width, f1,        width, label='F1-Score',  color='#C73E1D')
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        comparison_data = {
            'Model': ['SVM', 'Random Forest',
                      'Baseline CNN', 'VGG16 (Ours)'],
            'Type': ['Traditional ML', 'Traditional ML',
                     'Deep Learning', 'Deep Learning'],
            'Accuracy':  ['56.00%', '68.50%', '20.83%', '92.96%'],
            'Precision': ['46.93%', '67.98%', '15.00%', '92.05%'],
            'Recall':    ['56.00%', '68.50%', '20.83%', '92.17%'],
            'F1-Score':  ['48.67%', '68.13%', '18.00%', '91.83%']
        }
        df_show = pd.DataFrame(comparison_data)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.success("🏆 Best Model: VGG16 Transfer Learning — 92.96% Accuracy")

    st.divider()
    st.subheader("📝 Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Why VGG16 performs best:**
        - Pre-trained on 1.2M ImageNet images
        - Learns complex visual features automatically
        - Transfer learning overcomes small dataset limitation
        - Fine-tuned specifically for CT scan analysis
        """)
    with col2:
        st.markdown("""
        **Why traditional ML falls short:**
        - SVM uses only 4 GLCM texture features
        - Random Forest limited by feature quality
        - Cannot learn spatial patterns in CT scans
        - Deep learning extracts 1000s of features automatically
        """)


# ════════════════════════════════════
# PAGE 3 — ABOUT
# ════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.divider()

    st.markdown("""
    ## 🫁 Lung Cancer Detection AI
    
    This is a **Final Year Project** that uses deep learning to 
    automatically detect lung cancer from CT scan images.
    
    ### 🎯 Objective
    Build an AI system that assists doctors in early detection 
    of lung cancer with high accuracy and visual explainability.
    
    ### 🛠️ Technology Stack
    - **Model:** VGG16 Transfer Learning
    - **Framework:** TensorFlow / Keras
    - **Interface:** Streamlit
    - **Explainability:** Grad-CAM Visualization
    - **Report:** ReportLab PDF Generation
    
    ### 📊 Dataset
    - **Source:** Kaggle — Chest CT-Scan Images
    - **Total Images:** 1000
    - **Classes:** 4
    - **Split:** 80% Train / 20% Validation
    """)

    st.divider()
    st.subheader("👥 Team Members")
    team_data = {
        'Member': ['Anurima Das', 'Member 2', 'Member 3',
                   'Member 4', 'Member 5'],
        'Role': ['Deep Learning & App Development',
                 'ML Models & Evaluation',
                 'Data & Preprocessing',
                 'Literature Review & Presentation',
                 'Report Writing'],
        'Contribution': ['VGG16, Grad-CAM, Streamlit App',
                        'SVM, Random Forest, Metrics',
                        'Dataset, Preprocessing, Segmentation',
                        'Research Papers, Slides',
                        'Project Report, Documentation']
    }
    df_team = pd.DataFrame(team_data)
    st.dataframe(df_team, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📈 Model Results")
    results_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC (Normal)'],
        'Score': ['92.96%', '92.05%', '92.17%', '91.83%', '1.000']
    }
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("⚠️ For research and educational purposes only.")
