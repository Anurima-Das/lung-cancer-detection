import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import datetime
import pandas as pd
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as ReportlabImage
from reportlab.lib.units import inch

# ── Page config ──
st.set_page_config(
    page_title="MediScan AI — Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #00D4FF;
    --secondary: #7B2FFF;
    --danger: #FF4757;
    --success: #2ED573;
    --warning: #FFA502;
    --dark: #0A0E1A;
    --card: #111827;
    --border: rgba(0, 212, 255, 0.15);
    --text: #E2E8F0;
    --muted: #64748B;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1526 50%, #0A0E1A 100%);
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2332);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}

.metric-card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* Hero section */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF, #7B2FFF, #FF4757);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--muted);
    margin-bottom: 24px;
}

/* Result cards */
.result-normal {
    background: linear-gradient(135deg, rgba(46, 213, 115, 0.1), rgba(46, 213, 115, 0.05));
    border: 1px solid rgba(46, 213, 115, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.result-cancer {
    background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), rgba(255, 71, 87, 0.05));
    border: 1px solid rgba(255, 71, 87, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.result-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-normal .result-title { color: var(--success); }
.result-cancer .result-title { color: var(--danger); }

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-cyan {
    background: rgba(0, 212, 255, 0.15);
    color: var(--primary);
    border: 1px solid rgba(0, 212, 255, 0.3);
}

.badge-purple {
    background: rgba(123, 47, 255, 0.15);
    color: #A78BFA;
    border: 1px solid rgba(123, 47, 255, 0.3);
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin: 24px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: 12px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(0, 212, 255, 0.03) !important;
    border: 2px dashed rgba(0, 212, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 255, 0.4) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3) !important;
}

/* Download button */
.stDownloadButton button {
    background: linear-gradient(135deg, #2ED573, #00B4D8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #1a2332) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Progress bars */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    border-radius: 99px !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: var(--primary) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* Info boxes */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    color: var(--text) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* Animated gradient border */
@keyframes borderFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glow-card {
    background: #111827;
    border-radius: 16px;
    padding: 2px;
    background: linear-gradient(135deg, #00D4FF, #7B2FFF, #FF4757, #00D4FF);
    background-size: 300% 300%;
    animation: borderFlow 4s ease infinite;
}

.glow-card-inner {
    background: #111827;
    border-radius: 14px;
    padding: 20px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.2); }
}

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}

.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--muted); }
.stat-value {
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


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

CLASS_INFO = {
    'Adenocarcinoma': {
        'desc': 'Most common type of lung cancer. Starts in mucus-secreting glands.',
        'color': '#FF4757',
        'icon': '🔴'
    },
    'Large Cell Carcinoma': {
        'desc': 'Fast-growing cancer that can appear in any part of the lung.',
        'color': '#FF6B35',
        'icon': '🟠'
    },
    'Normal': {
        'desc': 'No signs of malignancy detected. Lung tissue appears healthy.',
        'color': '#2ED573',
        'icon': '🟢'
    },
    'Squamous Cell Carcinoma': {
        'desc': 'Arises from squamous cells lining the airways of the lung.',
        'color': '#FFA502',
        'icon': '🟡'
    }
}

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

def generate_pdf_report(original_img, heatmap, overlaid_img,
                        predicted, confidence, preds):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('T', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#00D4FF'), spaceAfter=5)
    sub_style = ParagraphStyle('S', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#64748B'))

    story.append(Paragraph("MediScan AI", title_style))
    story.append(Paragraph("Lung Cancer Detection Report", styles['Title']))
    story.append(Spacer(1, 8))

    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    story.append(Paragraph("Generated: " + now, sub_style))
    story.append(Paragraph("Model: VGG16 Transfer Learning | Accuracy: 92.96%", sub_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("━" * 80, styles['Normal']))
    story.append(Spacer(1, 10))

    result_color = colors.HexColor('#FF4757') if predicted != 'Normal' else colors.HexColor('#2ED573')
    result_style = ParagraphStyle('R', parent=styles['Heading1'],
        fontSize=18, textColor=result_color)

    status = "⚠ ABNORMAL — Malignancy Detected" if predicted != 'Normal' else "✓ NORMAL — No Malignancy Detected"
    story.append(Paragraph("DIAGNOSIS RESULT", styles['Heading1']))
    story.append(Spacer(1, 5))
    story.append(Paragraph(status, result_style))
    story.append(Spacer(1, 8))

    info = CLASS_INFO.get(predicted, {})
    story.append(Paragraph("Classification: " + predicted, styles['Heading2']))
    story.append(Paragraph("Confidence Score: " + str(round(confidence, 2)) + "%", styles['Heading2']))
    if info.get('desc'):
        story.append(Paragraph(info['desc'], styles['Normal']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Class Probability Distribution", styles['Heading2']))
    story.append(Spacer(1, 5))

    cls_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
    table_data = [['Class', 'Probability (%)', 'Result']]
    for cls, prob in zip(cls_names, preds[0]):
        s = '✓ DETECTED' if prob == max(preds[0]) else '—'
        table_data.append([cls, f"{prob*100:.2f}%", s])

    table = Table(table_data, colWidths=[3*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0D1526')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00D4FF')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1a2332')),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#111827'), colors.HexColor('#0f172a')]),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#E2E8F0')),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Visual Analysis", styles['Heading2']))
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
        Paragraph('Grad-CAM Overlay (Highlighted Regions)', styles['Normal'])
    ]]

    img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1a2332')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#111827')),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("━" * 80, styles['Normal']))
    disclaimer_style = ParagraphStyle('D', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#64748B'))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by MediScan AI for research and "
        "educational purposes only. This does NOT constitute a medical diagnosis. "
        "Always consult a qualified medical professional for proper diagnosis and treatment.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem;'>🫁</div>
        <div style='font-size:1.3rem; font-weight:700; 
                    background: linear-gradient(135deg, #00D4FF, #7B2FFF);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;'>
            MediScan AI
        </div>
        <div style='font-size:0.75rem; color:#64748B; margin-top:4px;'>
            <span class='status-dot'></span> System Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "🔬  CT Scan Analysis",
        "📊  Model Comparison",
        "📈  Statistics",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.7rem; color:#64748B; text-transform:uppercase; 
                letter-spacing:0.1em; margin-bottom:12px;'>
        Model Performance
    </div>
    """, unsafe_allow_html=True)

    metrics = [
        ("Accuracy", "92.96%"),
        ("Precision", "92.05%"),
        ("Recall", "92.17%"),
        ("F1-Score", "91.83%"),
        ("AUC Score", "0.991"),
    ]
    for label, value in metrics:
        st.markdown(f"""
        <div class='stat-row'>
            <span class='stat-label'>{label}</span>
            <span class='stat-value'>{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#64748B; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:12px;'>
        Detectable Classes
    </div>
    """, unsafe_allow_html=True)

    for cls, info in CLASS_INFO.items():
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; 
                    padding:6px 0; font-size:0.85rem;'>
            <span>{info['icon']}</span>
            <span style='color:#E2E8F0;'>{cls}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#64748B; text-align:center;'>
        Powered by VGG16 + Grad-CAM<br>
        Final Year Project 2026-2027
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════
# PAGE 1 — CT SCAN ANALYSIS
# ════════════════════════════════════
if "CT Scan Analysis" in page:

    # Hero
    st.markdown("""
    <div style='padding: 32px 0 24px;'>
        <div class='hero-title'>Lung Cancer Detection</div>
        <div class='hero-subtitle'>
            Upload a chest CT scan for instant AI-powered analysis with 
            Grad-CAM explainability and downloadable medical report
        </div>
        <span class='badge badge-cyan'>VGG16 Transfer Learning</span>&nbsp;
        <span class='badge badge-purple'>Grad-CAM XAI</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_display = [
        (col1, "92.96%", "Accuracy"),
        (col2, "92.05%", "Precision"),
        (col3, "92.17%", "Recall"),
        (col4, "91.83%", "F1-Score"),
        (col5, "1000", "Training Images"),
    ]
    for col, val, label in metrics_display:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    st.markdown("<div class='section-header'>📤 Upload CT Scan</div>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop a chest CT scan image here",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = PILImage.open(uploaded_file).convert('RGB')
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔬 Analyzing CT scan with AI..."):
            preds = model.predict(img_array, verbose=0)
            heatmap = get_gradcam_heatmap(img_array, model, gradcam_model)
            overlaid = overlay_heatmap(image, heatmap)

        predicted = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100
        info = CLASS_INFO[predicted]

        # Result banner
        is_normal = predicted == 'Normal'
        result_class = 'result-normal' if is_normal else 'result-cancer'
        result_icon = '✅' if is_normal else '⚠️'
        result_status = 'No Malignancy Detected' if is_normal else 'Malignancy Detected — Consult Specialist'

        st.markdown(f"""
        <div class='{result_class}' style='margin: 20px 0;'>
            <div style='font-size:2.5rem; margin-bottom:8px;'>{result_icon}</div>
            <div class='result-title'>{predicted}</div>
            <div style='color: #94A3B8; font-size:0.95rem;'>{result_status}</div>
            <div style='margin-top:12px; font-size:2rem; font-weight:700; 
                        font-family: JetBrains Mono, monospace;
                        color:{"#2ED573" if is_normal else "#FF4757"};'>
                {confidence:.1f}% confidence
            </div>
            <div style='margin-top:8px; color:#64748B; font-size:0.85rem;'>
                {info["desc"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Images
        st.markdown("<div class='section-header'>🖼️ Visual Analysis</div>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📷 Original CT Scan**")
            st.image(image, use_container_width=True)
            st.caption("Input chest CT scan image")

        with col2:
            st.markdown("**🔥 Grad-CAM Heatmap**")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            im = ax.imshow(heatmap, cmap='jet')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Activation Map', color='white', fontsize=10, pad=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("Red regions = AI focus area")

        with col3:
            st.markdown("**🎯 AI Detection Overlay**")
            st.image(overlaid, use_container_width=True)
            st.caption("Heatmap superimposed on CT scan")

        # Probability bars
        st.markdown("<div class='section-header'>📊 Class Probabilities</div>",
                    unsafe_allow_html=True)

        prob_cols = st.columns(4)
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds[0])):
            with prob_cols[i]:
                cls_info = CLASS_INFO[cls]
                is_top = prob == max(preds[0])
                st.markdown(f"""
                <div class='metric-card' style='{"border-color: " + cls_info["color"] + ";" if is_top else ""}'>
                    <div style='font-size:1.8rem; font-weight:700;
                                color:{cls_info["color"]};
                                font-family: JetBrains Mono, monospace;'>
                        {prob*100:.1f}%
                    </div>
                    <div style='font-size:0.8rem; color:#94A3B8; margin-top:4px;'>
                        {cls_info["icon"]} {cls}
                    </div>
                    {"<div style='margin-top:8px;'><span style='background:rgba(46,213,115,0.15); color:#2ED573; border:1px solid rgba(46,213,115,0.3); border-radius:99px; padding:2px 8px; font-size:0.7rem;'>DETECTED</span></div>" if is_top else ""}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Prediction", predicted)
        with col2:
            st.metric("📊 Confidence", f"{confidence:.2f}%")
        with col3:
            st.metric("🏥 Status", "Healthy" if is_normal else "Abnormal")

        # PDF Report
        st.markdown("<div class='section-header'>📄 Medical Report</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.15);
                    border-radius:12px; padding:16px; margin-bottom:16px;'>
            <div style='color:#E2E8F0; font-size:0.95rem;'>
                Generate a professional PDF medical report containing the diagnosis result, 
                class probabilities, CT scan images, and Grad-CAM analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Generate Medical PDF Report"):
            with st.spinner("Generating report..."):
                pdf_buffer = generate_pdf_report(
                    image, heatmap, overlaid,
                    predicted, confidence, preds
                )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name="mediscan_report_" +
                          datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
                mime="application/pdf"
            )
            st.success("✅ Report generated successfully!")

        st.markdown("---")
        st.caption("⚠️ MediScan AI is for research purposes only. Not a substitute for professional medical diagnosis.")

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; 
                    background: rgba(0,212,255,0.02);
                    border: 2px dashed rgba(0,212,255,0.15);
                    border-radius:20px; margin-top:24px;'>
            <div style='font-size:4rem; margin-bottom:16px;'>🫁</div>
            <div style='font-size:1.3rem; font-weight:600; color:#E2E8F0; margin-bottom:8px;'>
                Upload a CT Scan to Begin
            </div>
            <div style='color:#64748B; max-width:400px; margin:0 auto; line-height:1.7;'>
                Our AI model will analyze your chest CT scan and provide 
                instant classification with Grad-CAM visual explanation
            </div>
            <div style='margin-top:24px; display:flex; justify-content:center; gap:16px;'>
                <span class='badge badge-cyan'>✓ 92.96% Accuracy</span>
                <span class='badge badge-purple'>✓ Grad-CAM XAI</span>
                <span class='badge badge-cyan'>✓ PDF Report</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How it works
        st.markdown("<div class='section-header' style='margin-top:40px;'>⚡ How It Works</div>",
                    unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        steps = [
            ("01", "Upload", "Upload any chest CT scan in JPG or PNG format"),
            ("02", "Analyze", "VGG16 model analyzes image features"),
            ("03", "Visualize", "Grad-CAM highlights cancer regions"),
            ("04", "Report", "Download professional PDF diagnosis"),
        ]
        for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem; font-weight:800; 
                                color: rgba(0,212,255,0.3);
                                font-family: JetBrains Mono;'>
                        {num}
                    </div>
                    <div style='font-weight:600; color:#E2E8F0; margin:8px 0 4px;'>
                        {title}
                    </div>
                    <div style='font-size:0.8rem; color:#64748B; line-height:1.5;'>
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════
elif "Model Comparison" in page:

    st.markdown("""
    <div style='padding: 32px 0 24px;'>
        <div class='hero-title'>Model Comparison</div>
        <div class='hero-subtitle'>
            Traditional Machine Learning vs Deep Learning Performance Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    model_metrics = [
        (col1, "SVM", "54.50%", "-38.46%", "#FF4757"),
        (col2, "Random Forest", "64.00%", "-28.96%", "#FFA502"),
        (col3, "Baseline CNN", "20.83%", "-72.13%", "#FF6B35"),
        (col4, "VGG16 (Ours)", "92.96%", "🏆 Best", "#2ED573"),
    ]
    for col, name, acc, diff, color in model_metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-color: {color}22;'>
                <div style='font-size:0.75rem; color:#64748B; 
                            text-transform:uppercase; letter-spacing:0.05em;'>
                    {name}
                </div>
                <div style='font-size:2rem; font-weight:700; color:{color};
                            font-family: JetBrains Mono, monospace; margin:8px 0;'>
                    {acc}
                </div>
                <div style='font-size:0.8rem; color:#64748B;'>{diff}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Performance Chart", "📋 Detailed Table", "🔍 Key Insights"])

    with tab1:
        models_list = ['SVM', 'Random Forest', 'Baseline CNN', 'VGG16 (Ours)']
        accuracy  = [54.50, 64.00, 20.83, 92.96]
        precision = [45.42, 63.15, 15.00, 92.05]
        recall    = [54.50, 64.00, 20.83, 92.17]
        f1        = [47.12, 63.29, 18.00, 91.83]

        x = np.arange(len(models_list))
        width = 0.2

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#0A0E1A')

        for ax in axes:
            ax.set_facecolor('#111827')
            ax.tick_params(colors='#64748B')
            ax.spines['bottom'].set_color('#1a2332')
            ax.spines['left'].set_color('#1a2332')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        bar_colors = ['#FF4757', '#A78BFA', '#FFA502', '#00D4FF']
        b1 = axes[0].bar(x - 1.5*width, accuracy, width,
                         label='Accuracy', color='#00D4FF', alpha=0.9)
        b2 = axes[0].bar(x - 0.5*width, precision, width,
                         label='Precision', color='#7B2FFF', alpha=0.9)
        b3 = axes[0].bar(x + 0.5*width, recall, width,
                         label='Recall', color='#2ED573', alpha=0.9)
        b4 = axes[0].bar(x + 1.5*width, f1, width,
                         label='F1-Score', color='#FFA502', alpha=0.9)

        axes[0].set_xlabel('Models', color='#64748B')
        axes[0].set_ylabel('Score (%)', color='#64748B')
        axes[0].set_title('Performance Metrics Comparison',
                         color='#E2E8F0', fontsize=12, pad=15)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models_list, color='#64748B')
        axes[0].legend(facecolor='#111827', edgecolor='#1a2332',
                      labelcolor='#E2E8F0')
        axes[0].set_ylim(0, 110)
        axes[0].axhline(y=90, color='#00D4FF', linestyle='--',
                       alpha=0.3, label='90% threshold')
        axes[0].yaxis.label.set_color('#64748B')
        axes[0].grid(True, alpha=0.1, axis='y')

        for bar, acc_val in zip(b1, accuracy):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{acc_val:.0f}', ha='center', va='bottom',
                        fontsize=7, color='#94A3B8')

        model_colors_bar = ['#FF4757', '#FFA502', '#FF6B35', '#2ED573']
        for i, (m, a, c) in enumerate(zip(models_list, accuracy, model_colors_bar)):
            axes[1].bar(i, a, color=c, alpha=0.9, width=0.6,
                       edgecolor=c, linewidth=2)
            axes[1].text(i, a + 1.5, f'{a}%', ha='center',
                        fontsize=11, fontweight='bold', color=c)

        axes[1].set_title('Accuracy Comparison', color='#E2E8F0',
                         fontsize=12, pad=15)
        axes[1].set_ylabel('Accuracy (%)', color='#64748B')
        axes[1].set_xticks(range(len(models_list)))
        axes[1].set_xticklabels(models_list, color='#64748B')
        axes[1].set_ylim(0, 110)
        axes[1].axhline(y=90, color='#00D4FF', linestyle='--',
                       alpha=0.3, label='90% line')
        axes[1].legend(facecolor='#111827', edgecolor='#1a2332',
                      labelcolor='#E2E8F0')
        axes[1].grid(True, alpha=0.1, axis='y')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        comparison_data = {
            'Model': ['SVM', 'Random Forest', 'Baseline CNN', 'VGG16 Transfer Learning'],
            'Type': ['Traditional ML', 'Traditional ML', 'Deep Learning', 'Deep Learning'],
            'Accuracy': ['54.50%', '64.00%', '20.83%', '92.96% 🏆'],
            'Precision': ['45.42%', '63.15%', '15.00%', '92.05%'],
            'Recall': ['54.50%', '64.00%', '20.83%', '92.17%'],
            'F1-Score': ['47.12%', '63.29%', '18.00%', '91.83%'],
            'CV Score': ['56.80%', '63.80%', 'N/A', '~92%']
        }
        df_show = pd.DataFrame(comparison_data)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.success("🏆 VGG16 Transfer Learning outperforms all models by 28.96%+")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.15);
                        border-radius:16px; padding:20px;'>
                <div style='color:#00D4FF; font-weight:600; margin-bottom:12px; font-size:1rem;'>
                    ✅ Why VGG16 Outperforms
                </div>
                <ul style='color:#94A3B8; line-height:2; list-style:none; padding:0;'>
                    <li>→ Pre-trained on 1.2M ImageNet images</li>
                    <li>→ Automatically learns spatial features</li>
                    <li>→ Transfer learning overcomes small dataset</li>
                    <li>→ 138M parameters capture fine details</li>
                    <li>→ Batch normalization for stability</li>
                    <li>→ Dropout prevents overfitting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background: rgba(255,71,87,0.05); border: 1px solid rgba(255,71,87,0.15);
                        border-radius:16px; padding:20px;'>
                <div style='color:#FF4757; font-weight:600; margin-bottom:12px; font-size:1rem;'>
                    ❌ Why Traditional ML Falls Short
                </div>
                <ul style='color:#94A3B8; line-height:2; list-style:none; padding:0;'>
                    <li>→ SVM uses only 4 GLCM texture features</li>
                    <li>→ Cannot learn spatial patterns in CT scans</li>
                    <li>→ Random Forest limited by feature quality</li>
                    <li>→ Manual feature engineering loses information</li>
                    <li>→ No end-to-end learning from raw pixels</li>
                    <li>→ Cannot capture hierarchical features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════
# PAGE 3 — STATISTICS
# ════════════════════════════════════
elif "Statistics" in page:

    st.markdown("""
    <div style='padding: 32px 0 24px;'>
        <div class='hero-title'>Project Statistics</div>
        <div class='hero-subtitle'>
            Comprehensive analysis of model performance and dataset distribution
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    st.markdown("<div class='section-header'>📦 Dataset Overview</div>",
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    dataset_stats = [
        (col1, "1,000", "Total Images"),
        (col2, "4", "Classes"),
        (col3, "800", "Training Images"),
        (col4, "200", "Validation Images"),
        (col5, "224×224", "Image Size"),
    ]
    for col, val, label in dataset_stats:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>🗂️ Class Distribution</div>",
                    unsafe_allow_html=True)
        classes = ['Adenocarcinoma', 'Squamous Cell', 'Normal', 'Large Cell']
        counts = [338, 260, 215, 187]
        colors_pie = ['#FF4757', '#FFA502', '#2ED573', '#FF6B35']

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        wedges, texts, autotexts = ax.pie(
            counts, labels=classes, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.6, edgecolor='#0A0E1A', linewidth=2)
        )
        for text in texts:
            text.set_color('#94A3B8')
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Dataset Class Distribution',
                    color='#E2E8F0', fontsize=11, pad=15)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>📊 Per-Class Performance</div>",
                    unsafe_allow_html=True)
        cls_perf = {
            'Class': ['Adenocarcinoma', 'Large Cell', 'Normal', 'Squamous Cell'],
            'Precision': [0.8732, 0.8780, 0.9773, 0.9535],
            'Recall': [0.9254, 0.9730, 1.0000, 0.7885],
            'F1-Score': [0.8986, 0.9231, 0.9885, 0.8632],
            'Support': [67, 37, 43, 52]
        }
        df_perf = pd.DataFrame(cls_perf)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')

        x = np.arange(len(df_perf['Class']))
        width = 0.25
        ax.bar(x - width, df_perf['Precision'], width,
               label='Precision', color='#00D4FF', alpha=0.9)
        ax.bar(x, df_perf['Recall'], width,
               label='Recall', color='#7B2FFF', alpha=0.9)
        ax.bar(x + width, df_perf['F1-Score'], width,
               label='F1-Score', color='#2ED573', alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(['Adeno', 'Large Cell', 'Normal', 'Squamous'],
                           color='#64748B', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title('Per-Class Metrics', color='#E2E8F0', fontsize=11, pad=15)
        ax.legend(facecolor='#111827', edgecolor='#1a2332', labelcolor='#E2E8F0')
        ax.tick_params(colors='#64748B')
        ax.spines['bottom'].set_color('#1a2332')
        ax.spines['left'].set_color('#1a2332')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.1, axis='y')
        ax.yaxis.label.set_color('#64748B')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # AUC scores
    st.markdown("<div class='section-header'>🎯 AUC Scores by Class</div>",
                unsafe_allow_html=True)

    auc_data = [
        ('Normal', 1.000, '#2ED573'),
        ('Large Cell Carcinoma', 0.998, '#00D4FF'),
        ('Squamous Cell Carcinoma', 0.988, '#FFA502'),
        ('Adenocarcinoma', 0.979, '#FF4757'),
    ]

    for cls_name, auc_val, color in auc_data:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style='margin: 6px 0;'>
                <div style='display:flex; justify-content:space-between; 
                            margin-bottom:4px;'>
                    <span style='color:#94A3B8; font-size:0.9rem;'>{cls_name}</span>
                    <span style='color:{color}; font-family: JetBrains Mono; 
                                font-weight:600;'>{auc_val:.3f}</span>
                </div>
                <div style='background:#1a2332; border-radius:99px; height:8px;'>
                    <div style='background: linear-gradient(90deg, {color}, {color}88);
                                width:{auc_val*100}%; height:100%; border-radius:99px;
                                transition: width 0.5s ease;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div style='padding: 32px 0 24px;'>
        <div class='hero-title'>About MediScan AI</div>
        <div class='hero-subtitle'>
            Final Year Project — AI-Powered Medical Diagnosis System
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #111827, #1a2332);
                    border: 1px solid rgba(0,212,255,0.15);
                    border-radius:16px; padding:24px; margin-bottom:20px;'>
            <div style='color:#00D4FF; font-size:1.1rem; font-weight:600; margin-bottom:12px;'>
                🎯 Project Overview
            </div>
            <div style='color:#94A3B8; line-height:1.9;'>
                MediScan AI is an advanced deep learning system that automatically 
                detects lung cancer from chest CT scan images. Using VGG16 Transfer 
                Learning trained on 1000 CT scan images, the system achieves 92.96% 
                accuracy in classifying 4 types of lung conditions.
                <br><br>
                The system features Grad-CAM explainability — showing radiologists 
                exactly which regions of the CT scan influenced the AI decision, 
                making it a transparent and trustworthy diagnostic aid.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: linear-gradient(135deg, #111827, #1a2332);
                    border: 1px solid rgba(123,47,255,0.15);
                    border-radius:16px; padding:24px;'>
            <div style='color:#A78BFA; font-size:1.1rem; font-weight:600; margin-bottom:12px;'>
                🛠️ Technology Stack
            </div>
            <div style='display:grid; grid-template-columns: 1fr 1fr; gap:12px;'>
        """, unsafe_allow_html=True)

        tech_items = [
            ("🧠 Model", "VGG16 Transfer Learning"),
            ("⚡ Framework", "TensorFlow / Keras"),
            ("👁️ XAI", "Grad-CAM Visualization"),
            ("🎨 Interface", "Streamlit"),
            ("📄 Reports", "ReportLab PDF"),
            ("🔬 CV", "OpenCV + scikit-image"),
            ("📊 Analysis", "Pandas + Matplotlib"),
            ("☁️ Training", "Google Colab GPU"),
        ]

        cols = st.columns(4)
        for i, (tech, val) in enumerate(tech_items):
            with cols[i % 4]:
                st.markdown(f"""
                <div style='background:#0D1526; border:1px solid rgba(0,212,255,0.1);
                            border-radius:10px; padding:10px; text-align:center;
                            margin-bottom:8px;'>
                    <div style='font-size:0.75rem; color:#64748B;'>{tech}</div>
                    <div style='font-size:0.85rem; color:#E2E8F0; font-weight:500; 
                                margin-top:4px;'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #111827, #1a2332);
                    border: 1px solid rgba(46,213,115,0.15);
                    border-radius:16px; padding:24px;'>
            <div style='color:#2ED573; font-size:1.1rem; font-weight:600; margin-bottom:16px;'>
                🏆 Key Achievements
            </div>
        """, unsafe_allow_html=True)

        achievements = [
            ("92.96%", "Model Accuracy"),
            ("4/4", "Test Predictions"),
            ("1.000", "Normal AUC"),
            ("0.991", "Avg AUC Score"),
            ("1000", "Training Images"),
            ("4", "Cancer Classes"),
        ]

        for val, label in achievements:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; 
                        padding:10px 0; border-bottom:1px solid rgba(0,212,255,0.08);'>
                <span style='color:#64748B; font-size:0.85rem;'>{label}</span>
                <span style='color:#2ED573; font-family: JetBrains Mono; 
                            font-weight:600;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

    # Team
    st.markdown("<div class='section-header' style='margin-top:32px;'>👥 Team Members</div>",
                unsafe_allow_html=True)

    team = [
        ("Anurima Das", "M3", "Deep Learning & App Dev", "VGG16, Grad-CAM, Streamlit", "#00D4FF"),
        ("Anushka De", "M2", "ML Models & Evaluation", "SVM, Random Forest, Metrics", "#7B2FFF"),
        ("Siddheeka Ray", "M1", "Data & Preprocessing", "Dataset, Preprocessing, Segmentation", "#2ED573"),
        ("Ankit Kumar", "M4", "Literature & Presentation", "Research Papers, Slides", "#FFA502"),
        ("Avinaw Shah", "M5", "Report Writing", "Project Report, Documentation", "#FF4757"),
    ]

    cols = st.columns(5)
    for col, (name, role_num, role, contrib, color) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-color:{color}22; text-align:left;'>
                <div style='width:36px; height:36px; border-radius:50%;
                            background: linear-gradient(135deg, {color}33, {color}11);
                            border: 2px solid {color}44;
                            display:flex; align-items:center; justify-content:center;
                            font-size:0.75rem; font-weight:700; color:{color};
                            margin-bottom:10px;'>
                    {role_num}
                </div>
                <div style='font-weight:600; color:#E2E8F0; font-size:0.9rem;'>
                    {name}
                </div>
                <div style='color:{color}; font-size:0.75rem; margin:4px 0;'>
                    {role}
                </div>
                <div style='color:#64748B; font-size:0.75rem; line-height:1.5;'>
                    {contrib}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#64748B; font-size:0.85rem; padding:16px 0;'>
        ⚠️ MediScan AI is developed for research and educational purposes only.<br>
        This system does not replace professional medical diagnosis.<br><br>
        <span style='color:#00D4FF;'>Final Year Project — B.Tech Computer Science(Cybersecurity including IOT and Blockchain Technology) 2026–2027</span>
    </div>
    """, unsafe_allow_html=True)
