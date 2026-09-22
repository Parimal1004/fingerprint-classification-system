import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "../models/best_fingerprint_model.keras"
CONFUSION_MATRIX_PATH = "../results/confusion_matrix.png"

CLASS_NAMES = ["Arch", "Whorl", "Loop"]
IMAGE_SIZE = (224, 224)

TEST_ACCURACY = 87.65


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fingerprint Classification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1100px 520px at 8% -8%, rgba(139, 92, 246, 0.38), transparent 55%),
            radial-gradient(900px 420px at 100% 0%, rgba(6, 182, 212, 0.28), transparent 52%),
            radial-gradient(800px 380px at 70% 100%, rgba(236, 72, 153, 0.22), transparent 50%),
            #0b1024;
        color: #f5f7ff;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151a3c 0%, #0e1430 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.28);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8faff !important;
    }

    .main-header {
        text-align: center;
        padding: 1.6rem 1.2rem 1.2rem 1.2rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(6, 182, 212, 0.16) 50%, rgba(236, 72, 153, 0.18));
        border: 1px solid rgba(196, 181, 253, 0.28);
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.35);
        margin-bottom: 1.2rem;
    }

    .main-header h1 {
        font-size: 2.55rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, #c4b5fd, #67e8f9, #f9a8d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        font-size: 1.08rem;
        color: #dbe4ff;
        opacity: 0.92;
        margin-bottom: 0.9rem;
    }

    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.55rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-block;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #0b1024;
    }

    .badge-violet { background: #c4b5fd; }
    .badge-cyan { background: #67e8f9; }
    .badge-pink { background: #f9a8d4; }
    .badge-amber { background: #fcd34d; }

    .notice {
        padding: 0.95rem 1.1rem;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.18), rgba(236, 72, 153, 0.12));
        border: 1px solid rgba(251, 191, 36, 0.35);
        color: #fde68a;
        margin: 0.4rem 0 1.1rem 0;
        font-weight: 500;
    }

    .section-title {
        font-size: 1.42rem;
        font-weight: 750;
        margin-top: 1.4rem;
        margin-bottom: 0.85rem;
        color: #eef2ff;
        padding-left: 0.7rem;
        border-left: 4px solid #a78bfa;
    }

    .info-card {
        padding: 1.2rem 1.15rem;
        border-radius: 18px;
        min-height: 165px;
        color: #f8faff;
        box-shadow: 0 12px 30px rgba(2, 6, 23, 0.22);
    }

    .info-card h4 {
        margin-bottom: 0.45rem;
        font-size: 1.12rem;
    }

    .card-violet {
        background: linear-gradient(160deg, rgba(139, 92, 246, 0.42), rgba(67, 56, 202, 0.22));
        border: 1px solid rgba(196, 181, 253, 0.32);
    }

    .card-cyan {
        background: linear-gradient(160deg, rgba(6, 182, 212, 0.38), rgba(14, 116, 144, 0.2));
        border: 1px solid rgba(103, 232, 249, 0.3);
    }

    .card-pink {
        background: linear-gradient(160deg, rgba(236, 72, 153, 0.38), rgba(157, 23, 77, 0.2));
        border: 1px solid rgba(249, 168, 212, 0.32);
    }

    .card-amber {
        background: linear-gradient(160deg, rgba(245, 158, 11, 0.28), rgba(180, 83, 9, 0.16));
        border: 1px solid rgba(252, 211, 77, 0.3);
    }

    .card-emerald {
        background: linear-gradient(160deg, rgba(16, 185, 129, 0.3), rgba(6, 95, 70, 0.16));
        border: 1px solid rgba(110, 231, 183, 0.28);
    }

    .prediction-card {
        padding: 1.7rem 1.2rem;
        border-radius: 22px;
        text-align: center;
        margin: 0.7rem 0 1.1rem 0;
        color: #fff;
        box-shadow: 0 16px 40px rgba(76, 29, 149, 0.28);
    }

    .pred-arch {
        background: linear-gradient(135deg, #06b6d4, #2563eb);
    }

    .pred-whorl {
        background: linear-gradient(135deg, #8b5cf6, #db2777);
    }

    .pred-loop {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
    }

    .prediction-label {
        font-size: 0.98rem;
        opacity: 0.92;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .prediction-class {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }

    .prediction-confidence {
        font-size: 1.35rem;
        font-weight: 650;
    }

    .sidebar-chip {
        display: inline-block;
        margin: 0.18rem 0.18rem 0.18rem 0;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        font-weight: 650;
        font-size: 0.82rem;
        color: #0b1024;
    }

    .chip-arch { background: #67e8f9; }
    .chip-whorl { background: #c4b5fd; }
    .chip-loop { background: #f9a8d4; }

    .method-card {
        padding: 1rem 1.05rem;
        border-radius: 16px;
        margin-bottom: 0.75rem;
        color: #eef2ff;
    }

    .method-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 1.02rem;
    }

    .footer {
        text-align: center;
        padding: 1.4rem 1rem;
        margin-top: 1.4rem;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.18), rgba(6, 182, 212, 0.14));
        border: 1px solid rgba(196, 181, 253, 0.22);
        color: #dbe4ff;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(196, 181, 253, 0.22);
        border-radius: 16px;
        padding: 0.7rem 0.85rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(22, 27, 58, 0.72);
        border: 1px dashed rgba(167, 139, 250, 0.55);
        border-radius: 16px;
        padding: 0.6rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #db2777) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.32);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(236, 72, 153, 0.32);
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(167, 139, 250, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_fingerprint_model():
    return load_model(MODEL_PATH)


model = load_fingerprint_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_fingerprint(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Add channel dimension
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = (
        predictions[predicted_index] * 100
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔍 Fingerprint AI")

    st.markdown(
        '<p style="color:#c4b5fd;margin-top:-0.6rem;">Pattern detection studio</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("About the Model")

    st.write(
        "A convolutional neural network trained "
        "to classify fingerprint patterns."
    )

    st.markdown(
        """
        <div>
            <span class="sidebar-chip chip-arch">Arch</span>
            <span class="sidebar-chip chip-whorl">Whorl</span>
            <span class="sidebar-chip chip-loop">Loop</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Model Performance")

    st.metric(
        "Test Accuracy",
        f"{TEST_ACCURACY:.2f}%"
    )

    st.metric(
        "Test Images",
        "243"
    )

    st.markdown("---")

    st.caption(
        "Deep Learning • TensorFlow • CNN"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>Fingerprint Classification System</h1>
        <p>Deep learning based fingerprint pattern detection</p>
        <div class="badge-row">
            <span class="badge badge-violet">CNN</span>
            <span class="badge badge-cyan">TensorFlow</span>
            <span class="badge badge-pink">Arch · Whorl · Loop</span>
            <span class="badge badge-amber">87.65% Accuracy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="notice">
        This system classifies fingerprint <b>patterns</b> only
        (Arch, Whorl, or Loop). It does <b>not</b> identify a person's identity.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📌 Project Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card card-violet">
            <h4>🧠 Deep Learning</h4>
            Uses a Convolutional Neural Network
            to automatically learn fingerprint
            pattern features.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card card-cyan">
            <h4>📷 Image Processing</h4>
            Input images are converted to
            grayscale, resized to 224×224,
            and normalized before prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card card-pink">
            <h4>🎯 Three Classes</h4>
            The model distinguishes between
            Arch, Whorl, and Loop fingerprint
            patterns.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Fingerprint</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a fingerprint image",
    type=["png", "jpg", "jpeg"],
    help="Upload a fingerprint image in PNG, JPG, or JPEG format."
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown(
        '<div class="section-title">🖼️ Input Image</div>',
        unsafe_allow_html=True
    )

    image_col1, image_col2 = st.columns(
        [1, 1]
    )

    with image_col1:

        st.image(
            image,
            caption="Uploaded Fingerprint",
            width=420
        )

    with image_col2:

        st.markdown(
            f"""
            <div class="info-card card-violet">
                <h4>Image Information</h4>
                <p><b>Filename:</b> {uploaded_file.name}</p>
                <p><b>Original Size:</b> {image.width} × {image.height}</p>
                <p><b>Color Mode:</b> {image.mode}</p>
                <p><b>Model Input:</b> 224 × 224 grayscale</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        classify_button = st.button(
            "🔍 Classify Fingerprint",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if classify_button:

        predicted_class, confidence, predictions = (
            predict_fingerprint(image)
        )

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🎯 Classification Result</div>',
            unsafe_allow_html=True
        )

        pred_style = {
            "Arch": "pred-arch",
            "Whorl": "pred-whorl",
            "Loop": "pred-loop"
        }.get(predicted_class, "pred-whorl")

        st.markdown(
            f"""
            <div class="prediction-card {pred_style}">
                <div class="prediction-label">Predicted Fingerprint Pattern</div>
                <div class="prediction-class">{predicted_class}</div>
                <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # PROBABILITY BAR CHART
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Class Probabilities</div>',
            unsafe_allow_html=True
        )

        probability_data = {
            "Fingerprint Pattern": CLASS_NAMES,
            "Probability (%)": [
                float(p * 100)
                for p in predictions
            ]
        }

        st.bar_chart(
            probability_data,
            x="Fingerprint Pattern",
            y="Probability (%)",
            horizontal=True
        )

        # Individual probability values
        prob_col1, prob_col2, prob_col3 = st.columns(3)

        for column, class_name, probability in zip(
            [prob_col1, prob_col2, prob_col3],
            CLASS_NAMES,
            predictions
        ):

            with column:

                st.metric(
                    class_name,
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

perf1, perf2, perf3, perf4 = st.columns(4)

with perf1:
    st.metric(
        "Test Accuracy",
        f"{TEST_ACCURACY:.2f}%"
    )

with perf2:
    st.metric(
        "Test Images",
        "243"
    )

with perf3:
    st.metric(
        "Classes",
        "3"
    )

with perf4:
    st.metric(
        "Input Size",
        "224 × 224"
    )


# ============================================================
# CLASS PERFORMANCE
# ============================================================

st.markdown(
    "### Class-wise Performance"
)

class_performance = {
    "Class": [
        "Arch",
        "Whorl",
        "Loop"
    ],
    "Precision (%)": [
        92.31,
        85.39,
        85.53
    ],
    "Recall (%)": [
        88.89,
        93.83,
        80.25
    ],
    "F1 Score (%)": [
        90.57,
        89.41,
        82.80
    ]
}

st.dataframe(
    class_performance,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔬 Confusion Matrix</div>',
    unsafe_allow_html=True
)

st.write(
    "The confusion matrix shows the number of correctly "
    "and incorrectly classified fingerprints for each class."
)

try:

    st.image(
        CONFUSION_MATRIX_PATH,
        caption="Fingerprint Classification Confusion Matrix",
        width=650
    )

except Exception:

    st.info(
        "Confusion matrix image is not available."
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">ℹ️ Methodology</div>',
    unsafe_allow_html=True
)

method_col1, method_col2 = st.columns(2)

with method_col1:

    st.markdown(
        """
        <div class="method-card card-violet">
            <h4>1. Data Collection</h4>
            Fingerprint images from the NIST DB4-based
            dataset are organized into Arch, Whorl,
            and Loop classes.
        </div>
        <div class="method-card card-cyan">
            <h4>2. Preprocessing</h4>
            Images are converted to grayscale,
            resized to 224 × 224 pixels, and
            normalized to the range 0–1.
        </div>
        <div class="method-card card-pink">
            <h4>3. Data Augmentation</h4>
            Rotation, zoom, and translation are
            applied during training to improve
            model generalization.
        </div>
        """,
        unsafe_allow_html=True
    )

with method_col2:

    st.markdown(
        """
        <div class="method-card card-amber">
            <h4>4. CNN Architecture</h4>
            The model uses multiple convolutional
            layers with Batch Normalization and
            Max Pooling.
        </div>
        <div class="method-card card-emerald">
            <h4>5. Classification</h4>
            A Softmax output layer produces
            probabilities for the three fingerprint
            patterns.
        </div>
        <div class="method-card card-violet">
            <h4>6. Evaluation</h4>
            The model was evaluated on a separate
            test set using accuracy, precision,
            recall, F1-score, and a confusion matrix.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL DETAILS
# ============================================================

st.markdown(
    "### 🧠 Model Details"
)

details_col1, details_col2 = st.columns(2)

with details_col1:

    st.markdown(
        """
        | Component | Value |
        |---|---|
        | Framework | TensorFlow / Keras |
        | Architecture | CNN |
        | Optimizer | Adam |
        | Loss | Sparse Categorical Crossentropy |
        | Input | 224 × 224 × 1 |
        """
    )

with details_col2:

    st.markdown(
        """
        | Component | Value |
        |---|---|
        | Output Classes | 3 |
        | Classes | Arch, Whorl, Loop |
        | Test Accuracy | 87.65% |
        | Hardware | NVIDIA RTX 3050 |
        | Training | GPU Accelerated |
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Fingerprint Classification System · Deep Learning Project
        <br><br>
        Pattern classification only — not biometric identity recognition.
    </div>
    """,
    unsafe_allow_html=True
)
