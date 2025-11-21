import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown 
from tensorflow.keras.models import load_model

MODEL_PATH = "model_daun.keras"
MODEL_URL = "https://drive.google.com/uc?id=1J0Kstvfh3dg1lo41xyFmsvZyo2YIldMW"

print("Files:", os.listdir("."))
print("Model path:", MODEL_PATH)

model = load_model(MODEL_PATH)

st.set_page_config(
    page_title="Deteksi Penyakit Tanaman",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
        }
        .sub {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f0f7f0;
            border: 1px solid #cce3cc;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


class_names = ["Healthy", "Rusty", "Powdery"]
colors = {"Healthy": "green", "Rusty": "orange", "Powdery": "purple"}

IMG_SIZE = 224

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.sidebar.title("ℹ️ Information")
st.sidebar.write("""
Upload gambar daun dan model akan memprediksi apakah daunnya:
- **Healthy**  
- **Rusty**  
- **Powdery**
""")
st.sidebar.write("Created by Bayu Ardi Putranto")


st.markdown("<h1 class='title'>🌿 Deteksi Penyakit Tanaman</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload gambar daun untuk mendeteksi kesehatannya</p>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📸 Upload gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img_processed = preprocess_image(img)

    # Predict
    prediction = model.predict(img_processed)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # ========== RESULT CARD ==========
    st.markdown(
        f"""
        <div class='result-box'>
            <h2 style='color:{colors[predicted_class]};text-align:center;'>🔍 Hasil Prediksi</h2>
            <h3 style='text-align:center;'>
                Kelas: <b style='color:{colors[predicted_class]};'>{predicted_class}</b>
            </h3>
            <p style='text-align:center;font-size:18px;'>Confidence: <b>{confidence:.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========== PROBABILITIES ==========
    st.write("### 📊 Probabilitas Kelas:")

    for cname, p in zip(class_names, prediction):
        st.write(f"**{cname}** — {p:.2f}")
        st.progress(float(p))
