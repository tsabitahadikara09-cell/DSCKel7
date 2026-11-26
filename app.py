import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dataset import load_dataset_index, class_summary, sample_image
from preprocessing import preprocess_image, METHOD_DESCRIPTIONS
from training import train_model
from evaluation import evaluate_model

st.set_page_config(page_title="DermaCNN | ISIC 2018", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.stApp { background:#111019; color:#f3efff; }
[data-testid="stSidebar"] { background:#191521; }
h1,h2,h3 { color:#d7b8ff; }
[data-testid="metric-container"] { background:#201a2c; border:1px solid #4b3767; padding:14px; border-radius:12px; }
div.stButton > button { border-radius:9px; }
</style>
""", unsafe_allow_html=True)

CLASS_INFO = {
    "MEL": ("Melanoma", "Lesi melanoma"),
    "NV": ("Melanocytic Nevus", "Nevus / tahi lalat"),
    "BKL": ("Benign Keratosis", "Keratosis jinak"),
}
METHODS = ["Original", "Filtering", "Segmentation & Masking", "Enhancement", "Hair Removal", "Grayscale"]

st.sidebar.title("🧬 DermaCNN")
st.sidebar.caption("Skin Lesion Classification | ISIC 2018")
page = st.sidebar.radio("Navigasi", [
    "Home", "Dataset", "Filtering", "Segmentation & Masking",
    "Enhancement", "Hair Removal", "Grayscale", "Training (CNN)", "Evaluation"
])
st.sidebar.divider()
st.sidebar.subheader("Dataset Configuration")
images_dir = st.sidebar.text_input("Folder gambar", value="dataset/images")
csv_path = st.sidebar.text_input("Ground Truth CSV", value="dataset/ISIC2018_Task3_Training_GroundTruth.csv")
if st.sidebar.button("🔄 Refresh Dataset", use_container_width=True):
    st.cache_data.clear()

@st.cache_data(show_spinner="Membaca dataset dan label...")
def cached_dataset(folder, labels):
    return load_dataset_index(folder, labels)

try:
    df = cached_dataset(images_dir, csv_path) if (images_dir and csv_path) else pd.DataFrame()
    dataset_error = None
except Exception as e:
    df = pd.DataFrame()
    dataset_error = str(e)

if df.empty:
    st.sidebar.warning("Dataset belum ditemukan / belum terbaca.")
else:
    st.sidebar.success(f"Dataset terbaca: {len(df):,} gambar")

def require_data():
    if df.empty:
        st.warning("Dataset belum terbaca. Pastikan folder gambar dan CSV label tersedia pada path di sidebar.")
        st.info("Untuk deploy Streamlit Cloud, dataset harus tersedia di repository (jika ukurannya memungkinkan) atau storage yang di-mount/diakses server. Path laptop tidak dapat diakses dari cloud.")
        if dataset_error:
            st.caption(f"Detail: {dataset_error}")
        return False
    return True

def class_select():
    available = [c for c in ["MEL", "NV", "BKL"] if c in df.label.unique()]
    if not available:
        st.warning("Tidak ditemukan kelas MEL, NV, atau BKL pada CSV.")
        return None
    selected = st.selectbox("Pilih kelas", available, format_func=lambda x: f"{CLASS_INFO[x][0]} ({x})")
    return selected

def sample_selector(class_code):
    subset = df[df.label == class_code].reset_index(drop=True)
    if subset.empty:
        st.warning("Tidak ada gambar pada kelas ini.")
        return None, subset
    idx = st.slider("Pilih sampel", 0, len(subset)-1, 0, key=f"slider_{page}_{class_code}")
    return subset.iloc[idx], subset

def show_class_gallery(subset, limit=12):
    st.markdown("#### Galeri sampel")
    if subset.empty:
        st.info("Tidak ada sampel.")
        return
    gallery = subset.sample(min(limit, len(subset)), random_state=21)
    cols = st.columns(4)
    for i, row in enumerate(gallery.itertuples()):
        with cols[i % 4]:
            im = sample_image(row.path)
            if im is not None:
                st.image(im, caption=row.image_id, use_container_width=True)

def show_preprocess_page(method):
    st.title(f"🧪 {method}")
    st.write(METHOD_DESCRIPTIONS.get(method, "Eksperimen preprocessing citra lesi kulit."))
    if not require_data():
        return
    selected_class = class_select()
    if selected_class is None:
        return
    row, subset = sample_selector(selected_class)
    if row is None:
        return
    image = sample_image(row.path)
    if image is None:
        st.error("Gambar gagal dibaca.")
        return
    processed = preprocess_image(image, method)
    st.markdown("### Hasil preprocessing")
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption=f"Original | {row.image_id}", use_container_width=True)
    with c2:
        st.image(processed, caption=method, use_container_width=True)
    st.caption(f"ID: {row.image_id} • Kelas: {CLASS_INFO[selected_class][0]} • Resolusi: {image.shape[1]} × {image.shape[0]}")
    show_class_gallery(subset)

if page == "Home":
    st.title("🧬 DermaCNN")
    st.subheader("Skin Lesion Classification using CNN")
    st.write("Analisis komparatif pengaruh teknik preprocessing terhadap performa CNN pada dataset ISIC 2018.")
    if not df.empty:
        counts = class_summary(df)
        total = len(df)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total gambar", f"{total:,}")
        c2.metric("Melanoma (MEL)", f"{counts.get('MEL',0):,}")
        c3.metric("Nevus (NV)", f"{counts.get('NV',0):,}")
        c4.metric("Benign Keratosis (BKL)", f"{counts.get('BKL',0):,}")
        st.markdown("### Distribusi kelas")
        fig, ax = plt.subplots(figsize=(8,3.5))
        keys = [k for k in ["MEL","NV","BKL"] if k in counts]
        ax.bar([CLASS_INFO[k][0] for k in keys], [counts[k] for k in keys])
        ax.set_ylabel("Jumlah citra")
        ax.set_title("Dataset ISIC 2018 — subset 3 kelas")
        st.pyplot(fig)
    else:
        st.info("Atur folder gambar dan CSV Ground Truth di sidebar, kemudian refresh dataset.")
    st.markdown("### Pipeline Project")
    st.markdown("**Dataset → Preprocessing → Training CNN → Evaluasi → Perbandingan hasil**")
    st.markdown("### Metode Preprocessing")
    st.write("Filtering • Segmentation & Masking • Enhancement • Hair Removal • Grayscale")
    st.markdown("### Novelty & Contribution")
    st.write("Analisis komparatif pengaruh teknik preprocessing terhadap performa klasifikasi lesi kulit menggunakan CNN, dengan evaluasi metrik keseluruhan dan per kelas.")

elif page == "Dataset":
    st.title("📁 Dataset Explorer")
    if require_data():
        counts = class_summary(df)
        st.dataframe(pd.DataFrame([{"Kode": k, "Kelas": CLASS_INFO[k][0], "Jumlah": v} for k,v in counts.items()]), use_container_width=True)
        selected = class_select()
        if selected:
            subset = df[df.label == selected]
            st.metric("Jumlah data kelas", len(subset))
            show_class_gallery(subset, 16)

elif page in ["Filtering", "Segmentation & Masking", "Enhancement", "Hair Removal", "Grayscale"]:
    show_preprocess_page(page)

elif page == "Training (CNN)":
    st.title("🧠 Training CNN")
    st.write("Training CNN dari dataset hasil preprocessing. Untuk eksperimen yang adil, split train/validation/test harus konsisten antar metode.")
    if require_data():
        method = st.selectbox("Metode preprocessing", METHODS[1:])
        epochs = st.slider("Epoch", 1, 50, 10)
        batch = st.selectbox("Batch size", [8, 16, 32], index=1)
        img_size = st.selectbox("Image size", [128, 160, 224], index=2)
        limit = st.number_input("Maksimum gambar per kelas (0 = semua)", min_value=0, value=300, step=100)
        if st.button("🚀 Mulai Training", type="primary"):
            with st.spinner("Training berjalan..."):
                result = train_model(df, method, epochs, batch, img_size, int(limit))
            st.session_state["training_result"] = result
            st.success("Training selesai.")
        if "training_result" in st.session_state:
            result = st.session_state["training_result"]
            st.metric("Test Accuracy", f"{result['test_accuracy']:.4f}")
            fig, ax = plt.subplots()
            ax.plot(result["history"]["accuracy"], label="Train Accuracy")
            ax.plot(result["history"]["val_accuracy"], label="Validation Accuracy")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            st.pyplot(fig)
            fig, ax = plt.subplots()
            ax.plot(result["history"]["loss"], label="Train Loss")
            ax.plot(result["history"]["val_loss"], label="Validation Loss")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            st.pyplot(fig)
            st.write("Laporan evaluasi:")
            st.text(result["report"])
            st.session_state["latest_training"] = result

elif page == "Evaluation":
    st.title("📊 Evaluation")
    result = st.session_state.get("latest_training")
    if result is None:
        st.info("Jalankan training terlebih dahulu untuk melihat evaluasi.")
    else:
        evaluation = evaluate_model(result)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{evaluation['accuracy']:.4f}")
        c2.metric("Macro Precision", f"{evaluation['precision']:.4f}")
        c3.metric("Macro Recall", f"{evaluation['recall']:.4f}")
        c4.metric("Macro F1", f"{evaluation['f1']:.4f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(evaluation["confusion_matrix"])
        ax.set_xticks(range(3), labels=evaluation["class_names"], rotation=30)
        ax.set_yticks(range(3), labels=evaluation["class_names"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, evaluation["confusion_matrix"][i,j], ha="center", va="center")
        st.pyplot(fig)
        st.text(evaluation["report"])
