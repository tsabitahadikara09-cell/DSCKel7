import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from modules.dataset import IMAGE_DIR, CSV_PATH, load_dataset_index, class_summary, sample_image
from modules.preprocessing import preprocess_image, METHOD_DESCRIPTIONS
from modules.training import train_model, MODEL_PATH
from modules.evaluation import evaluate_model

st.set_page_config(page_title="DermaCNN | ISIC 2018", page_icon="🧬", layout="wide")

st.markdown("""<style>
.stApp{background:#111019;color:#f3efff}.block-container{padding-top:1.3rem}
[data-testid='stSidebar']{background:#191521}.metric-card{background:#211d2b;border:1px solid #393243;padding:18px;border-radius:14px}
div.stButton>button{border-radius:10px}h1,h2,h3{color:#f4efff}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset_index()

df, dataset_error = get_data()
counts = class_summary(df)



def show_image(image, caption=None):
    if image is not None:
        st.image(image, caption=caption, use_container_width=True)



def sidebar():
    st.sidebar.markdown("# 🧬 DermaCNN")
    st.sidebar.caption("Skin Lesion Classification | ISIC 2018")
    pages = ["Home", "Dataset", "Filtering", "Segmentation & Masking", "Enhancement", "Hair Removal", "Grayscale", "Training (CNN)", "Evaluation"]
    page = st.sidebar.radio("Navigasi", pages, label_visibility="visible")
    st.sidebar.divider()
    st.sidebar.markdown("### Dataset Status")
    if dataset_error:
        st.sidebar.warning("Dataset belum terbaca")
        st.sidebar.caption(dataset_error)
    else:
        st.sidebar.success(f"Dataset terbaca: {len(df):,} gambar")
        st.sidebar.caption(f"Kelas terdeteksi: {df.label.nunique()} / 3")
    if st.sidebar.button("🔄 Refresh Dataset", use_container_width=True):
        get_data.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("DermaCNN | Biomedical Deep Learning Project")
    st.sidebar.caption("ISIC 2018 • MEL • NV • BKL")
    return page


page = sidebar()


def page_home():
    st.title("🧬 DermaCNN")
    st.subheader("Skin Lesion Classification Dashboard")
    st.write("Klasifikasi citra lesi kulit menggunakan dataset ISIC 2018 dan CNN. Dataset dibaca otomatis dari direktori lokal.")
    if dataset_error:
        st.error(dataset_error)
        st.info("Pastikan folder Training Input dan file Ground Truth tersedia pada lokasi path yang telah ditentukan di modules/dataset.py.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", f"{len(df):,}")
    c2.metric("Total Classes", df.label.nunique())
    c3.metric("MEL", f"{counts['MEL']:,}")
    c4.metric("NV", f"{counts['NV']:,}")
    st.markdown("### Distribusi Kelas")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values())
    ax.set_ylabel("Jumlah gambar")
    ax.set_title("ISIC 2018 — MEL, NV, BKL")
    st.pyplot(fig)
    st.markdown("### Sampel Citra")
    cols = st.columns(3)
    for col, label in zip(cols, ["MEL", "NV", "BKL"]):
        image, image_id = sample_image(df, label)
        with col:
            st.markdown(f"**{label}**")
            show_image(image, image_id)


def page_dataset():
    st.title("📁 Dataset Explorer")
    st.caption("ISIC 2018 — Training Dataset (kelas MEL, NV, BKL)")
    if dataset_error:
        st.warning("Dataset belum terbaca")
        st.error(dataset_error)
        st.info("Path dataset dan CSV sudah ditentukan otomatis di modules/dataset.py. Path laptop tidak akan tersedia pada Streamlit Cloud.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total gambar", f"{len(df):,}")
    c2.metric("Jumlah kelas", df.label.nunique())
    c3.metric("MEL", f"{counts['MEL']:,}")
    c4.metric("NV + BKL", f"{counts['NV'] + counts['BKL']:,}")
    st.subheader("Distribusi Dataset")
    chart_df = pd.DataFrame({"Kelas": list(counts.keys()), "Jumlah": list(counts.values())})
    st.bar_chart(chart_df.set_index("Kelas"))
    st.subheader("Contoh gambar per kelas")
    cols = st.columns(3)
    for col, label in zip(cols, ["MEL", "NV", "BKL"]):
        img, imgid = sample_image(df, label)
        with col:
            st.markdown(f"**{label}**")
            show_image(img, imgid)
    with st.expander("Lihat indeks dataset"):
        st.dataframe(df.head(100), use_container_width=True)


def page_preprocess(method):
    st.title(method)
    st.write(METHOD_DESCRIPTIONS.get(method, "Pemrosesan citra."))
    if dataset_error:
        st.error(dataset_error)
        return
    label = st.selectbox("Pilih kelas", ["MEL", "NV", "BKL"])
    subset = df[df.label == label]
    if subset.empty:
        st.warning("Tidak ada gambar pada kelas ini.")
        return
    idx = st.slider("Pilih indeks gambar", 0, len(subset)-1, 0)
    row = subset.iloc[idx]
    original = Image.open(row.path).convert("RGB")
    kwargs = {}
    if method == "Filtering":
        ftype = st.selectbox("Jenis filter", ["Median", "Gaussian"])
        kernel = st.slider("Kernel size", 3, 15, 5, step=2)
        kwargs.update(filter_type=ftype, kernel=kernel)
    elif method == "Enhancement":
        kwargs["clip_limit"] = st.slider("Contrast limit", 1.0, 5.0, 2.0, 0.5)
    result = preprocess_image(original, method, **kwargs)
    st.caption(f"Image ID: {row.image_id} • Class: {row.label}")
    if isinstance(result, tuple):
        processed, mask = result
        a, b, c = st.columns(3)
        with a: show_image(original, "Original")
        with b: show_image(processed, "Processed")
        with c: show_image(mask, "Mask")
    else:
        a, b = st.columns(2)
        with a: show_image(original, "Original")
        with b: show_image(result, method)


def page_training():
    st.title("🧠 Training CNN")
    st.write("Latih model CNN menggunakan tiga kelas: MEL, NV, dan BKL.")
    if dataset_error:
        st.error(dataset_error)
        return
    with st.form("training_config"):
        method = st.selectbox("Metode preprocessing", ["Filtering", "Enhancement", "Grayscale"])
        epochs = st.slider("Epoch", 1, 50, 10)
        batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
        image_size = st.selectbox("Image size", [128, 160, 224], index=2)
        limit = st.number_input("Maksimum gambar per kelas (0 = semua)", min_value=0, max_value=5000, value=300, step=50)
        start = st.form_submit_button("🚀 Mulai Training", type="primary")
    if start:
        progress = st.progress(0)
        status = st.empty()
        status.info("Training berjalan. Untuk training yang panjang, proses mungkin memerlukan waktu.")
        try:
            model, history, val_data, class_names = train_model(df, method, epochs, batch_size, image_size, int(limit), progress_callback=None)
            progress.progress(100)
            st.success(f"Training selesai. Model tersimpan: {MODEL_PATH}")
            hist = history.history
            st.subheader("Training Curves")
            metrics = pd.DataFrame({k: v for k, v in hist.items()})
            st.line_chart(metrics[[c for c in ["accuracy", "val_accuracy"] if c in metrics.columns]])
            st.line_chart(metrics[[c for c in ["loss", "val_loss"] if c in metrics.columns]])
            st.session_state["last_model"] = model
            st.session_state["val_data"] = val_data
            st.session_state["class_names"] = class_names
            st.success("Model dan data validasi siap digunakan di halaman Evaluation.")
        except Exception as exc:
            st.error(f"Training gagal: {exc}")


def page_evaluation():
    st.title("📊 Evaluation")
    if "last_model" not in st.session_state or "val_data" not in st.session_state:
        st.info("Jalankan Training terlebih dahulu agar data evaluasi tersedia.")
        if MODEL_PATH.exists():
            st.caption("Model tersimpan ditemukan, tetapi data validasi dari training sebelumnya tidak disimpan. Jalankan training ulang untuk evaluasi lengkap.")
        return
    model = st.session_state["last_model"]
    X_val, y_val = st.session_state["val_data"]
    names = st.session_state.get("class_names", ["MEL", "NV", "BKL"])
    try:
        results = evaluate_model(model, X_val, y_val, names)
        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        report = pd.DataFrame(results["report"]).T
        st.subheader("Classification Report")
        st.dataframe(report, use_container_width=True)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(results["confusion_matrix"])
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, results["confusion_matrix"][i, j], ha="center", va="center")
        st.pyplot(fig)
        st.download_button("Download report CSV", report.to_csv().encode("utf-8"), "classification_report.csv", "text/csv")
    except Exception as exc:
        st.error(f"Evaluasi gagal: {exc}")


if page == "Home":
    page_home()
elif page == "Dataset":
    page_dataset()
elif page in METHOD_DESCRIPTIONS:
    page_preprocess(page)
elif page == "Training (CNN)":
    page_training()
elif page == "Evaluation":
    page_evaluation()
