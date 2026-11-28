
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from modules import dataset as ds

from modules.metadata import build_metadata

from modules.preprocessing import (
    METHOD_DESCRIPTIONS,
    preprocess_image
)

from modules.training import train_model, MODEL_PATH
from modules.evaluation import evaluate_model


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DermaCNN | ISIC 2018",
    page_icon="🧬",
    layout="wide"
)

CLASS_CODES = getattr(
    ds, "CLASS_CODES",
    getattr(ds, "CLASSES", ["MEL", "NV", "BKL"])
)

IMAGE_DIR = getattr(ds, "IMAGE_DIR", None)
CSV_PATH = getattr(ds, "CSV_PATH", None)

load_dataset_index = ds.load_dataset_index
class_summary = ds.class_summary
sample_image = ds.sample_image


# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #111019;
    color: #f3efff;
}

.block-container {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] {
    background-color: #191521;
}

h1, h2, h3 {
    color: #f4efff;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATASET LOADING
# ==========================================

@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset_index()


df, dataset_error = get_dataset()

counts = class_summary(df)


# ==========================================
# NAVIGATION
# ==========================================

PAGES = [
    "Home",
    "Dataset",
    "Metadata Dataset",
    "Filtering",
    "Segmentation & Masking",
    "Enhancement",
    "Hair Removal",
    "Grayscale",
    "Training (CNN)",
    "Evaluation"
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def show_image(image, caption):

    if image is not None:
        st.image(
            image,
            caption=caption,
            use_container_width=True
        )


def choose_image():

    if dataset_error or df.empty:
        st.error(
            dataset_error or "Dataset gambar kosong."
        )
        return None, None

    label = st.selectbox(
        "Pilih kelas",
        CLASS_CODES
    )

    subset = df[df["label"] == label]

    if subset.empty:
        st.warning(
            f"Belum ada gambar untuk kelas {label}."
        )
        return None, None

    idx = st.selectbox(
        "Pilih gambar",
        range(len(subset)),
        format_func=lambda i: subset.iloc[i]["image_id"]
    )

    row = subset.iloc[idx]

    try:
        image = Image.open(
            row["path"]
        ).convert("RGB")

        return image, row

    except Exception as exc:

        st.error(
            f"Gambar gagal dibuka: {exc}"
        )

        return None, None


# ==========================================
# HOME
# ==========================================

def page_home():

    st.title("🧬 DermaCNN")

    st.subheader(
        "Skin Lesion Classification | ISIC 2018"
    )

    st.write(
        "Eksplorasi dataset, preprocessing citra, "
        "training CNN, dan evaluasi untuk kelas "
        "MEL, NV, dan BKL."
    )

    if dataset_error:

        st.warning(dataset_error)

        st.info(
            "Metadata CSV tetap dapat dilihat pada "
            "halaman Metadata Dataset. "
            "Untuk preprocessing dan CNN, "
            "file gambar harus tersedia."
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Gambar terbaca", len(df))

    c2.metric(
        "Kelas",
        sum(v > 0 for v in counts.values())
    )

    c3.metric("MEL", counts["MEL"])

    c4.metric(
        "NV + BKL",
        counts["NV"] + counts["BKL"]
    )

    st.subheader("Distribusi Dataset")

    st.bar_chart(
        pd.DataFrame({
            "Jumlah": counts
        })
    )

    st.caption(f"Folder gambar: {IMAGE_DIR}")

    st.caption(
        f"Ground Truth CSV: {CSV_PATH}"
    )


# ==========================================
# DATASET EXPLORER
# ==========================================

def page_dataset():

    st.title("📁 Dataset Explorer")

    if df.empty:

        st.warning(
            dataset_error or "Tidak ada gambar."
        )

        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total gambar", len(df))

    c2.metric("MEL", counts["MEL"])

    c3.metric("NV", counts["NV"])

    c4.metric("BKL", counts["BKL"])

    st.subheader("Distribusi Kelas")

    st.bar_chart(
        pd.DataFrame({
            "Jumlah": counts
        })
    )

    st.subheader("Contoh Gambar")

    cols = st.columns(3)

    for col, code in zip(cols, CLASS_CODES):

        img, image_id = sample_image(df, code)

        with col:

            st.markdown(
                f"**{code} — {image_id or 'Tidak tersedia'}**"
            )

            show_image(img, code)

    st.subheader("Dataset Index")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# METADATA DATASET
# ==========================================

def page_metadata():

    st.title("🗂️ Metadata Dataset")

    metadata, error = build_metadata()

    if error:

        st.error(error)

        return

    st.caption(
        "Metadata label berasal dari CSV Ground Truth. "
        "Informasi gambar berasal dari file gambar yang ditemukan."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total metadata",
        len(metadata)
    )

    c2.metric(
        "MEL",
        int((metadata["label"] == "MEL").sum())
    )

    c3.metric(
        "NV",
        int((metadata["label"] == "NV").sum())
    )

    c4.metric(
        "BKL",
        int((metadata["label"] == "BKL").sum())
    )

    if "image_available" in metadata.columns:

        st.metric(
            "File gambar tersedia",
            int(metadata["image_available"].sum())
        )

    st.divider()

    st.subheader("Pencarian Metadata")

    query = st.text_input(
        "Cari Image ID"
    )

    filtered = metadata.copy()

    if query:

        filtered = filtered[
            filtered["image_id"].astype(str).str.contains(
                query,
                case=False,
                na=False
            )
        ]

    selected_classes = st.multiselect(
        "Filter kelas",
        CLASS_CODES,
        default=CLASS_CODES
    )

    filtered = filtered[
        filtered["label"].isin(selected_classes)
    ]

    st.write(
        f"Menampilkan {len(filtered)} dari {len(metadata)} data."
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Metadata CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "dermacnn_metadata.csv",
        "text/csv"
    )


# ==========================================
# PREPROCESSING
# ==========================================

def page_preprocessing(method):

    st.title(method)

    st.write(
        METHOD_DESCRIPTIONS.get(
            method,
            "Pemrosesan citra lesi kulit."
        )
    )

    image, row = choose_image()

    if image is None:
        return

    st.caption(
        f"Image ID: {row['image_id']} | "
        f"Class: {row['label']}"
    )

    kwargs = {}

    # -------------------------------
    # FILTERING
    # -------------------------------

    if method == "Filtering":

        filter_type = st.selectbox(
            "Pilih jenis filter",
            [
                "Mean",
                "Median",
                "Gaussian"
            ]
        )

        kernel = st.slider(
            "Kernel size",
            3, 15, 5, step=2
        )

        kwargs.update(
            filter_type=filter_type,
            kernel=kernel
        )

    # -------------------------------
    # ENHANCEMENT
    # -------------------------------

    elif method == "Enhancement":

        kwargs["clip_limit"] = st.slider(
            "Contrast limit",
            1.0, 5.0, 2.0, 0.5
        )

    # -------------------------------
    # HAIR REMOVAL
    # -------------------------------

    elif method == "Hair Removal":

        kwargs["kernel_size"] = st.slider(
            "Black-hat kernel",
            9, 31, 17, step=2
        )

        kwargs["inpaint_radius"] = st.slider(
            "Inpainting radius",
            1, 7, 3
        )

    # -------------------------------
    # PROCESS IMAGE
    # -------------------------------

    try:

        with st.spinner(
            f"Memproses gambar: {method}..."
        ):

            result = preprocess_image(
                image,
                method,
                **kwargs
            )

        st.divider()

        # -------------------------------
        # SEGMENTATION
        # -------------------------------

        if method == "Segmentation & Masking":

            if not isinstance(result, tuple) or len(result) != 2:

                st.error(
                    "Output segmentation harus berupa "
                    "(mask, segmented). Periksa preprocessing.py."
                )

                return

            mask, segmented = result

            st.subheader("Segmentation & Masking")

            a, b, c = st.columns(3)

            with a:
                show_image(
                    image,
                    "Original"
                )

            with b:
                show_image(
                    mask,
                    "Mask — Lesi putih, background hitam"
                )

            with c:
                show_image(
                    segmented,
                    "Segmented"
                )

        # -------------------------------
        # HAIR REMOVAL
        # -------------------------------

        elif method == "Hair Removal":

            if not isinstance(result, tuple) or len(result) != 3:

                st.error(
                    "Output Hair Removal harus berupa "
                    "(blackhat, mask, cleaned). "
                    "Periksa preprocessing.py."
                )

                return

            blackhat, mask, cleaned = result

            st.subheader("Hair Removal Process")

            a, b, c, d = st.columns(4)

            with a:
                show_image(
                    image,
                    "1. Original"
                )

            with b:
                show_image(
                    blackhat,
                    "2. Black Hat"
                )

            with c:
                show_image(
                    mask,
                    "3. Hair Mask"
                )

            with d:
                show_image(
                    cleaned,
                    "4. Hair Removed"
                )

        # -------------------------------
        # OTHER PREPROCESSING
        # -------------------------------

        else:

            a, b = st.columns(2)

            with a:
                show_image(
                    image,
                    "Original"
                )

            with b:
                show_image(
                    result,
                    "Filtered" if method == "Filtering"
                    else method
                )

    except Exception as exc:

        st.error(
            f"Terjadi kesalahan saat preprocessing: {exc}"
        )


# ==========================================
# TRAINING CNN
# ==========================================

def page_training():

    st.title("🧠 Training CNN")

    st.write(
        "Training menggunakan tiga kelas: MEL, NV, dan BKL."
    )

    if dataset_error or df.empty:

        st.error(
            dataset_error or "Dataset gambar kosong."
        )

        return

    with st.form("training_config"):

        method = st.selectbox(
            "Preprocessing",
            [
                "Original",
                "Filtering",
                "Enhancement",
                "Grayscale"
            ]
        )

        epochs = st.slider(
            "Epoch",
            1, 30, 8
        )

        batch_size = st.selectbox(
            "Batch size",
            [8, 16, 32],
            index=1
        )

        image_size = st.selectbox(
            "Image size",
            [96, 128, 160, 224],
            index=1
        )

        limit = st.number_input(
            "Maksimum gambar per kelas (0 = semua)",
            0, 5000, 300, 50
        )

        start = st.form_submit_button(
            "🚀 Mulai Training",
            type="primary"
        )

    if start:

        try:

            with st.spinner(
                "Training CNN berjalan..."
            ):

                model, history, val_data, class_names = train_model(
                    df,
                    method,
                    epochs,
                    batch_size,
                    image_size,
                    int(limit)
                )

            st.session_state["last_model"] = model

            st.session_state["val_data"] = val_data

            st.session_state["class_names"] = class_names

            st.success(
                f"Training selesai. Model tersimpan di {MODEL_PATH}"
            )

            hist = pd.DataFrame(
                history.history
            )

            if {
                "accuracy",
                "val_accuracy"
            }.issubset(hist.columns):

                st.subheader("Accuracy")

                st.line_chart(
                    hist[["accuracy", "val_accuracy"]]
                )

            if {
                "loss",
                "val_loss"
            }.issubset(hist.columns):

                st.subheader("Loss")

                st.line_chart(
                    hist[["loss", "val_loss"]]
                )

        except Exception as exc:

            st.exception(exc)


# ==========================================
# EVALUATION
# ==========================================

def page_evaluation():

    st.title("📊 Evaluation")

    if (
        "last_model" not in st.session_state
        or "val_data" not in st.session_state
    ):

        st.info(
            "Jalankan Training terlebih dahulu."
        )

        return

    try:

        x_val, y_val = st.session_state["val_data"]

        names = st.session_state["class_names"]

        result = evaluate_model(
            st.session_state["last_model"],
            x_val,
            y_val,
            names
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Accuracy",
            f"{result['accuracy'] * 100:.2f}%"
        )

        b.metric(
            "Precision",
            f"{result['precision'] * 100:.2f}%"
        )

        c.metric(
            "Recall",
            f"{result['recall'] * 100:.2f}%"
        )

        d.metric(
            "F1-score",
            f"{result['f1'] * 100:.2f}%"
        )

        report = pd.DataFrame(
            result["report"]
        ).T

        st.subheader("Classification Report")

        st.dataframe(
            report,
            use_container_width=True
        )

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.imshow(
            result["confusion_matrix"]
        )

        ax.set_xticks(
            range(len(names)),
            names
        )

        ax.set_yticks(
            range(len(names)),
            names
        )

        ax.set_xlabel("Predicted")

        ax.set_ylabel("Actual")

        for i in range(len(names)):

            for j in range(len(names)):

                ax.text(
                    j,
                    i,
                    result["confusion_matrix"][i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        plt.close(fig)

        st.download_button(
            "📥 Download Report CSV",
            report.to_csv().encode("utf-8"),
            "classification_report.csv",
            "text/csv"
        )

    except Exception as exc:

        st.exception(exc)


# ==========================================
# SIDEBAR
# ==========================================

page = st.sidebar.radio(
    "Navigasi",
    PAGES
)

st.sidebar.divider()

st.sidebar.markdown(
    "### Dataset Status"
)

if dataset_error:

    st.sidebar.warning(
        "Dataset belum lengkap"
    )

    st.sidebar.caption(
        dataset_error
    )

else:

    st.sidebar.success(
        f"{len(df):,} gambar terbaca"
    )

    st.sidebar.caption(
        f"Kelas: {df['label'].nunique()} / 3"
    )

if st.sidebar.button("🔄 Refresh Dataset"):

    get_dataset.clear()

    st.rerun()

st.sidebar.caption(
    "DermaCNN | ISIC 2018 | MEL / NV / BKL"
)


# ==========================================
# PAGE ROUTING
# ==========================================

if page == "Home":

    page_home()

elif page == "Dataset":

    page_dataset()

elif page == "Metadata Dataset":

    page_metadata()

elif page in METHOD_DESCRIPTIONS:

    page_preprocessing(page)

elif page == "Training (CNN)":

    page_training()

elif page == "Evaluation":

    page_evaluation()
