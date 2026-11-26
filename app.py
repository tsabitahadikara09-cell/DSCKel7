
# ============================================================
# DERMA CNN - ISIC 2018
# Deep Learning for Biomedical Image Processing
# ============================================================

import os
import random
import warnings
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DermaCNN | ISIC 2018",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #111019;
    color: #F3F0FA;
}

[data-testid="stSidebar"] {
    background-color: #191622;
}

h1, h2, h3 {
    color: #F3F0FA;
}

.metric-card {
    background-color: #211E2D;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #373244;
}

.info-card {
    background-color: #211E2D;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #373244;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATH
# ============================================================

# Struktur:
#
# DSC FP/
# ├── ISIC2018_Task3_Training_Input/
# ├── ISIC2018_Task3_Training_GroundTruth/
# └── Streamlit/
#     └── app.py

APP_DIR = Path(__file__).resolve().parent

PROJECT_DIR = APP_DIR.parent

IMAGE_DIR = r"D:\DSC FP\ISIC2018_Task3_Training_Input"
CSV_PATH = r"D:\DSC FP\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"

# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASS_CODES = ["MEL", "NV", "BKL"]

CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BKL": "Benign Keratosis"
}

CLASS_COLORS = {
    "MEL": "#E57373",
    "NV": "#64B5F6",
    "BKL": "#81C784"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def discover_images(folder):

    folder = Path(folder)

    if not folder.exists():
        return []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    image_files = []

    for file in folder.iterdir():

        if file.is_file() and file.suffix.lower() in extensions:
            image_files.append(file)

    return image_files


def find_ground_truth_csv():

    search_roots = [
        PROJECT_DIR,
        GT_DIR,
        APP_DIR
    ]

    csv_candidates = []

    for root in search_roots:

        if not root.exists():
            continue

        try:
            csv_candidates.extend(root.rglob("*.csv"))
        except Exception:
            pass

    # Remove duplicate paths
    csv_candidates = list(set(csv_candidates))

    required_columns = {"MEL", "NV", "BKL"}

    for csv_file in csv_candidates:

        try:

            df = pd.read_csv(
                csv_file,
                nrows=5
            )

            columns = set(df.columns)

            if not required_columns.issubset(columns):
                continue

            image_columns = [
                "image",
                "image_id",
                "image_name",
                "filename"
            ]

            if any(col in columns for col in image_columns):

                return csv_file

        except Exception:
            continue

    return None


def get_image_column(df):

    candidates = [
        "image",
        "image_id",
        "image_name",
        "filename"
    ]

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    # --------------------------------------------------------
    # CHECK IMAGE DIRECTORY
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        return pd.DataFrame(), None, (
            f"Folder gambar tidak ditemukan:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # FIND CSV
    # --------------------------------------------------------

    csv_path = find_ground_truth_csv()

    if csv_path is None:

        return pd.DataFrame(), None, (
            "CSV Ground Truth dengan kolom MEL, NV, dan BKL "
            "tidak ditemukan."
        )

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    try:

        label_df = pd.read_csv(csv_path)

    except Exception as e:

        return pd.DataFrame(), None, (
            f"Gagal membaca CSV: {e}"
        )

    image_column = get_image_column(label_df)

    if image_column is None:

        return pd.DataFrame(), csv_path, (
            "Kolom identitas gambar tidak ditemukan di CSV."
        )

    # --------------------------------------------------------
    # CHECK LABEL COLUMNS
    # --------------------------------------------------------

    missing_classes = [
        c for c in CLASS_CODES
        if c not in label_df.columns
    ]

    if missing_classes:

        return pd.DataFrame(), csv_path, (
            f"Kolom label tidak lengkap: {missing_classes}"
        )

    # --------------------------------------------------------
    # DISCOVER IMAGES
    # --------------------------------------------------------

    image_files = discover_images(IMAGE_DIR)

    image_map = {
        file.stem: str(file)
        for file in image_files
    }

    # --------------------------------------------------------
    # MATCH IMAGE AND LABEL
    # --------------------------------------------------------

    records = []

    for _, row in label_df.iterrows():

        image_id = str(row[image_column])

        if image_id not in image_map:
            continue

        active_classes = []

        for class_code in CLASS_CODES:

            try:

                value = float(row[class_code])

                if value == 1:
                    active_classes.append(class_code)

            except Exception:
                pass

        # Only samples belonging to one of the selected classes
        # are retained.
        #
        # Samples with multiple active labels among the selected
        # classes are excluded.

        if len(active_classes) != 1:
            continue

        label = active_classes[0]

        records.append({
            "image_id": image_id,
            "path": image_map[image_id],
            "label": label,
            "class_name": CLASS_NAMES[label]
        })

    dataset = pd.DataFrame(records)

    if dataset.empty:

        return dataset, csv_path, (
            "Tidak ditemukan gambar yang cocok dengan "
            "kelas MEL, NV, dan BKL."
        )

    return dataset, csv_path, None


# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def to_array(image):

    return np.array(
        image.convert("RGB")
    )


def filtering_image(image, method="Median"):

    img = to_array(image)

    if not CV2_AVAILABLE:
        return image.filter(ImageFilter.MedianFilter(size=3))

    if method == "Median":

        result = cv2.medianBlur(img, 3)

    elif method == "Gaussian":

        result = cv2.GaussianBlur(
            img,
            (5, 5),
            0
        )

    elif method == "Bilateral":

        result = cv2.bilateralFilter(
            img,
            9,
            75,
            75
        )

    elif method == "Average":

        result = cv2.blur(
            img,
            (3, 3)
        )

    else:

        result = img

    return Image.fromarray(result)


def enhancement_image(image, method="CLAHE"):

    img = to_array(image)

    if method == "Brightness":

        return ImageEnhance.Brightness(image).enhance(1.3)

    if method == "Contrast":

        return ImageEnhance.Contrast(image).enhance(1.5)

    if method == "Sharpness":

        return ImageEnhance.Sharpness(image).enhance(2.0)

    if method == "Color":

        return ImageEnhance.Color(image).enhance(1.5)

    if method == "CLAHE" and CV2_AVAILABLE:

        lab = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(l)

        merged = cv2.merge([
            enhanced_l,
            a,
            b
        ])

        result = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2RGB
        )

        return Image.fromarray(result)

    return image


def grayscale_image(image):

    return ImageOps.grayscale(image).convert("RGB")


def hair_removal_image(image):

    if not CV2_AVAILABLE:
        return image

    img = to_array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Black-hat morphological operation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Hair threshold
    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Inpainting
    result = cv2.inpaint(
        img,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return Image.fromarray(result)


def segmentation_image(image):

    img = to_array(image)

    if not CV2_AVAILABLE:

        gray = grayscale_image(image)

        return gray, gray

    # Resize for processing
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Gaussian blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu threshold
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Determine lesion polarity
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Keep largest connected component
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels > 1:

        largest_component = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = np.zeros_like(mask)

        mask[labels == largest_component] = 255

    # Create masked lesion image
    segmented = cv2.bitwise_and(
        img,
        img,
        mask=mask
    )

    return (
        Image.fromarray(mask),
        Image.fromarray(segmented)
    )


def process_image(image, method, parameter=None):

    if method == "Original":
        return image

    if method == "Median Filtering":
        return filtering_image(image, "Median")

    if method == "Gaussian Filtering":
        return filtering_image(image, "Gaussian")

    if method == "Bilateral Filtering":
        return filtering_image(image, "Bilateral")

    if method == "Average Filtering":
        return filtering_image(image, "Average")

    if method == "CLAHE":
        return enhancement_image(image, "CLAHE")

    if method == "Brightness":
        return enhancement_image(image, "Brightness")

    if method == "Contrast":
        return enhancement_image(image, "Contrast")

    if method == "Sharpness":
        return enhancement_image(image, "Sharpness")

    if method == "Hair Removal":
        return hair_removal_image(image)

    if method == "Grayscale":
        return grayscale_image(image)

    return image


# ============================================================
# DATASET PREPARATION FOR CNN
# ============================================================

def prepare_image(path, image_size=96, grayscale=False):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (image_size, image_size)
    )

    if grayscale:

        image = image.convert("L")

        image = image.convert("RGB")

    arr = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    return arr


# ============================================================
# CNN MODEL
# ============================================================

def create_cnn(image_size=96, num_classes=3):

    model = models.Sequential([

        layers.Input(
            shape=(image_size, image_size, 3)
        ),

        layers.Conv2D(
            32,
            3,
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            3,
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Conv2D(
            128,
            3,
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(title, subtitle=""):

    st.title(title)

    if subtitle:

        st.markdown(
            f"<p style='color:#AAA'>{subtitle}</p>",
            unsafe_allow_html=True
        )

    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🧬 DermaCNN")

    st.caption(
        "Skin Lesion Classification | ISIC 2018"
    )

    st.divider()

    menu = st.radio(
        "Navigasi",
        [
            "Home",
            "Dataset",
            "Filtering",
            "Segmentation & Masking",
            "Enhancement",
            "Hair Removal",
            "Grayscale",
            "Training (CNN)",
            "Evaluation"
        ]
    )

    st.divider()

    st.subheader("Dataset")

    st.caption(
        "Dataset dibaca otomatis dari folder project."
    )

    st.code(
        str(IMAGE_DIR),
        language="text"
    )


# ============================================================
# LOAD DATASET
# ============================================================

dataset_df, detected_csv, dataset_error = load_dataset()


# ============================================================
# HOME
# ============================================================

if menu == "Home":

    page_header(
        "🧬 DermaCNN",
        "Deep Learning-Based Skin Lesion Image Processing "
        "and Classification"
    )

    st.markdown("""
    ### Project Overview

    DermaCNN merupakan aplikasi pengolahan citra lesi kulit
    menggunakan dataset ISIC 2018 dengan pendekatan
    Convolutional Neural Network (CNN).

    Sistem dirancang untuk mengeksplorasi pengaruh berbagai
    preprocessing terhadap kualitas citra dan performa
    klasifikasi lesi kulit.
    """)

    st.subheader("Pipeline")

    cols = st.columns(4)

    pipeline = [
        ("01", "Dataset", "ISIC 2018"),
        ("02", "Preprocessing", "Image Processing"),
        ("03", "Training", "CNN"),
        ("04", "Evaluation", "Model Performance")
    ]

    for col, item in zip(cols, pipeline):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h2>{item[0]}</h2>
                    <b>{item[1]}</b>
                    <p>{item[2]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("Classification Classes")

    cols = st.columns(3)

    for col, code in zip(cols, CLASS_CODES):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="color:{CLASS_COLORS[code]}">
                        {code}
                    </h3>
                    <b>{CLASS_NAMES[code]}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DATASET PAGE
# ============================================================

elif menu == "Dataset":

    page_header(
        "📁 Dataset Explorer",
        "ISIC 2018 Training Dataset"
    )

    if dataset_error:

        st.error(dataset_error)

        st.info(
            "Pastikan folder dataset tersedia di folder "
            "induk project."
        )

        st.code(str(IMAGE_DIR))

    else:

        st.success("Dataset berhasil dibaca.")

        st.write(
            f"**Ground Truth:** `{detected_csv}`"
        )

        summary = class_summary if False else None

        counts = dataset_df["label"].value_counts()

        total_images = len(dataset_df)

        st.subheader("Dataset Statistics")

        cols = st.columns(4)

        cols[0].metric(
            "Total Images",
            f"{total_images:,}"
        )

        for i, code in enumerate(CLASS_CODES):

            cols[i + 1].metric(
                code,
                f"{counts.get(code, 0):,}"
            )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Class Distribution")

        distribution = (
            dataset_df["label"]
            .value_counts()
            .reindex(CLASS_CODES, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            [CLASS_NAMES[c] for c in CLASS_CODES],
            distribution.values
        )

        ax.set_ylabel("Number of Images")

        ax.set_xlabel("Class")

        ax.set_title("ISIC 2018 - Selected Classes")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # SAMPLE IMAGES
        # ----------------------------------------------------

        st.subheader("Sample Images by Class")

        for code in CLASS_CODES:

            st.markdown(
                f"### {code} — {CLASS_NAMES[code]}"
            )

            class_df = dataset_df[
                dataset_df["label"] == code
            ]

            samples = class_df.sample(
                n=min(5, len(class_df)),
                random_state=42
            )

            cols = st.columns(5)

            for col, (_, row) in zip(
                cols,
                samples.iterrows()
            ):

                with col:

                    try:

                        image = Image.open(
                            row["path"]
                        ).convert("RGB")

                        st.image(
                            image,
                            caption=row["image_id"],
                            use_container_width=True
                        )

                    except Exception:
                        st.warning("Image gagal dibaca.")


# ============================================================
# FILTERING
# ============================================================

elif menu == "Filtering":

    page_header(
        "🔬 Filtering",
        "Noise reduction and smoothing"
    )

    if dataset_error:

        st.error(dataset_error)

    else:

        selected_class = st.selectbox(
            "Select Class",
            CLASS_CODES,
            format_func=lambda x: (
                f"{x} — {CLASS_NAMES[x]}"
            )
        )

        method = st.selectbox(
            "Filtering Method",
            [
                "Median Filtering",
                "Gaussian Filtering",
                "Bilateral Filtering",
                "Average Filtering"
            ]
        )

        class_df = dataset_df[
            dataset_df["label"] == selected_class
        ]

        st.write(
            f"Total images: {len(class_df)}"
        )

        sample_count = st.slider(
            "Number of samples",
            1,
            10,
            5
        )

        samples = class_df.sample(
            n=min(sample_count, len(class_df)),
            random_state=42
        )

        for _, row in samples.iterrows():

            original = Image.open(
                row["path"]
            ).convert("RGB")

            processed = process_image(
                original,
                method
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    original,
                    caption="Original",
                    use_container_width=True
                )

            with col2:

                st.image(
                    processed,
                    caption=method,
                    use_container_width=True
                )

            st.caption(
                f"{row['image_id']} | {CLASS_NAMES[selected_class]}"
            )


# ============================================================
# SEGMENTATION & MASKING
# ============================================================

elif menu == "Segmentation & Masking":

    page_header(
        "🎯 Segmentation & Masking",
        "Lesion region extraction"
    )

    if dataset_error:

        st.error(dataset_error)

    else:

        selected_class = st.selectbox(
            "Select Class",
            CLASS_CODES,
            format_func=lambda x: (
                f"{x} — {CLASS_NAMES[x]}"
            )
        )

        class_df = dataset_df[
            dataset_df["label"] == selected_class
        ]

        sample_count = st.slider(
            "Number of samples",
            1,
            10,
            5
        )

        samples = class_df.sample(
            n=min(sample_count, len(class_df)),
            random_state=42
        )

        for _, row in samples.iterrows():

            original = Image.open(
                row["path"]
            ).convert("RGB")

            mask, segmented = segmentation_image(
                original
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.image(
                    original,
                    caption="Original",
                    use_container_width=True
                )

            with col2:

                st.image(
                    mask,
                    caption="Lesion Mask",
                    use_container_width=True
                )

            with col3:

                st.image(
                    segmented,
                    caption="Segmented Image",
                    use_container_width=True
                )

            st.caption(row["image_id"])


# ============================================================
# ENHANCEMENT
# ============================================================

elif menu == "Enhancement":

    page_header(
        "✨ Enhancement",
        "Image contrast and visual enhancement"
    )

    if dataset_error:

        st.error(dataset_error)

    else:

        selected_class = st.selectbox(
            "Select Class",
            CLASS_CODES,
            format_func=lambda x: (
                f"{x} — {CLASS_NAMES[x]}"
            )
        )

        method = st.selectbox(
            "Enhancement Method",
            [
                "CLAHE",
                "Brightness",
                "Contrast",
                "Sharpness",
                "Color"
            ]
        )

        class_df = dataset_df[
            dataset_df["label"] == selected_class
        ]

        sample_count = st.slider(
            "Number of samples",
            1,
            10,
            5
        )

        samples = class_df.sample(
            n=min(sample_count, len(class_df)),
            random_state=42
        )

        for _, row in samples.iterrows():

            original = Image.open(
                row["path"]
            ).convert("RGB")

            processed = enhancement_image(
                original,
                method
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    original,
                    caption="Original",
                    use_container_width=True
                )

            with col2:

                st.image(
                    processed,
                    caption=method,
                    use_container_width=True
                )


# ============================================================
# HAIR REMOVAL
# ============================================================

elif menu == "Hair Removal":

    page_header(
        "🧹 Hair Removal",
        "Removal of hair artifacts from dermoscopic images"
    )

    if dataset_error:

        st.error(dataset_error)

    elif not CV2_AVAILABLE:

        st.error(
            "OpenCV belum terinstal. "
            "Install opencv-python."
        )

    else:

        selected_class = st.selectbox(
            "Select Class",
            CLASS_CODES,
            format_func=lambda x: (
                f"{x} — {CLASS_NAMES[x]}"
            )
        )

        class_df = dataset_df[
            dataset_df["label"] == selected_class
        ]

        sample_count = st.slider(
            "Number of samples",
            1,
            10,
            5
        )

        samples = class_df.sample(
            n=min(sample_count, len(class_df)),
            random_state=42
        )

        for _, row in samples.iterrows():

            original = Image.open(
                row["path"]
            ).convert("RGB")

            processed = hair_removal_image(
                original
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    original,
                    caption="Original",
                    use_container_width=True
                )

            with col2:

                st.image(
                    processed,
                    caption="Hair Removed",
                    use_container_width=True
                )


# ============================================================
# GRAYSCALE
# ============================================================

elif menu == "Grayscale":

    page_header(
        "⚫ Grayscale",
        "RGB to grayscale conversion"
    )

    if dataset_error:

        st.error(dataset_error)

    else:

        selected_class = st.selectbox(
            "Select Class",
            CLASS_CODES,
            format_func=lambda x: (
                f"{x} — {CLASS_NAMES[x]}"
            )
        )

        class_df = dataset_df[
            dataset_df["label"] == selected_class
        ]

        sample_count = st.slider(
            "Number of samples",
            1,
            10,
            5
        )

        samples = class_df.sample(
            n=min(sample_count, len(class_df)),
            random_state=42
        )

        for _, row in samples.iterrows():

            original = Image.open(
                row["path"]
            ).convert("RGB")

            processed = grayscale_image(
                original
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    original,
                    caption="Original RGB",
                    use_container_width=True
                )

            with col2:

                st.image(
                    processed,
                    caption="Grayscale",
                    use_container_width=True
                )


# ============================================================
# TRAINING CNN
# ============================================================

elif menu == "Training (CNN)":

    page_header(
        "🧠 CNN Training",
        "Train a convolutional neural network "
        "for skin lesion classification"
    )

    if dataset_error:

        st.error(dataset_error)

    elif not TF_AVAILABLE:

        st.error(
            "TensorFlow belum tersedia."
        )

        st.code(
            "pip install tensorflow"
        )

    else:

        st.info(
            "Training dilakukan menggunakan 3 kelas "
            "MEL, NV, dan BKL."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            image_size = st.selectbox(
                "Image Size",
                [64, 96, 128],
                index=1
            )

        with col2:

            epochs = st.slider(
                "Epochs",
                1,
                30,
                5
            )

        with col3:

            batch_size = st.selectbox(
                "Batch Size",
                [8, 16, 32],
                index=1
            )

        max_per_class = st.slider(
            "Maximum images per class",
            50,
            1000,
            200,
            step=50
        )

        grayscale_train = st.checkbox(
            "Use Grayscale Images",
            value=False
        )

        st.caption(
            "Jumlah data dibatasi untuk mengurangi "
            "beban memori dan waktu training."
        )

        if st.button(
            "🚀 Start CNN Training",
            type="primary"
        ):

            with st.spinner(
                "Preparing dataset..."
            ):

                train_df = dataset_df.copy()

                balanced_samples = []

                for code in CLASS_CODES:

                    class_df = train_df[
                        train_df["label"] == code
                    ]

                    class_df = class_df.sample(
                        n=min(
                            max_per_class,
                            len(class_df)
                        ),
                        random_state=42
                    )

                    balanced_samples.append(class_df)

                train_df = pd.concat(
                    balanced_samples,
                    ignore_index=True
                )

                random.seed(42)

                train_df = train_df.sample(
                    frac=1,
                    random_state=42
                ).reset_index(drop=True)

                X = []
                y = []

                label_mapping = {
                    code: index
                    for index, code
                    in enumerate(CLASS_CODES)
                }

                for _, row in train_df.iterrows():

                    try:

                        arr = prepare_image(
                            row["path"],
                            image_size,
                            grayscale_train
                        )

                        X.append(arr)

                        y.append(
                            label_mapping[row["label"]]
                        )

                    except Exception:
                        continue

                X = np.array(
                    X,
                    dtype=np.float32
                )

                y = np.array(
                    y,
                    dtype=np.int32
                )

            if len(X) < 10:

                st.error(
                    "Data terlalu sedikit untuk training."
                )

            else:

                from sklearn.model_selection import train_test_split
                from sklearn.utils.class_weight import compute_class_weight

                y_cat = tf.keras.utils.to_categorical(
                    y,
                    num_classes=3
                )

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y_cat,
                        test_size=0.2,
                        random_state=42,
                        stratify=y
                    )
                )

                y_train_labels = np.argmax(
                    y_train,
                    axis=1
                )

                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(y_train_labels),
                    y=y_train_labels
                )

                class_weight_dict = {
                    int(i): float(weight)
                    for i, weight
                    in enumerate(class_weights)
                }

                model = create_cnn(
                    image_size=image_size
                )

                progress = st.progress(0)

                status = st.empty()

                history_data = {}

                # ------------------------------------------------
                # TRAINING
                # ------------------------------------------------

                status.info(
                    "Training CNN sedang berlangsung..."
 
                )

                callback = EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True
                )

                history = model.fit(
                    X_train,
                    y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    callbacks=[callback],
                    verbose=0
                )

                progress.progress(100)

                status.success(
                    "Training selesai!"
                )

                # ------------------------------------------------
                # SAVE SESSION
                # ------------------------------------------------

                st.session_state["cnn_model"] = model

                st.session_state["cnn_history"] = (
                    history.history
                )

                st.session_state["X_test"] = X_test

                st.session_state["y_test"] = y_test

                st.session_state["image_size"] = (
                    image_size
                )

                st.session_state["training_classes"] = (
                    CLASS_CODES
                )

                st.session_state["training_done"] = True

                # ------------------------------------------------
                # TRAINING CURVES
                # ------------------------------------------------

                hist = history.history

                st.subheader("Training Results")

                col1, col2 = st.columns(2)

                with col1:

                    fig, ax = plt.subplots()

                    ax.plot(
                        hist["accuracy"],
                        label="Training Accuracy"
                    )

                    ax.plot(
                        hist["val_accuracy"],
                        label="Validation Accuracy"
                    )

                    ax.set_title("Accuracy")

                    ax.set_xlabel("Epoch")

                    ax.set_ylabel("Accuracy")

                    ax.legend()

                    st.pyplot(fig)

                    plt.close(fig)

                with col2:

                    fig, ax = plt.subplots()

                    ax.plot(
                        hist["loss"],
                        label="Training Loss"
                    )

                    ax.plot(
                        hist["val_loss"],
                        label="Validation Loss"
                    )

                    ax.set_title("Loss")

                    ax.set_xlabel("Epoch")

                    ax.set_ylabel("Loss")

                    ax.legend()

                    st.pyplot(fig)

                    plt.close(fig)

                st.success(
                    "Model berhasil disimpan di session Streamlit."
 
                )


# ============================================================
# EVALUATION
# ============================================================

elif menu == "Evaluation":

    page_header(
        "📊 Model Evaluation",
        "Performance evaluation of CNN"
    )

    if not TF_AVAILABLE:

        st.error(
            "TensorFlow belum tersedia."
        )

    elif "training_done" not in st.session_state:

        st.warning(
            "Silakan jalankan Training CNN terlebih dahulu."
        )

    else:

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report
        )

        model = st.session_state["cnn_model"]

        X_test = st.session_state["X_test"]

        y_test = st.session_state["y_test"]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "Evaluating model..."
        ):

            probabilities = model.predict(
                X_test,
                verbose=0
            )

            y_pred = np.argmax(
                probabilities,
                axis=1
            )

            y_true = np.argmax(
                y_test,
                axis=1
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        st.subheader("Performance Metrics")

        cols = st.columns(4)

        cols[0].metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        cols[1].metric(
            "Precision",
            f"{precision:.4f}"
        )

        cols[2].metric(
            "Recall",
            f"{recall:.4f}"
        )

        cols[3].metric(
            "F1-Score",
            f"{f1:.4f}"
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2]
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        im = ax.imshow(
            cm,
            interpolation="nearest"
        )

        ax.set_xticks(
            range(3)
        )

        ax.set_yticks(
            range(3)
        )

        ax.set_xticklabels(
            CLASS_CODES
        )

        ax.set_yticklabels(
            CLASS_CODES
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "True Label"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        for i in range(3):

            for j in range(3):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(im)

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                CLASS_NAMES[c]
                for c in CLASS_CODES
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        csv_report = report_df.to_csv().encode(
            "utf-8"
        )

        st.download_button(
            "Download Evaluation Report",
            data=csv_report,
            file_name="cnn_evaluation_report.csv",
            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "DermaCNN | Biomedical Deep Learning Project"
)

st.sidebar.caption(
    "ISIC 2018 | MEL • NV • BKL"
)
