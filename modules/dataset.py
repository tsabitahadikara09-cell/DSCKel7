
from pathlib import Path
import pandas as pd
from PIL import Image

# =========================================================
# CLASS CONFIGURATION
# =========================================================

CLASS_CODES = ["MEL", "NV", "BKL"]

CLASS_NAMES = {
    "MEL": "Melanoma",
    ""NV": "Melanocytic Nevus",
    ""BKL": "Benign Keratosis",
}

# =========================================================
# PROJECT PATH
# =========================================================

# Folder modules -> folder Streamlit -> folder DSC FP
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_DIR.parent


def get_dataset_paths():

    image_dir = DATA_ROOT / "ISIC2018_Task3_Training_Input"

    gt_dir = DATA_ROOT / "ISIC2018_Task3_Training_GroundTruth"

    return image_dir, gt_dir


# =========================================================
# FIND IMAGE FILES
# =========================================================

def discover_images(image_folder):

    image_folder = Path(image_folder)

    extensions = {".jpg", ".jpeg", ".png"}

    files = []

    if not image_folder.exists():
        return files

    for file in image_folder.iterdir():

        if file.is_file() and file.suffix.lower() in extensions:
            files.append(file)

    return files


# =========================================================
# FIND VALID GROUND TRUTH CSV
# =========================================================

def find_ground_truth_csv(search_roots):

    required_classes = {"MEL", "NV", "BKL"}

    candidates = []

    for root in search_roots:

        root = Path(root)

        if not root.exists():
            continue

        # Cari CSV di dalam folder secara recursive
        candidates.extend(root.rglob("*.csv"))

    # Hindari file duplikat
    candidates = list(set(candidates))

    for csv_file in candidates:

        try:

            df = pd.read_csv(csv_file, nrows=5)

            columns = set(df.columns)

            # CSV harus memiliki kolom kelas MEL, NV, dan BKL
            if required_classes.issubset(columns):

                # Harus ada kolom identitas gambar
                possible_image_columns = {
                    "image",
                    "image_id",
                    "image_name",
                    "filename"
                }

                if columns.intersection(possible_image_columns):

                    return csv_file

        except Exception:
            continue

    return None


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset_index(image_folder=None, csv_path=None):

    default_image_dir, default_gt_dir = get_dataset_paths()

    if image_folder is None:
        image_folder = default_image_dir

    image_folder = Path(image_folder)

    if not image_folder.exists():

        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan:\n{image_folder}"
        )

    # Jika CSV tidak ditentukan, cari otomatis
    if csv_path is None:

        csv_path = find_ground_truth_csv([
            default_gt_dir,
            DATA_ROOT,
            PROJECT_DIR
        ])

    if csv_path is None:

        raise FileNotFoundError(
            "CSV Ground Truth yang memiliki label "
            "MEL, NV, dan BKL tidak ditemukan."
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"CSV Ground Truth tidak ditemukan:\n{csv_path}"
        )

    # Baca CSV
    label_df = pd.read_csv(csv_path)

    # Deteksi kolom ID gambar
    image_column = None

    for column in ["image", "image_id", "image_name", "filename"]:

        if column in label_df.columns:

            image_column = column
            break

    if image_column is None:

        raise ValueError(
            "Kolom nama gambar tidak ditemukan dalam CSV."
        )

    # Pastikan kelas tersedia
    missing_classes = [
        class_code
        for class_code in CLASS_CODES
        if class_code not in label_df.columns
    ]

    if missing_classes:

        raise ValueError(
            f"Label kelas tidak ditemukan: {missing_classes}"
        )

    # Mapping gambar berdasarkan ID
    image_files = discover_images(image_folder)

    image_map = {}

    for path in image_files:

        image_id = path.stem

        image_map[image_id] = str(path)

    rows = []

    # =====================================================
    # MATCH IMAGE WITH LABEL
    # =====================================================

    for _, row in label_df.iterrows():

        image_id = str(row[image_column])

        if image_id not in image_map:
            continue

        # Ambil label MEL, NV, BKL
        active_classes = []

        for class_code in CLASS_CODES:

            try:

                if float(row[class_code]) == 1:
                    active_classes.append(class_code)

            except (ValueError, TypeError):
                continue

        # Hanya ambil gambar yang mempunyai tepat satu
        # label aktif di antara 3 kelas yang dipilih
        if len(active_classes) != 1:
            continue

        label = active_classes[0]

        rows.append({
            "image_id": image_id,
            "path": image_map[image_id],
            "label": label,
            "class_name": CLASS_NAMES[label]
        })

    dataset = pd.DataFrame(rows)

    if dataset.empty:

        raise ValueError(
            "Tidak ada gambar yang berhasil dicocokkan "
            "dengan label MEL, NV, atau BKL.\n"
            "Periksa format CSV dan nama file gambar."
        )

    return dataset


# =========================================================
# CLASS SUMMARY
# =========================================================

def class_summary(df):

    counts = df["label"].value_counts()

    result = {}

    for class_code in CLASS_CODES:

        result[class_code] = int(
            counts.get(class_code, 0)
        )

    return result


# =========================================================
# SAMPLE IMAGE
# =========================================================

def sample_image(path):

    try:

        image = Image.open(path).convert("RGB")

        return image

    except Exception:

        return None
