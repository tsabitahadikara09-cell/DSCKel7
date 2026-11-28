
from pathlib import Path
import pandas as pd
from PIL import Image

# =====================================
# PROJECT CONFIGURATION
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_CODES = ["MEL", "NV", "BKL"]

CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BKL": "Benign Keratosis-like Lesions"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =====================================
# FIND GROUND TRUTH CSV
# =====================================

def find_ground_truth_csv():

    candidates = [
        PROJECT_ROOT / "ISIC2018_Task3_Training_GroundTruth.csv",
        PROJECT_ROOT / "ISIC2018_Task3_Validation_GroundTruth.csv",
        PROJECT_ROOT / "dataset" / "ISIC2018_Task3_Training_GroundTruth.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


# =====================================
# FIND IMAGE DIRECTORY
# =====================================

def find_image_dir():

    candidates = [
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "images",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "ISIC2018_Task3_Training_Input",
    ]

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    return PROJECT_ROOT


# =====================================
# BUILD IMAGE MAP
# =====================================

def build_image_map(image_dir):

    image_map = {}

    if not image_dir or not image_dir.exists():
        return image_map

    for path in image_dir.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and ".git" not in path.parts
        ):
            image_map[path.stem] = path

    return image_map


# =====================================
# BUILD METADATA
# =====================================

def build_metadata(csv_path=None, image_dir=None):

    csv_path = (
        Path(csv_path)
        if csv_path
        else find_ground_truth_csv()
    )

    image_dir = (
        Path(image_dir)
        if image_dir
        else find_image_dir()
    )

    if csv_path is None or not csv_path.exists():
        return pd.DataFrame(), "CSV Ground Truth tidak ditemukan."

    try:
        source = pd.read_csv(csv_path)
    except Exception as e:
        return pd.DataFrame(), f"Gagal membaca CSV: {e}"

    # Normalize column names
    source.columns = source.columns.astype(str).str.strip()

    id_col = next(
        (
            c for c in [
                "image",
                "image_id",
                "image_name",
                "filename",
                "file_name"
            ]
            if c in source.columns
        ),
        None
    )

    if id_col is None:
        return pd.DataFrame(), (
            "Kolom ID gambar tidak ditemukan dalam CSV."
        )

    missing = [
        c for c in CLASS_CODES
        if c not in source.columns
    ]

    if missing:
        return pd.DataFrame(), (
            f"Kolom label tidak lengkap: {missing}"
        )

    image_map = build_image_map(image_dir)

    rows = []

    for _, row in source.iterrows():

        raw_id = str(row[id_col]).strip()
        image_id = Path(raw_id).stem

        if not image_id or image_id.lower() == "nan":
            continue

        # Determine class from one-hot labels
        active = []

        for code in CLASS_CODES:

            try:
                if float(row[code]) == 1:
                    active.append(code)
            except (ValueError, TypeError):
                continue

        # Skip invalid or ambiguous labels
        if len(active) != 1:
            continue

        code = active[0]

        image_path = image_map.get(image_id)

        width = None
        height = None
        size_kb = None
        image_format = None
        valid = False

        if image_path is not None:

            try:
                with Image.open(image_path) as img:

                    width, height = img.size
                    image_format = img.format
                    img.verify()

                size_kb = round(
                    image_path.stat().st_size / 1024,
                    2
                )

                valid = True

            except Exception:
                valid = False

        rows.append({

            "image_id": image_id,

            "filename": (
                image_path.name
                if image_path
                else f"{image_id}.jpg"
            ),

            "label": code,

            "class_name": CLASS_NAMES[code],

            "image_path": (
                str(image_path)
                if image_path
                else ""
            ),

            "format": image_format,

            "width": width,

            "height": height,

            "size_kb": size_kb,

            "image_available": valid,

        })

    metadata = pd.DataFrame(rows)

    if metadata.empty:
        return metadata, "Metadata kosong atau tidak memiliki label valid."

    return metadata, None
