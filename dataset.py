
import os
import glob
import pandas as pd
from PIL import Image

# 3 kelas yang digunakan
CLASS_CODES = ["MEL", "NV", "BKL"]

CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BKL": "Benign Keratosis"
}


def discover_images(folder):
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(folder, ext))
        )

    return files


def load_dataset_index(image_folder, csv_path):

    if not os.path.exists(image_folder):
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {image_folder}"
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File label tidak ditemukan: {csv_path}"
        )

    label_df = pd.read_csv(csv_path)

    # Membaca seluruh gambar
    image_files = discover_images(image_folder)

    # Membuat mapping nama file -> lokasi
    image_map = {}

    for path in image_files:
        image_id = os.path.splitext(
            os.path.basename(path)
        )[0]

        image_map[image_id] = path

    rows = []

    # Memproses label dataset
    for _, row in label_df.iterrows():

        image_id = str(row["image"])

        # Skip jika gambar tidak ditemukan
        if image_id not in image_map:
            continue

        # Ambil label aktif dari 3 kelas pilihan
        active_classes = []

        for class_code in CLASS_CODES:

            if class_code in label_df.columns:

                if row[class_code] == 1:
                    active_classes.append(class_code)

        # Hanya gambar yang memiliki tepat 1 label
        if len(active_classes) != 1:
            continue

        label = active_classes[0]

        rows.append({
            "image_id": image_id,
            "path": image_map[image_id],
            "label": label
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Tidak ditemukan gambar yang cocok "
            "dengan label MEL, NV, atau BKL."
        )

    return df


def class_summary(df):
    return df["label"].value_counts().to_dict()


def sample_image(path):

    try:
        img = Image.open(path)
        return img.convert("RGB")

    except Exception:
        return None
