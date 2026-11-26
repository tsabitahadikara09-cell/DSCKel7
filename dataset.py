import os
import glob
import pandas as pd
from PIL import Image

CLASS_CODES = ["MEL", "NV", "BKL"]

def discover_images(folder):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder, pattern), recursive=True))
    return files

def load_dataset_index(image_folder, csv_path):
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {image_folder}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV Ground Truth tidak ditemukan: {csv_path}")
    label_df = pd.read_csv(csv_path)
    if "image" not in label_df.columns:
        raise ValueError("CSV harus memiliki kolom 'image'.")
    image_map = {os.path.splitext(os.path.basename(p))[0]: p for p in discover_images(image_folder)}
    rows = []
    for _, row in label_df.iterrows():
        image_id = str(row["image"])
        path = image_map.get(image_id)
        if not path:
            continue
        active = [c for c in CLASS_CODES if c in label_df.columns and pd.to_numeric(row[c], errors="coerce") == 1]
        if len(active) != 1:
            continue
        rows.append({"image_id": image_id, "path": path, "label": active[0]})
    if not rows:
        raise ValueError("Tidak ada gambar yang cocok dengan label MEL, NV, atau BKL. Periksa nama file dan CSV.")
    return pd.DataFrame(rows)

def class_summary(df):
    return df["label"].value_counts().to_dict()

def sample_image(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None
