from pathlib import Path
import pandas as pd
from PIL import Image
from .dataset import PROJECT_ROOT, CLASS_CODES, CLASS_NAMES, find_ground_truth_csv, find_image_dir

def build_metadata(csv_path=None, image_dir=None):
    csv_path = Path(csv_path) if csv_path else find_ground_truth_csv()
    image_dir = Path(image_dir) if image_dir else find_image_dir()
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame(), "CSV Ground Truth tidak ditemukan."

    source = pd.read_csv(csv_path)
    id_col = next((c for c in ("image", "image_id", "image_name", "filename", "file_name")
                   if c in source.columns), None)
    if id_col is None:
        return pd.DataFrame(), "Kolom ID gambar tidak ditemukan di CSV."
    missing = [c for c in CLASS_CODES if c not in source.columns]
    if missing:
        return pd.DataFrame(), f"Kolom label tidak lengkap: {missing}"

    image_map = {}
    if image_dir.exists():
        for p in image_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_map[p.stem] = p

    rows = []
    for _, r in source.iterrows():
        image_id = str(r[id_col]).strip()
        active = []
        for c in CLASS_CODES:
            try:
                if float(r[c]) == 1:
                    active.append(c)
            except (TypeError, ValueError):
                pass
        if len(active) != 1:
            continue
        code = active[0]
        path = image_map.get(image_id)
        width = height = size_kb = None
        fmt = None
        valid = False
        if path:
            try:
                with Image.open(path) as im:
                    width, height = im.size
                    fmt = im.format
                    im.verify()
                size_kb = round(path.stat().st_size / 1024, 2)
                valid = True
            except Exception:
                valid = False
        rows.append({
            "image_id": image_id,
            "filename": path.name if path else f"{image_id}.jpg",
            "label": code,
            "class_name": CLASS_NAMES[code],
            "image_path": str(path) if path else "",
            "format": fmt,
            "width": width,
            "height": height,
            "size_kb": size_kb,
            "image_available": valid,
        })
    return pd.DataFrame(rows), None
