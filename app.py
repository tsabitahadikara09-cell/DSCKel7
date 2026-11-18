from __future__ import annotations

import io
import json
import math
import random
import zipfile
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Optional deep-learning dependencies. The Dataset Explorer works without PyTorch.
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
        precision_score, recall_score, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(
    page_title="DermaSense | ISIC 2019",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "ISIC_2019_Training_Input"
MODEL_DIR = BASE_DIR / "models"
MANUAL_IMAGE_DIR = DATA_DIR / "manual_uploads"
MANUAL_DB = DATA_DIR / "manual_records.db"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
MANUAL_IMAGE_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
}
CLASS_COLORS = {"MEL": "#F47D62", "NV": "#079B8B", "BCC": "#65D3CA"}
CLASS_ORDER = ["MEL", "NV", "BCC"]
EXPECTED_COUNTS = {"MEL": 4522, "NV": 12875, "BCC": 3323}
SPLIT_RATIOS = (0.70, 0.15, 0.15)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root { --navy:#092E3A; --deep:#073B42; --teal:#079B8B; --mint:#DDF5F1; --ink:#142C40; --muted:#71838D; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#F4F8FA; color:var(--ink); }
.block-container { max-width:1660px; padding-top:3.5rem !important; padding-bottom:2rem; }
/* Keep the page heading clear of Streamlit Cloud top toolbar */
main .block-container { padding-top:3.5rem !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#063B42 0%,#052F39 100%); min-width:350px !important; max-width:350px !important; }
[data-testid="stSidebarContent"] { overflow-y:auto !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0.55rem; }
.sidebar-input-card { background:rgba(12,105,108,.48); border:1px solid rgba(130,220,211,.18); border-radius:15px; padding:12px; margin:8px 0 12px; }
.sidebar-section-title { font-family:Manrope,sans-serif;font-weight:800;font-size:17px;color:#F4FFFF;margin:0 0 2px; }
.sidebar-hint { font-size:10px;color:#C3DDDC;margin-bottom:8px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { background:rgba(255,255,255,.06); border:1px dashed #80C7C3; padding:5px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { padding:5px; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div { background:#F5FBFB !important; color:#183B42 !important; border-radius:8px !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label, [data-testid="stSidebar"] label { font-size:11px !important; }
[data-testid="stSidebar"] * { color:#F2FAFA !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#C3D8D9 !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label { background:transparent; border-radius:11px; padding:8px 10px; margin:3px 0; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#0B5E61; }
.brand {font-family:Manrope,sans-serif;font-size:25px;font-weight:800;letter-spacing:-.8px;color:white;}
.brand-mark {font-size:29px;color:#48D8C7;margin-right:8px;}
.brand-sub {font-size:11px;color:#C4DADB;line-height:1.5;margin-top:-5px;}
.hero-title {font-family:Manrope,sans-serif;font-size:29px;font-weight:800;letter-spacing:-.7px;color:#102A43;margin:0 0 2px 0;}
.hero-sub {font-size:13px;color:#617482;margin-bottom:15px;}
.top-tag {background:#087F77;color:white;padding:8px 15px;border-radius:30px;font-size:12px;font-weight:700;}
.panel {background:white;border:1px solid #E0E9ED;border-radius:17px;padding:15px 17px;box-shadow:0 4px 14px rgba(18,55,70,.045);}
.panel-title {font-family:Manrope,sans-serif;font-size:16px;font-weight:800;color:#102A43;margin:0 0 9px 0;}
.panel-caption {font-size:11px;color:#71838D;margin-top:-5px;margin-bottom:10px;}
.metric-card {background:white;border:1px solid #E0E9ED;border-radius:16px;padding:14px 16px;box-shadow:0 4px 14px rgba(18,55,70,.04);min-height:100px;}
.metric-label {font-size:11px;font-weight:700;letter-spacing:.5px;color:#647986;}
.metric-value {font-family:Manrope,sans-serif;font-size:25px;font-weight:800;color:#102A43;margin-top:4px;}
.metric-help {font-size:11px;color:#0A8D83;margin-top:2px;}
.section-heading {font-family:Manrope,sans-serif;font-size:20px;font-weight:800;color:#102A43;margin:5px 0 2px;}
.soft-note {background:#E8F7F4;border:1px solid #C9ECE6;border-radius:12px;padding:10px 13px;color:#255E61;font-size:12px;}
.status-pill {display:inline-block;background:#E8F7F4;color:#087F77;border-radius:30px;padding:5px 10px;font-size:11px;font-weight:700;}
.stButton>button {border-radius:10px;font-weight:700;min-height:38px;}
.stDownloadButton>button {border-radius:10px;font-weight:700;}
div[data-testid="stFileUploader"] {background:#FBFDFE;border:1px dashed #B8CDD5;border-radius:12px;padding:8px;}
div[data-baseweb="tab-list"] {gap:6px;}
button[data-baseweb="tab"] {border-radius:10px 10px 0 0;font-weight:700;}
div[data-testid="stDataFrame"] {border:1px solid #E2EAED;border-radius:10px;overflow:hidden;}
.small-muted {font-size:11px;color:#738691;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE + DATA HELPERS
# ============================================================
DEFAULTS = {
    "gt": None, "meta": None, "data_df": None, "image_dir": str(IMAGE_DIR),
    "trained_model": None, "training_history": None, "eval_result": None,
    "model_status": "Belum dilatih", "last_model_path": None,
    "split_df": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def normalize_image_id(value) -> str:
    value = str(value).strip()
    return Path(value).stem

def prepare_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    image_col = find_col(df, ["image", "image_id", "isic_id", "isic"])
    if image_col is None:
        raise ValueError("CSV ground truth harus memiliki kolom image atau image_id.")
    df = df.rename(columns={image_col: "image"})
    df["image"] = df["image"].map(normalize_image_id)

    # ISIC 2019 official ground truth is one-hot encoded.
    present = [c for c in CLASS_ORDER if c in df.columns]
    if set(CLASS_ORDER).issubset(present):
        # Keep only rows whose one-hot target belongs to one of the selected classes.
        # Rows from the other five ISIC 2019 classes have MEL/NV/BCC all equal to zero.
        selected_targets = df[CLASS_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)
        df = df.loc[selected_targets.sum(axis=1) > 0].copy()
        selected_targets = selected_targets.loc[df.index]
        df["diagnosis"] = selected_targets.idxmax(axis=1)
    else:
        label_col = find_col(df, ["diagnosis", "label", "class", "target"])
        if label_col is None:
            raise ValueError("Tidak menemukan kolom label MEL/NV/BCC atau kolom one-hot MEL, NV, BCC.")
        df = df.rename(columns={label_col: "diagnosis"})
        df["diagnosis"] = df["diagnosis"].astype(str).str.upper().str.strip()
        aliases = {"MELANOMA":"MEL", "NEVUS":"NV", "MELANOCYTIC NEVUS":"NV",
                   "BASAL CELL CARCINOMA":"BCC"}
        df["diagnosis"] = df["diagnosis"].replace(aliases)
    df = df[df["diagnosis"].isin(CLASS_ORDER)].copy()
    return df

def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    image_col = find_col(df, ["image", "image_id", "isic_id", "isic"])
    if image_col is None:
        raise ValueError("CSV metadata harus memiliki kolom image atau image_id.")
    rename = {image_col: "image"}
    for target, candidates in {
        "age": ["age_approx", "age"],
        "sex": ["sex", "gender"],
        "site": ["anatom_site_general", "anatomical_site", "site"],
        "lesion_id": ["lesion_id"],
    }.items():
        col = find_col(df, candidates)
        if col:
            rename[col] = target
    df = df.rename(columns=rename)
    df["image"] = df["image"].map(normalize_image_id)
    keep = ["image"] + [c for c in ["age", "sex", "site", "lesion_id"] if c in df.columns]
    return df[keep].drop_duplicates("image")

def merge_data(gt: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if meta is not None:
        merged = gt.merge(meta, on="image", how="left")
    else:
        merged = gt.copy()
    for col in ["age", "sex", "site", "lesion_id"]:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged

def get_image_path(image_id: str, image_dir: str) -> Optional[Path]:
    base = Path(image_dir)
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = base / f"{image_id}{ext}"
        if p.exists():
            return p
    # Also support ZIP files that contain a nested top-level folder.
    if base.exists():
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            matches = list(base.rglob(f"{image_id}{ext}"))
            if matches:
                return matches[0]
    return None

def load_image(image_id: str, image_dir: str):
    p = get_image_path(image_id, image_dir)
    if p is None:
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    # Group by lesion_id where available to reduce leakage.
    work = df.copy().reset_index(drop=True)
    work["split"] = ""
    if "lesion_id" in work.columns and work["lesion_id"].notna().any():
        # Group-aware split: assign lesion groups, stratification approximated by group label.
        groups = work[work["lesion_id"].notna()].groupby("lesion_id")["diagnosis"].agg(lambda x: x.mode().iloc[0]).reset_index()
        try:
            train_g, temp_g = train_test_split(groups, test_size=.30, random_state=seed, stratify=groups["diagnosis"])
            val_g, test_g = train_test_split(temp_g, test_size=.50, random_state=seed, stratify=temp_g["diagnosis"])
            train_ids, val_ids, test_ids = set(train_g.lesion_id), set(val_g.lesion_id), set(test_g.lesion_id)
            mask = work["lesion_id"].notna()
            work.loc[mask & work.lesion_id.isin(train_ids), "split"] = "Train"
            work.loc[mask & work.lesion_id.isin(val_ids), "split"] = "Validation"
            work.loc[mask & work.lesion_id.isin(test_ids), "split"] = "Test"
            # Records without lesion_id are split separately.
            no_id = work[work["lesion_id"].isna()].copy()
            if len(no_id):
                tr, tmp = train_test_split(no_id.index, test_size=.30, random_state=seed,
                                           stratify=no_id["diagnosis"])
                va, te = train_test_split(tmp, test_size=.50, random_state=seed,
                                          stratify=work.loc[tmp, "diagnosis"])
                work.loc[tr, "split"] = "Train"
                work.loc[va, "split"] = "Validation"
                work.loc[te, "split"] = "Test"
            return work
        except Exception:
            pass
    tr, tmp = train_test_split(work.index, test_size=.30, random_state=seed, stratify=work["diagnosis"])
    va, te = train_test_split(tmp, test_size=.50, random_state=seed, stratify=work.loc[tmp, "diagnosis"])
    work.loc[tr, "split"] = "Train"
    work.loc[va, "split"] = "Validation"
    work.loc[te, "split"] = "Test"
    return work

def make_metric_card(label, value, hint):
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-help">{hint}</div></div>""", unsafe_allow_html=True)

def page_header(title, subtitle, icon=""):
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:2px"><div style="font-size:27px;color:#078F83">{icon}</div><div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div></div>', unsafe_allow_html=True)

def display_distribution(df):
    counts = df["diagnosis"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    chart = pd.DataFrame({"Kelas": counts.index, "Jumlah Gambar": counts.values})
    fig = px.bar(chart, x="Kelas", y="Jumlah Gambar", color="Kelas",
                 color_discrete_map=CLASS_COLORS, text="Jumlah Gambar")
    fig.update_layout(height=255, margin=dict(l=5,r=5,t=10,b=5),
                      showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="DM Sans", color="#233B4A"),
                      xaxis_title="", yaxis_title="Jumlah Gambar")
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

def dataset_status():
    if st.session_state.data_df is None:
        return "Dataset belum dimuat"
    return f"{len(st.session_state.data_df):,} gambar · 3 kelas"

# ============================================================
# MODEL: EFFICIENTNET-B0
# ============================================================
if TORCH_AVAILABLE:
    class LesionDataset(Dataset):
        def __init__(self, frame, image_dir, train=False):
            self.frame = frame.reset_index(drop=True)
            self.image_dir = image_dir
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x:x),
                transforms.RandomVerticalFlip() if train else transforms.Lambda(lambda x:x),
                transforms.RandomRotation(15) if train else transforms.Lambda(lambda x:x),
                transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.08) if train else transforms.Lambda(lambda x:x),
                transforms.ToTensor(),
                transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
            ])
        def __len__(self): return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            p = get_image_path(row["image"], self.image_dir)
            if p is None:
                raise FileNotFoundError(f"Gambar tidak ditemukan: {row['image']}")
            image = Image.open(p).convert("RGB")
            x = self.transform(image)
            y = CLASS_ORDER.index(row["diagnosis"])
            return x, y

    def build_efficientnet(pretrained=True):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 3)
        return model

    def train_cnn(frame, image_dir, epochs, batch_size, lr, pretrained, progress):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_df = frame[frame.split == "Train"].copy()
        val_df = frame[frame.split == "Validation"].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("Split training/validation kosong.")
        train_ds = LesionDataset(train_df, image_dir, train=True)
        val_ds = LesionDataset(val_df, image_dir, train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_efficientnet(pretrained=pretrained).to(device)
        # Weighted CE based on training split only.
        counts = train_df["diagnosis"].value_counts().reindex(CLASS_ORDER, fill_value=1).values
        weights = len(train_df) / (3 * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=.5)

        history = []
        best_f1 = -1
        best_state = None
        for epoch in range(epochs):
            model.train()
            train_loss, train_true, train_pred = 0.0, [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                train_true.extend(y.detach().cpu().numpy())
                train_pred.extend(logits.argmax(1).detach().cpu().numpy())
            model.eval()
            val_loss, val_true, val_pred = 0.0, [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * len(y)
                    val_true.extend(y.cpu().numpy())
                    val_pred.extend(logits.argmax(1).cpu().numpy())
            tr_loss = train_loss / len(train_ds)
            va_loss = val_loss / len(val_ds)
            va_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            scheduler.step(va_loss)
            history.append({"epoch": epoch+1, "train_loss": tr_loss, "val_loss": va_loss,
                            "train_accuracy": accuracy_score(train_true, train_pred),
                            "val_accuracy": accuracy_score(val_true, val_pred), "val_macro_f1": va_f1})
            progress.progress((epoch+1)/epochs, text=f"Training epoch {epoch+1}/{epochs} · val Macro-F1 {va_f1:.4f}")
            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        return model, pd.DataFrame(history), device

    def evaluate_cnn(model, frame, image_dir, batch_size=32):
        device = next(model.parameters()).device
        test_df = frame[frame.split == "Test"].copy()
        if len(test_df) == 0:
            raise ValueError("Test split kosong.")
        ds = LesionDataset(test_df, image_dir, train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        y_true, y_pred, y_prob = [], [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_pred.extend(prob.argmax(axis=1))
                y_prob.extend(prob)
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        return {
            "accuracy": accuracy_score(y_true,y_pred),
            "macro_f1": f1_score(y_true,y_pred,average="macro",zero_division=0),
            "precision": precision_score(y_true,y_pred,average="macro",zero_division=0),
            "recall": recall_score(y_true,y_pred,average="macro",zero_division=0),
            "cm": confusion_matrix(y_true,y_pred,labels=[0,1,2]),
            "report": classification_report(y_true,y_pred,labels=[0,1,2],target_names=CLASS_ORDER,
                                            output_dict=True,zero_division=0),
            "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        }

# ============================================================
# MANUAL DATA STORAGE
# ============================================================
def init_manual_db():
    with sqlite3.connect(MANUAL_DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS manual_records (
            record_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            site TEXT,
            diagnosis TEXT NOT NULL
        )""")
        conn.commit()


def save_manual_record(uploaded_image, age, sex, site, diagnosis):
    record_id = f"MANUAL_{uuid.uuid4().hex[:10].upper()}"
    image = Image.open(uploaded_image).convert("RGB")
    image_path = MANUAL_IMAGE_DIR / f"{record_id}.jpg"
    image.save(image_path, format="JPEG", quality=92)
    relative_path = str(image_path.relative_to(BASE_DIR))
    with sqlite3.connect(MANUAL_DB) as conn:
        conn.execute(
            "INSERT INTO manual_records VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, datetime.now().isoformat(timespec="seconds"), relative_path,
             age if age != "" else None, sex, site, diagnosis)
        )
        conn.commit()
    return record_id


def get_manual_record_count():
    init_manual_db()
    with sqlite3.connect(MANUAL_DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM manual_records").fetchone()[0]


init_manual_db()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:8px 0 15px"><div class="brand"><span class="brand-mark">✚</span>DermaSense</div><div class="brand-sub">Skin Lesion Classification<br>(ISIC 2019)</div></div>', unsafe_allow_html=True)

    nav_labels = {
        "⌂  Dashboard": "Dashboard",
        "▤  Dataset": "Dataset",
        "⌘  Model CNN": "Model CNN",
        "◈  Model Multimodal": "Model Multimodal",
        "▥  Evaluation": "Evaluation",
        "ⓘ  About": "About",
    }
    selected_nav = st.radio(
        "NAVIGATION", list(nav_labels.keys()),
        format_func=lambda x: x,
        label_visibility="collapsed",
        key="main_navigation"
    )
    page = nav_labels[selected_nav]

    st.markdown("<div style='height:8px;border-bottom:1px solid rgba(180,225,222,.35);margin-bottom:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-input-card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;align-items:center;gap:9px"><span style="font-size:24px">▣</span><div><div class="sidebar-section-title">Input Data</div><div class="sidebar-hint">Tambah data baru (citra + metadata) ke dataset.</div></div></div>', unsafe_allow_html=True)

    def reset_sidebar_fields():
        for widget_key in ["sb_upload", "sb_age", "sb_sex", "sb_site", "sb_diagnosis"]:
            st.session_state.pop(widget_key, None)

    with st.form("sidebar_manual_input", clear_on_submit=True):
        uploaded_lesion = st.file_uploader(
            "Upload Citra Lesi *", type=["jpg", "jpeg", "png"],
            help="Format JPG/JPEG/PNG, maksimal 10 MB.", key="sb_upload"
        )
        age_input = st.text_input("Usia (tahun)", placeholder="Contoh: 45", key="sb_age")
        sex_input = st.selectbox("Jenis Kelamin", ["Pilih...", "Female", "Male"], key="sb_sex")
        site_input = st.selectbox(
            "Lokasi Anatomi", ["Pilih...", "Head/Neck", "Anterior torso", "Posterior torso",
            "Lateral torso", "Upper extremity", "Lower extremity", "Palms/Soles", "Oral/Genital", "Other"],
            key="sb_site"
        )
        diagnosis_input = st.selectbox(
            "Diagnosis (Label) *", ["Pilih...", "MEL", "NV", "BCC"],
            format_func=lambda x: "Pilih... (MEL / NV / BCC)" if x == "Pilih..." else f"{x} — {CLASS_NAMES[x]}",
            key="sb_diagnosis"
        )
        save_manual = st.form_submit_button("＋  Simpan Data", type="primary", use_container_width=True)

    st.button("↻  Reset Form", use_container_width=True, key="reset_sidebar_form", on_click=reset_sidebar_fields)

    if save_manual:
        validation_errors = []
        if uploaded_lesion is None:
            validation_errors.append("Pilih gambar lesi terlebih dahulu.")
        if diagnosis_input == "Pilih...":
            validation_errors.append("Pilih label diagnosis MEL, NV, atau BCC.")
        if age_input.strip() and (not age_input.strip().isdigit() or not 0 <= int(age_input.strip()) <= 120):
            validation_errors.append("Usia harus berupa angka 0–120.")
        if uploaded_lesion is not None and uploaded_lesion.size > 10 * 1024 * 1024:
            validation_errors.append("Ukuran gambar maksimal 10 MB.")
        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            try:
                new_id = save_manual_record(
                    uploaded_lesion,
                    int(age_input) if age_input.strip() else "",
                    None if sex_input == "Pilih..." else sex_input,
                    None if site_input == "Pilih..." else site_input,
                    diagnosis_input,
                )
                st.success(f"Data tersimpan · {new_id}")
            except Exception as exc:
                st.error(f"Gagal menyimpan data: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px;border-bottom:1px solid rgba(180,225,222,.35);margin-bottom:10px'></div>", unsafe_allow_html=True)
    st.markdown("**Dataset Aktif**")
    manual_count = get_manual_record_count()
    st.markdown(
        f'<div style="background:#0B5960;border-radius:13px;padding:12px;font-size:12px;color:white">'
        f'● {dataset_status()}<br><br>3 kelas: MEL · NV · BCC<br><br>'
        f'Data input manual: {manual_count}</div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.caption("Research prototype · Bukan alat diagnosis klinis")

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    page_header("Dashboard", "Ringkasan dataset ISIC 2019 dan status pengembangan model.", "◈")
    df = st.session_state.data_df
    if df is None:
        a,b,c,d = st.columns(4)
        with a: make_metric_card("TOTAL GAMBAR", "20,720*", "Jumlah tiga kelas pada ISIC 2019")
        with b: make_metric_card("TRAINING", "70%", "Rasio split yang direncanakan")
        with c: make_metric_card("VALIDATION", "15%", "Rasio split yang direncanakan")
        with d: make_metric_card("TESTING", "15%", "Rasio split yang direncanakan")
        st.info("Muat CSV Ground Truth ISIC 2019 pada menu Dataset untuk menampilkan statistik aktual.")
    else:
        a,b,c,d = st.columns(4)
        with a: make_metric_card("TOTAL GAMBAR", f"{len(df):,}", "Data yang berhasil dimuat")
        with b: make_metric_card("KELAS MEL", f"{(df.diagnosis=='MEL').sum():,}", "Melanoma")
        with c: make_metric_card("KELAS NV", f"{(df.diagnosis=='NV').sum():,}", "Melanocytic Nevus")
        with d: make_metric_card("KELAS BCC", f"{(df.diagnosis=='BCC').sum():,}", "Basal Cell Carcinoma")
        x,y = st.columns([1.1,1])
        with x:
            st.markdown('<div class="panel"><div class="panel-title">Distribusi Kelas</div>', unsafe_allow_html=True)
            display_distribution(df)
            st.markdown("</div>", unsafe_allow_html=True)
        with y:
            st.markdown('<div class="panel"><div class="panel-title">Status Model CNN</div>', unsafe_allow_html=True)
            st.metric("EfficientNet-B0", st.session_state.model_status)
            st.caption("Model klasifikasi tiga kelas lesi kulit.")
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DATASET EXPLORER (MATCHES USER REFERENCE)
# ============================================================
elif page == "Dataset":
    page_header("Dataset Explorer", "Eksplorasi dataset ISIC 2019 untuk tiga kelas (MEL, NV, BCC) beserta metadata klinis.", "▤")
    t_overview, t_viewer, t_metadata, t_pre = st.tabs(["▥ Overview", "▧ Data Viewer", "▦ Metadata", "⚙ Preprocessing"])

    with t_overview:
        left, mid, chart, samples = st.columns([1.02,1.12,1.18,1.35], gap="small")
        with left:
            st.markdown('<div class="panel"><div class="panel-title">▰ Pilih Dataset</div>', unsafe_allow_html=True)
            source = st.radio("Sumber dataset", ["ISIC 2019 (3 Kelas)", "Upload Dataset (CSV)"], label_visibility="collapsed")
            gt_upload = st.file_uploader("Upload Ground Truth CSV", type="csv", key="gt_upload")
            meta_upload = st.file_uploader("Upload Metadata CSV (opsional)", type="csv", key="meta_upload")
            image_dir_input = st.text_input("Folder gambar", value=st.session_state.image_dir,
                                            help="Path folder yang berisi file JPG ISIC.")
            zip_upload = st.file_uploader("Upload ZIP gambar (opsional)", type="zip", key="image_zip")
            load_clicked = st.button("Muat Dataset", type="primary", use_container_width=True)
            if load_clicked:
                try:
                    if gt_upload is None:
                        st.warning("Pilih file Ground Truth CSV terlebih dahulu.")
                    else:
                        gt = prepare_ground_truth(pd.read_csv(gt_upload))
                        meta = prepare_metadata(pd.read_csv(meta_upload)) if meta_upload else None
                        st.session_state.gt = gt
                        st.session_state.meta = meta
                        st.session_state.data_df = merge_data(gt, meta)
                        st.session_state.image_dir = image_dir_input
                        if zip_upload:
                            with zipfile.ZipFile(zip_upload) as zf:
                                for member in zf.infolist():
                                    destination = (IMAGE_DIR / member.filename).resolve()
                                    if destination == IMAGE_DIR.resolve() or IMAGE_DIR.resolve() in destination.parents:
                                        zf.extract(member, IMAGE_DIR)
                            st.session_state.image_dir = str(IMAGE_DIR)
                        st.session_state.split_df = None
                        st.success(f"Dataset dimuat: {len(st.session_state.data_df):,} gambar.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal memuat dataset: {e}")
            st.markdown('<div class="small-muted">Dataset resmi: ISIC 2019 · Filter kelas MEL, NV, BCC.</div></div>', unsafe_allow_html=True)

        df = st.session_state.data_df
        with mid:
            st.markdown('<div class="panel"><div class="panel-title">ⓘ Ringkasan Dataset</div>', unsafe_allow_html=True)
            if df is not None:
                counts = df.diagnosis.value_counts().reindex(CLASS_ORDER, fill_value=0)
                total = len(df)
                train_n, val_n, test_n = round(total*.70), round(total*.15), total-round(total*.70)-round(total*.15)
                rows = [("▧ Total Gambar", f"{total:,}"), ("▱ Training Set (70%)", f"{train_n:,}"),
                        ("▧ Validation Set (15%)", f"{val_n:,}"), ("▤ Testing Set (15%)", f"{test_n:,}"),
                        ("▥ Jumlah Kelas", "3 (MEL, NV, BCC)")]
            else:
                rows = [("▧ Total Gambar", "20,720*"), ("▱ Training Set (70%)", "14,504*"),
                        ("▧ Validation Set (15%)", "3,108*"), ("▤ Testing Set (15%)", "3,108*"),
                        ("▥ Jumlah Kelas", "3 (MEL, NV, BCC)")]
            for label, value in rows:
                st.markdown(f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid #E8EEF0;padding:7px 0;font-size:12px"><span>{label}</span><b>{value}</b></div>', unsafe_allow_html=True)
            if df is None:
                st.caption("*Angka referensi sebelum filter tambahan; muat CSV untuk statistik aktual.")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart:
            st.markdown('<div class="panel"><div class="panel-title">▥ Distribusi Kelas</div>', unsafe_allow_html=True)
            if df is not None:
                display_distribution(df)
            else:
                counts = pd.DataFrame({"Kelas": CLASS_ORDER, "Jumlah Gambar":[EXPECTED_COUNTS[c] for c in CLASS_ORDER]})
                fig = px.bar(counts, x="Kelas", y="Jumlah Gambar", color="Kelas",
                             color_discrete_map=CLASS_COLORS, text="Jumlah Gambar")
                fig.update_layout(height=255, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                                  plot_bgcolor="white", paper_bgcolor="white", xaxis_title="")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Distribusi referensi tiga kelas.")
            st.markdown("</div>", unsafe_allow_html=True)

        with samples:
            st.markdown('<div class="panel"><div class="panel-title">▣ Contoh Citra per Kelas</div>', unsafe_allow_html=True)
            df = st.session_state.data_df
            if df is not None:
                cols = st.columns(3)
                for i, cls in enumerate(CLASS_ORDER):
                    sample = df[df.diagnosis == cls].head(1)
                    with cols[i]:
                        if len(sample):
                            im = load_image(sample.iloc[0].image, st.session_state.image_dir)
                            if im: st.image(im, use_container_width=True)
                            else: st.markdown('<div style="background:#EDF4F5;height:85px;border-radius:10px;display:flex;align-items:center;justify-content:center">No image</div>', unsafe_allow_html=True)
                        st.caption(f"**{cls}**")
            else:
                st.info("Muat dataset dan folder gambar untuk menampilkan sampel asli.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        cnn_col, meta_col = st.columns([1.12,1], gap="small")
        with cnn_col:
            st.markdown('<div class="panel"><div class="panel-title">◉ Model CNN – EfficientNet-B0</div><div class="panel-caption">Klasifikasi citra lesi kulit menggunakan EfficientNet-B0 sebagai feature extractor.</div>', unsafe_allow_html=True)
            ta, tb = st.tabs(["Arsitektur Model", "Konfigurasi & Training"])
            with ta:
                st.markdown("""
                <div style="border:1px solid #E4ECEF;border-radius:13px;padding:12px;background:#FCFEFF">
                <b style="font-size:13px">Arsitektur CNN</b>
                <div style="display:flex;align-items:center;justify-content:space-between;gap:5px;margin-top:16px;flex-wrap:wrap">
                <div style="background:#F9D8CE;border-radius:9px;padding:12px 8px;text-align:center;font-size:11px"><b>Input</b><br>224×224×3</div>
                <span>➜</span><div style="background:#D9F4F0;border-radius:9px;padding:13px 8px;text-align:center;font-size:11px"><b>EfficientNet-B0</b><br>Feature Extractor</div>
                <span>➜</span><div style="background:#D9F4F0;border-radius:9px;padding:13px 8px;text-align:center;font-size:11px"><b>GAP</b><br>Global Average Pooling</div>
                <span>➜</span><div style="background:#D9F4F0;border-radius:9px;padding:13px 8px;text-align:center;font-size:11px"><b>Dense</b><br>256</div>
                <span>➜</span><div style="background:#D9F4F0;border-radius:9px;padding:13px 8px;text-align:center;font-size:11px"><b>Softmax</b><br>3 Kelas</div>
                </div><div style="display:flex;justify-content:flex-end;gap:7px;margin-top:12px"><span style="background:#FFE2DF;padding:5px 15px;border-radius:8px;font-size:11px">MEL</span><span style="background:#DDF4E4;padding:5px 15px;border-radius:8px;font-size:11px">NV</span><span style="background:#DFEBFF;padding:5px 15px;border-radius:8px;font-size:11px">BCC</span></div></div>
                """, unsafe_allow_html=True)
            with tb:
                st.caption("Konfigurasi juga tersedia pada menu Model CNN.")
                st.write("Input: 224×224 RGB · Optimizer: AdamW · Loss: Weighted Cross Entropy")
            st.markdown("</div>", unsafe_allow_html=True)

        with meta_col:
            st.markdown('<div class="panel"><div class="panel-title">▤ Metadata & Missing Data</div><div class="panel-caption">Gunakan metadata klinis dan atur tingkat missing metadata untuk menguji ketahanan model.</div>', unsafe_allow_html=True)
            m1,m2 = st.columns([1.05,1])
            with m1:
                st.markdown("**Metadata Klinis**")
                st.text_input("Usia (tahun)", placeholder="Contoh: 45", disabled=True)
                st.selectbox("Jenis Kelamin", ["Pilih...", "Female", "Male"], disabled=True)
                st.selectbox("Lokasi Anatomi", ["Pilih...", "Head/Neck", "Torso", "Upper extremity", "Lower extremity"], disabled=True)
            with m2:
                st.markdown("**Simulasi Missing Metadata**")
                st.caption("Persentase nilai metadata yang dihilangkan.")
                missing_rate = st.radio("Missing rate", [0,25,50,75,100],
                                        format_func=lambda x: "0% (Complete)" if x==0 else ("100% (Image Only)" if x==100 else f"{x}%"),
                                        horizontal=False, key="overview_missing")
                st.session_state["missing_rate"] = missing_rate
            st.markdown("</div>", unsafe_allow_html=True)

        if df is not None:
            st.markdown('<div class="panel" style="margin-top:10px"><div class="panel-title">Metadata Overview</div>', unsafe_allow_html=True)
            m1,m2,m3 = st.columns(3)
            with m1:
                if df.age.notna().any():
                    age_df = pd.to_numeric(df.age, errors="coerce").dropna()
                    fig = px.histogram(age_df, nbins=18, labels={"value":"Usia","count":"Jumlah"})
                    fig.update_layout(height=180,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)
                else: st.info("Metadata usia belum tersedia.")
            with m2:
                if df.sex.notna().any():
                    st.plotly_chart(px.pie(df.dropna(subset=["sex"]),names="sex",hole=.55),use_container_width=True)
                else: st.info("Metadata sex belum tersedia.")
            with m3:
                if df.site.notna().any():
                    top = df.site.value_counts().head(5).rename_axis("Anatomical site").reset_index(name="Jumlah")
                    st.plotly_chart(px.bar(top,x="Jumlah",y="Anatomical site",orientation="h"),use_container_width=True)
                else: st.info("Metadata lokasi belum tersedia.")
            st.markdown("</div>", unsafe_allow_html=True)

    with t_viewer:
        df = st.session_state.data_df
        if df is None:
            st.info("Muat dataset terlebih dahulu pada tab Overview.")
        else:
            f1,f2 = st.columns(2)
            with f1: cls = st.selectbox("Filter kelas", ["Semua"]+CLASS_ORDER, key="viewer_cls")
            with f2: query = st.text_input("Cari Image ID", key="viewer_search")
            view = df.copy()
            if cls != "Semua": view = view[view.diagnosis == cls]
            if query: view = view[view.image.str.contains(query,case=False,na=False)]
            st.caption(f"{len(view):,} record ditemukan.")
            st.dataframe(view.head(1000),use_container_width=True,hide_index=True)
            st.markdown("### Data tambahan dari Input Data sidebar")
            with sqlite3.connect(MANUAL_DB) as conn:
                manual_df = pd.read_sql_query("SELECT record_id, created_at, age, sex, site, diagnosis FROM manual_records ORDER BY created_at DESC", conn)
            if manual_df.empty:
                st.caption("Belum ada data manual yang disimpan.")
            else:
                st.dataframe(manual_df, use_container_width=True, hide_index=True)
                st.download_button("Download data manual (CSV)", manual_df.to_csv(index=False).encode("utf-8"), "dermasense_manual_records.csv", "text/csv")
            if len(view):
                selected = st.selectbox("Pilih gambar untuk preview",view.image.tolist()[:2000])
                im = load_image(selected,st.session_state.image_dir)
                if im: st.image(im,caption=f"{selected} · {view.loc[view.image==selected,'diagnosis'].iloc[0]}",width=360)
                else: st.warning("File gambar tidak ditemukan. Periksa folder gambar.")

    with t_metadata:
        df = st.session_state.data_df
        if df is None: st.info("Muat dataset terlebih dahulu.")
        else:
            st.markdown("### Kelengkapan metadata klinis")
            fields = [c for c in ["age","sex","site"] if c in df.columns]
            comp = pd.DataFrame({"Metadata":fields,
                                 "Tersedia":[df[c].notna().sum() for c in fields],
                                 "Hilang":[df[c].isna().sum() for c in fields]})
            if len(comp):
                comp["Missing (%)"] = (comp["Hilang"]/len(df)*100).round(2)
                st.dataframe(comp,use_container_width=True,hide_index=True)
                fig = px.bar(comp,x="Metadata",y="Missing (%)",color_discrete_sequence=["#079B8B"])
                st.plotly_chart(fig,use_container_width=True)
            st.markdown("### Preview metadata")
            st.dataframe(df[["image","diagnosis"]+fields].head(100),use_container_width=True,hide_index=True)

    with t_pre:
        df = st.session_state.data_df
        st.markdown("### Pengaturan preprocessing")
        p1,p2,p3 = st.columns(3)
        with p1: size = st.selectbox("Image size",["224 × 224","256 × 256","300 × 300"])
        with p2: norm = st.selectbox("Normalisasi",["ImageNet mean/std","Rescale 0–1"])
        with p3: aug = st.multiselect("Augmentasi training",["Horizontal flip","Vertical flip","Rotation ±15°","Brightness/contrast"],default=["Horizontal flip","Rotation ±15°"])
        st.markdown('<div class="soft-note">Augmentasi hanya diterapkan pada data training. Validation dan testing menggunakan preprocessing deterministik.</div>',unsafe_allow_html=True)
        if df is not None:
            if st.button("Buat split dataset 70/15/15",type="primary"):
                if not SKLEARN_AVAILABLE: st.error("scikit-learn belum terpasang.")
                else:
                    try:
                        split = make_splits(df)
                        st.session_state.split_df = split
                        st.success("Split berhasil dibuat.")
                    except Exception as e: st.error(str(e))
            if st.session_state.split_df is not None:
                split = st.session_state.split_df
                st.dataframe(split["split"].value_counts().rename_axis("Split").reset_index(name="Jumlah"),hide_index=True)
                st.download_button("Download CSV hasil split",split.to_csv(index=False).encode(), "isic2019_3class_split.csv","text/csv")

# ============================================================
# MODEL CNN
# ============================================================
elif page == "Model CNN":
    page_header("Model CNN – EfficientNet-B0", "Training model klasifikasi citra lesi kulit untuk MEL, NV, dan BCC.", "◉")
    if not TORCH_AVAILABLE:
        st.error("PyTorch/torchvision belum tersedia. Install dependency dari requirements.txt.")
    df = st.session_state.data_df
    if df is None:
        st.warning("Muat dataset ISIC 2019 terlebih dahulu melalui menu Dataset.")
    else:
        c1,c2 = st.columns([1.1,1])
        with c1:
            st.markdown('<div class="panel"><div class="panel-title">Konfigurasi Training</div>',unsafe_allow_html=True)
            epochs = st.number_input("Epochs",min_value=1,max_value=100,value=10,step=1)
            batch = st.selectbox("Batch Size",[8,16,32],index=1)
            lr = st.select_slider("Learning Rate",options=[0.00001,0.00005,0.0001,0.0005,0.001],value=0.0001,format_func=lambda x:f"{x:g}")
            pretrained = st.checkbox("Gunakan pretrained ImageNet weights",value=True)
            seed = st.number_input("Random seed",min_value=1,max_value=9999,value=42)
            st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="panel"><div class="panel-title">Arsitektur CNN</div>',unsafe_allow_html=True)
            st.markdown("**Input image** 224 × 224 × 3")
            st.markdown("↓")
            st.markdown("**EfficientNet-B0** · pretrained backbone")
            st.markdown("↓")
            st.markdown("**Global Average Pooling**")
            st.markdown("↓")
            st.markdown("**Dropout + Dense (3 output)**")
            st.markdown("↓")
            st.markdown("**Softmax: MEL · NV · BCC**")
            st.markdown("</div>",unsafe_allow_html=True)
        if st.session_state.split_df is None:
            if st.button("Buat split otomatis 70/15/15"):
                st.session_state.split_df = make_splits(df)
                st.rerun()
        split_df = st.session_state.split_df
        if split_df is not None:
            st.markdown("**Ringkasan split**")
            st.dataframe(split_df.groupby(["split","diagnosis"]).size().unstack(fill_value=0),use_container_width=True)
            image_dir = st.session_state.image_dir
            missing_files = [x for x in split_df.image.head(1000) if get_image_path(x,image_dir) is None]
            if missing_files:
                st.warning(f"Tidak menemukan sebagian file gambar di folder: {image_dir}. Contoh ID: {missing_files[:3]}")
            if st.button("▶ Train Model",type="primary",use_container_width=True,disabled=not TORCH_AVAILABLE):
                if not SKLEARN_AVAILABLE:
                    st.error("scikit-learn belum terpasang.")
                else:
                    try:
                        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                        progress = st.progress(0,text="Menyiapkan training...")
                        model, history, device = train_cnn(split_df,image_dir,int(epochs),int(batch),float(lr),pretrained,progress)
                        st.session_state.trained_model = model
                        st.session_state.training_history = history
                        st.session_state.model_status = "Training selesai"
                        model_path = MODEL_DIR / "dermasense_efficientnet_b0.pth"
                        torch.save({"state_dict":model.state_dict(),"classes":CLASS_ORDER},model_path)
                        st.session_state.last_model_path = str(model_path)
                        st.success(f"Training selesai pada {device}. Best validation Macro-F1: {history.val_macro_f1.max():.4f}")
                    except Exception as e:
                        st.error(f"Training gagal: {e}")
        if st.session_state.training_history is not None:
            hist = st.session_state.training_history
            st.markdown("### Training history")
            h1,h2 = st.columns(2)
            with h1:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=hist.epoch,y=hist.train_loss,name="Train loss"))
                fig.add_trace(go.Scatter(x=hist.epoch,y=hist.val_loss,name="Validation loss"))
                fig.update_layout(title="Loss per epoch",height=300)
                st.plotly_chart(fig,use_container_width=True)
            with h2:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=hist.epoch,y=hist.train_accuracy,name="Train accuracy"))
                fig.add_trace(go.Scatter(x=hist.epoch,y=hist.val_accuracy,name="Validation accuracy"))
                fig.update_layout(title="Accuracy per epoch",height=300,yaxis_range=[0,1])
                st.plotly_chart(fig,use_container_width=True)
            st.dataframe(hist,use_container_width=True,hide_index=True)

# ============================================================
# MODEL MULTIMODAL
# ============================================================
elif page == "Model Multimodal":
    page_header("Model Multimodal", "Rancangan penggabungan fitur citra EfficientNet-B0 dan metadata klinis.", "▱")
    st.markdown('<div class="panel"><div class="panel-title">Arsitektur Multimodal</div>',unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown("#### Image Branch")
        st.info("Input RGB → EfficientNet-B0 → Global Average Pooling → Image Features")
    with b:
        st.markdown("#### Metadata Branch")
        st.info("Age + Sex + Anatomical Site → Imputation + Encoding + Missing Mask → MLP")
    with c:
        st.markdown("#### Fusion & Output")
        st.success("Mask-aware Fusion → Dense → Softmax → MEL / NV / BCC")
    st.markdown("</div>",unsafe_allow_html=True)
    st.warning("Halaman ini merupakan rancangan arsitektur. Training multimodal belum diimplementasikan pada prototype CNN ini; jangan menganggap hasilnya sudah tersedia.")
    rate = st.radio("Simulasi missing metadata", [0,25,50,75,100],
                    format_func=lambda x:"Complete (0%)" if x==0 else ("Image only (100%)" if x==100 else f"{x}% missing"),
                    horizontal=True)
    st.caption(f"Skenario eksperimen terpilih: {rate}% nilai metadata dimasking.")

# ============================================================
# EVALUATION
# ============================================================
elif page == "Evaluation":
    page_header("Evaluasi Model", "Evaluasi model CNN menggunakan data testing yang tidak digunakan selama training.", "▥")
    if st.session_state.trained_model is None:
        st.info("Belum ada model terlatih. Selesaikan training pada menu Model CNN terlebih dahulu.")
    elif st.session_state.split_df is None:
        st.warning("Split dataset tidak tersedia.")
    else:
        if st.button("Evaluasi pada Test Set",type="primary"):
            try:
                with st.spinner("Menghitung metrik pada test set..."):
                    result = evaluate_cnn(st.session_state.trained_model,st.session_state.split_df,st.session_state.image_dir)
                    st.session_state.eval_result = result
                st.success("Evaluasi selesai.")
            except Exception as e: st.error(f"Evaluasi gagal: {e}")
        result = st.session_state.eval_result
        if result:
            a,b,c,d=st.columns(4)
            with a: make_metric_card("ACCURACY",f"{result['accuracy']:.4f}","Test set")
            with b: make_metric_card("MACRO-F1",f"{result['macro_f1']:.4f}","Metrik utama")
            with c: make_metric_card("PRECISION",f"{result['precision']:.4f}","Macro average")
            with d: make_metric_card("RECALL",f"{result['recall']:.4f}","Macro average")
            x,y=st.columns(2)
            with x:
                fig=px.imshow(result["cm"],x=CLASS_ORDER,y=CLASS_ORDER,text_auto=True,
                              color_continuous_scale="Teal",labels=dict(x="Predicted",y="Actual",color="Jumlah"))
                fig.update_layout(title="Confusion Matrix",height=400)
                st.plotly_chart(fig,use_container_width=True)
            with y:
                rep=pd.DataFrame(result["report"]).T
                st.markdown("### Classification Report")
                st.dataframe(rep.round(4),use_container_width=True)
            st.download_button("Download classification report",rep.to_csv().encode(),"classification_report.csv","text/csv")
            # Multiclass ROC-AUC one-vs-rest
            try:
                y_bin=label_binarize(result["y_true"],classes=[0,1,2])
                auc=roc_auc_score(y_bin,result["y_prob"],average="macro",multi_class="ovr")
                st.metric("Macro ROC-AUC (OvR)",f"{auc:.4f}")
            except Exception:
                st.caption("ROC-AUC tidak dapat dihitung pada kondisi test set ini.")

# ============================================================
# ABOUT
# ============================================================
else:
    page_header("About & Methods", "Informasi penelitian dan batasan sistem DermaSense.", "ⓘ")
    st.markdown("""
    <div class="panel">
    <h3>DermaSense</h3>
    <p>Prototype penelitian klasifikasi lesi kulit menggunakan subset tiga kelas ISIC 2019:</p>
    <ul><li><b>MEL</b> — Melanoma</li><li><b>NV</b> — Melanocytic Nevus</li><li><b>BCC</b> — Basal Cell Carcinoma</li></ul>
    <p>Model CNN yang digunakan adalah EfficientNet-B0. Arah pengembangan multimodal menggabungkan citra dan metadata klinis serta menguji ketahanan terhadap missing metadata.</p>
    </div>
    """,unsafe_allow_html=True)
    st.warning("DermaSense merupakan prototipe penelitian akademik, bukan alat diagnosis. Hasil model tidak boleh digunakan sebagai pengganti pemeriksaan dokter.")

st.markdown('<div style="text-align:center;color:#93A5AE;font-size:10px;padding:25px 0 5px">DERMASENSE · ISIC 2019 · MEL / NV / BCC · RESEARCH PROTOTYPE · NOT FOR CLINICAL DIAGNOSIS</div>',unsafe_allow_html=True)
