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
    try:
        import multimodal_lstm as mm_lstm
        MULTIMODAL_LSTM_AVAILABLE = True
    except Exception:
        mm_lstm = None
        MULTIMODAL_LSTM_AVAILABLE = False
except Exception:
    TORCH_AVAILABLE = False
    MULTIMODAL_LSTM_AVAILABLE = False
    mm_lstm = None

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
CLASS_COLORS = {"MEL": "#F47D62", "NV": "#079B8B", "BCC": "#58C7BE"}
CLASS_ORDER = ["MEL", "NV", "BCC"]
EXPECTED_COUNTS = {"MEL": 4522, "NV": 12875, "BCC": 3323}
SPLIT_RATIOS = (0.70, 0.15, 0.15)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
    --navy: #092E3A;
    --deep: #063B42;
    --teal: #079B8B;
    --teal-dark: #067A6E;
    --mint: #E6F6F3;
    --bg: #F3F8FA;
    --surface: #FFFFFF;
    --border: #DCE7EC;
    --ink: #102A43;
    --muted: #647885;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
}

main .block-container {
    max-width: 1680px;
    padding-top: 2.2rem !important;
    padding-bottom: 2.5rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063B42 0%, #042A33 100%);
    min-width: 330px !important;
    max-width: 340px !important;
    border-right: 1px solid rgba(154, 226, 220, 0.14);
}

[data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    padding-bottom: 1.2rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}

[data-testid="stSidebar"] * {
    color: #F2FAFA !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C4DBDC !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F7FCFC !important;
    color: #14363E !important;
    border-radius: 9px !important;
    border: 1px solid rgba(180, 230, 226, 0.35) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] input::placeholder {
    color: #355860 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px dashed rgba(138, 219, 213, 0.65);
    border-radius: 11px;
    padding: 6px;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    padding: 4px;
}

/* Radio navigation items in sidebar */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 3px;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: transparent;
    border-radius: 10px;
    padding: 8px 12px !important;
    margin: 1px 0;
    transition: background 0.16s ease, transform 0.16s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(15, 118, 120, 0.55);
}

[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(13, 148, 136, 0.45), rgba(8, 112, 116, 0.35));
    border-left: 3px solid #48E0CF;
}

.brand-box {
    text-align: center;
    padding: 10px 6px 14px;
    border-bottom: 1px solid rgba(180, 225, 222, 0.18);
    margin-bottom: 8px;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #FFFFFF;
}

.brand-mark {
    font-size: 27px;
    color: #48E0CF;
    margin-right: 7px;
}

.brand-sub {
    font-size: 11px;
    color: #C4DADB;
    line-height: 1.45;
    margin-top: 2px;
}

.sidebar-divider {
    height: 1px;
    background: rgba(180, 225, 222, 0.2);
    margin: 10px 0;
}

.sidebar-input-card {
    background: rgba(11, 98, 102, 0.44);
    border: 1px solid rgba(138, 224, 216, 0.2);
    border-radius: 14px;
    padding: 12px;
}

.sidebar-section-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 15px;
    color: #F4FFFF;
    margin: 0;
}

.sidebar-hint {
    font-size: 10.5px;
    color: #C5DFDE;
    line-height: 1.35;
}

.sidebar-status-box {
    background: rgba(11, 95, 102, 0.65);
    border: 1px solid rgba(145, 226, 218, 0.22);
    border-radius: 13px;
    padding: 12px 13px;
    font-size: 12px;
    line-height: 1.65;
    color: #FFFFFF;
}

/* Main page headers & cards */
.page-header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(16, 42, 67, 0.035);
}

.page-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.page-icon-badge {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--mint);
    color: var(--teal-dark);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    font-weight: 700;
    flex-shrink: 0;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: var(--ink);
    margin: 0;
    line-height: 1.2;
}

.hero-sub {
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
}

.top-badge {
    background: var(--mint);
    color: var(--teal-dark);
    border: 1px solid #C5ECE5;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 700;
    white-space: nowrap;
}

.panel {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    padding: 4px 0 6px;
    box-shadow: none;
    margin-bottom: 10px;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 15.5px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 6px 0;
}

.panel-caption {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
    line-height: 1.45;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 15px 17px;
    box-shadow: 0 4px 12px rgba(18, 55, 70, 0.038);
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #5E7382;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--ink);
    margin: 4px 0 2px;
    line-height: 1.15;
}

.metric-help {
    font-size: 11.5px;
    color: var(--teal-dark);
    font-weight: 500;
}

.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #EBF1F4;
    padding: 8px 0;
    font-size: 12.5px;
}

.kv-row:last-child {
    border-bottom: none;
}

.soft-note {
    background: var(--mint);
    border: 1px solid #C5ECE5;
    border-radius: 12px;
    padding: 11px 14px;
    color: #1F595C;
    font-size: 12.5px;
    line-height: 1.5;
    margin: 10px 0;
}

.arch-flow {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin-top: 8px;
}

.arch-step {
    background: #F6FBFB;
    border: 1px solid #D9EAE8;
    border-left: 4px solid var(--teal);
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 12.5px;
}

.arch-step b {
    color: var(--ink);
}

.arch-step span {
    color: var(--muted);
    font-size: 11.5px;
    display: block;
    margin-top: 1px;
}

.class-chip-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}

.class-chip {
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 700;
}

.small-muted {
    font-size: 11.5px;
    color: var(--muted);
    margin-top: 8px;
    line-height: 1.45;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 38px;
}

div[data-testid="stFileUploader"] {
    background: #FAFCFD;
    border: 1px dashed #B5CDD6;
    border-radius: 12px;
    padding: 6px;
}

div[data-baseweb="tab-list"] {
    gap: 6px;
    margin-bottom: 10px;
}

button[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 700 !important;
    padding: 8px 14px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.app-footer {
    text-align: center;
    color: #889BA5;
    font-size: 10.5px;
    letter-spacing: 0.4px;
    padding: 28px 0 8px;
}

/* ===== Perbaikan tampilan: area utama tetap terang walau tema browser gelap ===== */
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stHeader"] * { color: #324B5A !important; }
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stCaptionContainer"] * { color: var(--ink) !important; }
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-baseweb="input"] > div {
    background: #FFFFFF !important; color: var(--ink) !important; border-color: var(--border) !important;
}
[data-testid="stMain"] input:disabled { -webkit-text-fill-color: #7D8F99 !important; opacity: 1 !important; }
[data-testid="stMain"] [data-baseweb="select"] * { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
    background: #F6FAFB !important; border: 1px dashed #B5CBD3 !important;
}
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button {
    background: #FFFFFF !important; border: 1px solid var(--border) !important;
}
button[data-baseweb="tab"] { color: #4B6270 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal-dark) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }
button[data-testid="stBaseButton-primary"] {
    background: var(--teal) !important; border-color: var(--teal) !important; color: #FFFFFF !important;
}
button[data-testid="stBaseButton-primary"]:hover { background: var(--teal-dark) !important; }
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important; color: var(--ink) !important; border: 1px solid var(--border) !important;
}
[data-testid="stMain"] [data-baseweb="radio"]:has(input:checked) > div:first-child {
    background-color: var(--teal) !important; border-color: var(--teal) !important;
}
[data-testid="stMain"] [data-baseweb="radio"]:not(:has(input:checked)) > div:first-child {
    background-color: #FFFFFF !important; border: 1.5px solid #9FB3BD !important;
}
[data-testid="stMain"] [data-testid="stAlert"] {
    background: #EAF3F5 !important; border: 1px solid #CFE2E7 !important;
    border-left: 4px solid var(--teal) !important; border-radius: 10px !important;
}
[data-testid="stMain"] [data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] { width: 100%; border-radius: 10px; padding: 9px 12px; }
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) { background: rgba(120, 224, 212, 0.18); }

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE + DATA HELPERS
# ============================================================
DEFAULTS = {
    "gt": None,
    "meta": None,
    "data_df": None,
    "image_dir": str(IMAGE_DIR),
    "trained_model": None,
    "training_history": None,
    "eval_result": None,
    "model_status": "Belum dilatih",
    "last_model_path": None,
    "split_df": None,
    "mm_model": None,
    "mm_prep": None,
    "mm_history": None,
    "mm_eval_result": None,
    "mm_status": "Belum dilatih",
    "mm_model_path": None,
    "mm_missing_rate": 0,
    "missing_rate": 0,
    "sidebar_feedback": None,
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
        aliases = {
            "MELANOMA": "MEL",
            "NEVUS": "NV",
            "MELANOCYTIC NEVUS": "NV",
            "BASAL CELL CARCINOMA": "BCC",
        }
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
    work = df.copy().reset_index(drop=True)
    work["split"] = ""
    if "lesion_id" in work.columns and work["lesion_id"].notna().any():
        groups = (
            work[work["lesion_id"].notna()]
            .groupby("lesion_id")["diagnosis"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )
        try:
            train_g, temp_g = train_test_split(
                groups, test_size=0.30, random_state=seed, stratify=groups["diagnosis"]
            )
            val_g, test_g = train_test_split(
                temp_g, test_size=0.50, random_state=seed, stratify=temp_g["diagnosis"]
            )
            train_ids, val_ids, test_ids = set(train_g.lesion_id), set(val_g.lesion_id), set(test_g.lesion_id)
            mask = work["lesion_id"].notna()
            work.loc[mask & work.lesion_id.isin(train_ids), "split"] = "Train"
            work.loc[mask & work.lesion_id.isin(val_ids), "split"] = "Validation"
            work.loc[mask & work.lesion_id.isin(test_ids), "split"] = "Test"
            no_id = work[work["lesion_id"].isna()].copy()
            if len(no_id):
                tr, tmp = train_test_split(
                    no_id.index, test_size=0.30, random_state=seed, stratify=no_id["diagnosis"]
                )
                va, te = train_test_split(
                    tmp, test_size=0.50, random_state=seed, stratify=work.loc[tmp, "diagnosis"]
                )
                work.loc[tr, "split"] = "Train"
                work.loc[va, "split"] = "Validation"
                work.loc[te, "split"] = "Test"
            return work
        except Exception:
            pass
    tr, tmp = train_test_split(work.index, test_size=0.30, random_state=seed, stratify=work["diagnosis"])
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=work.loc[tmp, "diagnosis"])
    work.loc[tr, "split"] = "Train"
    work.loc[va, "split"] = "Validation"
    work.loc[te, "split"] = "Test"
    return work


def make_metric_card(label: str, value: str, hint: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, icon: str = "◈", badge: Optional[str] = None):
    badge_html = f'<div class="top-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header-wrap">
            <div class="page-header-left">
                <div>
                    <div class="hero-title">{title}</div>
                    <div class="hero-sub">{subtitle}</div>
                </div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_distribution(df: pd.DataFrame, height: int = 260):
    counts = df["diagnosis"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    chart = pd.DataFrame({"Kelas": counts.index, "Jumlah Gambar": counts.values})
    fig = px.bar(
        chart,
        x="Kelas",
        y="Jumlah Gambar",
        color="Kelas",
        color_discrete_map=CLASS_COLORS,
        text="Jumlah Gambar",
    )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=22, b=8),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans", color="#102A43"),
        xaxis_title="",
        yaxis_title="Jumlah Gambar",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def dataset_status() -> str:
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
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(15) if train else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
                if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.frame)

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
        counts = train_df["diagnosis"].value_counts().reindex(CLASS_ORDER, fill_value=1).values
        weights = len(train_df) / (3 * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        history = []
        best_f1 = -1.0
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
            history.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_accuracy": accuracy_score(train_true, train_pred),
                "val_accuracy": accuracy_score(val_true, val_pred),
                "val_macro_f1": va_f1,
            })
            progress.progress(
                (epoch + 1) / epochs,
                text=f"Training epoch {epoch + 1}/{epochs} · val Macro-F1 {va_f1:.4f}",
            )
            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "cm": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
            "report": classification_report(
                y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_ORDER,
                output_dict=True, zero_division=0
            ),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }


def render_eval_dashboard(result: dict, model_label: str):
    a, b, c, d = st.columns(4)
    with a:
        make_metric_card("ACCURACY", f"{result['accuracy']:.4f}", f"Test set · {model_label}")
    with b:
        make_metric_card("MACRO-F1", f"{result['macro_f1']:.4f}", "Metrik utama")
    with c:
        make_metric_card("PRECISION", f"{result['precision']:.4f}", "Macro average")
    with d:
        make_metric_card("RECALL", f"{result['recall']:.4f}", "Macro average")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    x, y = st.columns([1.05, 1.15], gap="medium")
    with x:
        st.markdown('<div class="panel"><div class="panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = px.imshow(
            result["cm"],
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            text_auto=True,
            color_continuous_scale="Teal",
            labels=dict(x="Predicted", y="Actual", color="Jumlah"),
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=24, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with y:
        st.markdown('<div class="panel"><div class="panel-title">Classification Report</div>', unsafe_allow_html=True)
        rep = pd.DataFrame(result["report"]).T
        st.dataframe(rep.round(4), use_container_width=True)
        st.download_button(
            f"Download classification report ({model_label})",
            rep.to_csv().encode(),
            f"classification_report_{model_label.lower().replace(' ', '_')}.csv",
            "text/csv",
        )
        try:
            y_bin = label_binarize(result["y_true"], classes=[0, 1, 2])
            auc = roc_auc_score(y_bin, result["y_prob"], average="macro", multi_class="ovr")
            st.metric("Macro ROC-AUC (OvR)", f"{auc:.4f}")
        except Exception:
            st.caption("ROC-AUC tidak dapat dihitung pada kondisi test set ini.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MANUAL DATA STORAGE
# ============================================================
def init_manual_db():
    with sqlite3.connect(MANUAL_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_records (
                record_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                site TEXT,
                diagnosis TEXT NOT NULL
            )
        """)
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
            (
                record_id,
                datetime.now().isoformat(timespec="seconds"),
                relative_path,
                age if age != "" else None,
                sex,
                site,
                diagnosis,
            ),
        )
        conn.commit()
    return record_id


def get_manual_record_count() -> int:
    init_manual_db()
    with sqlite3.connect(MANUAL_DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM manual_records").fetchone()[0]


init_manual_db()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand"><span class="brand-mark">✚</span>DermaSense</div>
            <div class="brand-sub">Skin Lesion Classification · ISIC 2019<br>MEL · NV · BCC Research Prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_labels = {
        ":material/dashboard:  Dashboard": "Dashboard",
        ":material/database:  Dataset": "Dataset",
        ":material/model_training:  Model CNN": "Model CNN",
        ":material/account_tree:  Model Multimodal": "Model Multimodal",
        ":material/monitoring:  Evaluation": "Evaluation",
        ":material/info:  About": "About",
    }
    selected_nav = st.radio(
        "NAVIGATION",
        list(nav_labels.keys()),
        format_func=lambda x: x,
        label_visibility="collapsed",
        key="main_navigation",
    )
    page = nav_labels[selected_nav]

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Status Dataset & Model**")
    st.markdown(
        f"""
        <div class="sidebar-status-box">
            <div>● <b>{dataset_status()}</b></div>
            <div>CNN: <code>{st.session_state.model_status}</code></div>
            <div>Multimodal: <code>{st.session_state.mm_status}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Research prototype · Bukan alat diagnosis klinis")

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    page_header(
        "Dashboard",
        "Ringkasan dataset ISIC 2019 dan status pengembangan model.",
        badge=dataset_status(),
    )
    df = st.session_state.data_df
    if df is None:
        a, b, c, d = st.columns(4, gap="medium")
        with a:
            make_metric_card("TOTAL GAMBAR", "20,720*", "Referensi 3 kelas ISIC 2019")
        with b:
            make_metric_card("TRAINING", "70%", "Rasio split terencana")
        with c:
            make_metric_card("VALIDATION", "15%", "Rasio split terencana")
        with d:
            make_metric_card("TESTING", "15%", "Rasio split terencana")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.info("Muat CSV Ground Truth ISIC 2019 pada menu **Dataset** untuk menampilkan statistik aktual.")
    else:
        a, b, c, d = st.columns(4, gap="medium")
        with a:
            make_metric_card("TOTAL GAMBAR", f"{len(df):,}", "Data yang berhasil dimuat")
        with b:
            make_metric_card("KELAS MEL", f"{(df.diagnosis == 'MEL').sum():,}", "Melanoma")
        with c:
            make_metric_card("KELAS NV", f"{(df.diagnosis == 'NV').sum():,}", "Melanocytic Nevus")
        with d:
            make_metric_card("KELAS BCC", f"{(df.diagnosis == 'BCC').sum():,}", "Basal Cell Carcinoma")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        x, y = st.columns([1.25, 0.95], gap="medium")
        with x:
            st.markdown('<div class="panel"><div class="panel-title">Distribusi Kelas</div>', unsafe_allow_html=True)
            display_distribution(df, height=275)
            st.markdown("</div>", unsafe_allow_html=True)
        with y:
            st.markdown('<div class="panel"><div class="panel-title">Status Pengembangan Model</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kv-row"><span>CNN (EfficientNet-B0)</span><b>{st.session_state.model_status}</b></div>
                <div class="kv-row"><span>Multimodal (Hybrid CNN-BiLSTM)</span><b>{st.session_state.mm_status}</b></div>
                <div class="kv-row"><span>Split Dataset (70/15/15)</span><b>{"Tersedia" if st.session_state.split_df is not None else "Belum dibuat"}</b></div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Gunakan menu Model CNN, Model Multimodal, dan Evaluation untuk melatih serta menguji model.")
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DATASET EXPLORER
# ============================================================
elif page == "Dataset":
    page_header(
        "Dataset Explorer",
        "Eksplorasi dataset ISIC 2019 untuk tiga kelas (MEL, NV, BCC) beserta metadata klinis.",
        badge=dataset_status(),
    )
    t_overview, t_viewer, t_metadata, t_pre = st.tabs(
        [":material/bar_chart: Overview", ":material/table_rows: Data Viewer", ":material/fact_check: Metadata", ":material/tune: Preprocessing"]
    )

    with t_overview:
        left, mid, chart, samples = st.columns([1.08, 1.05, 1.18, 1.32], gap="medium")
        with left:
            st.markdown('<div class="panel"><div class="panel-title">Pilih & Muat Dataset</div>', unsafe_allow_html=True)
            source = st.radio(
                "Sumber dataset",
                ["ISIC 2019 (3 Kelas)", "Upload Dataset (CSV)"],
                label_visibility="collapsed",
            )
            if source == "ISIC 2019 (3 Kelas)":
                st.caption("Gunakan file `ISIC_2019_Training_GroundTruth.csv` dan metadata resmi ISIC 2019.")
            else:
                st.caption("Unggah CSV kustom dengan kolom `image` dan label `MEL/NV/BCC` atau kolom one-hot.")

            gt_upload = st.file_uploader("Upload Ground Truth CSV", type="csv", key="gt_upload")
            meta_upload = st.file_uploader("Upload Metadata CSV (opsional)", type="csv", key="meta_upload")
            image_dir_input = st.text_input(
                "Folder gambar",
                value=st.session_state.image_dir,
                help="Path folder yang berisi file JPG/PNG citra lesi.",
            )
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
                        st.session_state.eval_result = None
                        st.session_state.mm_eval_result = None
                        st.success(f"Dataset dimuat: {len(st.session_state.data_df):,} gambar.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal memuat dataset: {e}")

            st.markdown('<div class="small-muted">Filter otomatis untuk 3 kelas: MEL, NV, dan BCC.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        df = st.session_state.data_df
        with mid:
            st.markdown('<div class="panel"><div class="panel-title">Ringkasan Dataset</div>', unsafe_allow_html=True)
            if df is not None:
                total = len(df)
                train_n, val_n = round(total * 0.70), round(total * 0.15)
                test_n = total - train_n - val_n
                rows = [
                    ("Total Gambar", f"{total:,}"),
                    ("Training Set (70%)", f"{train_n:,}"),
                    ("Validation Set (15%)", f"{val_n:,}"),
                    ("Testing Set (15%)", f"{test_n:,}"),
                    ("Jumlah Kelas", "3 (MEL, NV, BCC)"),
                ]
            else:
                rows = [
                    ("Total Gambar", "20,720*"),
                    ("Training Set (70%)", "14,504*"),
                    ("Validation Set (15%)", "3,108*"),
                    ("Testing Set (15%)", "3,108*"),
                    ("Jumlah Kelas", "3 (MEL, NV, BCC)"),
                ]
            for label, value in rows:
                st.markdown(f'<div class="kv-row"><span>{label}</span><b>{value}</b></div>', unsafe_allow_html=True)
            if df is None:
                st.caption("*Angka referensi sebelum filter tambahan; muat CSV untuk statistik aktual.")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart:
            st.markdown('<div class="panel"><div class="panel-title">Distribusi Kelas</div>', unsafe_allow_html=True)
            if df is not None:
                display_distribution(df, height=250)
            else:
                counts = pd.DataFrame({
                    "Kelas": CLASS_ORDER,
                    "Jumlah Gambar": [EXPECTED_COUNTS[c] for c in CLASS_ORDER],
                })
                fig = px.bar(
                    counts,
                    x="Kelas",
                    y="Jumlah Gambar",
                    color="Kelas",
                    color_discrete_map=CLASS_COLORS,
                    text="Jumlah Gambar",
                )
                fig.update_layout(
                    height=250,
                    margin=dict(l=6, r=6, t=20, b=6),
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis_title="",
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Distribusi referensi tiga kelas ISIC 2019.")
            st.markdown("</div>", unsafe_allow_html=True)

        with samples:
            st.markdown('<div class="panel"><div class="panel-title">Contoh Citra per Kelas</div>', unsafe_allow_html=True)
            if df is not None:
                cols = st.columns(3, gap="small")
                for i, cls in enumerate(CLASS_ORDER):
                    sample = df[df.diagnosis == cls].head(1)
                    with cols[i]:
                        if len(sample):
                            im = load_image(sample.iloc[0].image, st.session_state.image_dir)
                            if im:
                                st.image(im, use_container_width=True)
                            else:
                                st.markdown(
                                    '<div style="background:#EDF4F5;height:86px;border-radius:10px;'
                                    'display:flex;align-items:center;justify-content:center;font-size:11px;color:#647885">No image</div>',
                                    unsafe_allow_html=True,
                                )
                        st.caption(f"**{cls}**")
            else:
                st.info("Muat dataset dan folder gambar untuk menampilkan sampel citra per kelas.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        cnn_col, meta_col = st.columns([1.12, 1.0], gap="medium")
        with cnn_col:
            st.markdown(
                '<div class="panel"><div class="panel-title">Model CNN – EfficientNet-B0</div>'
                '<div class="panel-caption">Klasifikasi citra lesi kulit menggunakan EfficientNet-B0 sebagai backbone.</div>',
                unsafe_allow_html=True,
            )
            ta, tb = st.tabs(["Arsitektur Model", "Konfigurasi & Training"])
            with ta:
                st.markdown(
                    """
                    <div class="arch-flow">
                        <div class="arch-step"><b>1. Input Citra</b><span>Resolusi 224 × 224 × 3 (RGB)</span></div>
                        <div class="arch-step"><b>2. EfficientNet-B0 Backbone</b><span>Ekstraksi fitur spasial & tekstur lesi</span></div>
                        <div class="arch-step"><b>3. Global Average Pooling + Classifier</b><span>Linear head menuju 3 kelas output</span></div>
                    </div>
                    <div class="class-chip-row">
                        <span class="class-chip" style="background:#FFE4DE;color:#9B351E">MEL · Melanoma</span>
                        <span class="class-chip" style="background:#DCF5F1;color:#086C62">NV · Nevus</span>
                        <span class="class-chip" style="background:#E0F2FE;color:#075985">BCC · Carcinoma</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with tb:
                st.markdown(
                    """
                    <div class="kv-row"><span>Input Size</span><b>224 × 224 RGB</b></div>
                    <div class="kv-row"><span>Optimizer</span><b>AdamW (weight_decay=1e-4)</b></div>
                    <div class="kv-row"><span>Loss Function</span><b>Weighted Cross Entropy</b></div>
                    <div class="kv-row"><span>LR Scheduler</span><b>ReduceLROnPlateau</b></div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Atur parameter training lengkap pada menu **Model CNN**.")
            st.markdown("</div>", unsafe_allow_html=True)

        with meta_col:
            st.markdown(
                '<div class="panel"><div class="panel-title">Simulasi Missing Metadata</div>'
                '<div class="panel-caption">Atur persentase metadata yang dihilangkan untuk menguji ketahanan model multimodal.</div>',
                unsafe_allow_html=True,
            )
            missing_rate = st.radio(
                "Tingkat missing metadata",
                [0, 25, 50, 75, 100],
                format_func=lambda x: "0% (Complete)" if x == 0 else ("100% (Image Only)" if x == 100 else f"{x}% missing"),
                horizontal=False,
                key="overview_missing",
            )
            st.session_state["missing_rate"] = missing_rate
            st.markdown(
                f'<div class="soft-note">Tingkat missing metadata aktif: <b>{missing_rate}%</b>. '
                'Nilai ini membantu menguji performa saat usia, jenis kelamin, atau lokasi anatomi tidak lengkap.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if df is not None:
            st.markdown('<div class="panel"><div class="panel-title">Metadata Overview</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3, gap="medium")
            with m1:
                if df.age.notna().any():
                    age_df = pd.to_numeric(df.age, errors="coerce").dropna()
                    fig = px.histogram(age_df, nbins=18, labels={"value": "Usia", "count": "Jumlah"})
                    fig.update_layout(height=210, margin=dict(l=6, r=6, t=10, b=6), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Metadata usia belum tersedia.")
            with m2:
                if df.sex.notna().any():
                    fig = px.pie(df.dropna(subset=["sex"]), names="sex", hole=0.55)
                    fig.update_layout(height=210, margin=dict(l=6, r=6, t=10, b=6))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Metadata sex belum tersedia.")
            with m3:
                if df.site.notna().any():
                    top = df.site.value_counts().head(5).rename_axis("Anatomical site").reset_index(name="Jumlah")
                    fig = px.bar(top, x="Jumlah", y="Anatomical site", orientation="h")
                    fig.update_layout(height=210, margin=dict(l=6, r=6, t=10, b=6), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Metadata lokasi belum tersedia.")
            st.markdown("</div>", unsafe_allow_html=True)

    with t_viewer:
        df = st.session_state.data_df
        if df is None:
            st.info("Muat dataset terlebih dahulu pada tab **Overview**.")
            st.markdown('<div class="panel"><div class="panel-title">Tampilan setelah dataset dimuat</div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "image": ["ISIC_xxxxxxx", "ISIC_xxxxxxx"],
                "diagnosis": ["MEL", "NV"],
                "age": [55, 40],
                "sex": ["female", "male"],
                "site": ["anterior torso", "upper extremity"],
                "lesion_id": ["BCN_xxxxx", "HAM_xxxxx"],
            }), use_container_width=True, hide_index=True)
            st.caption("Contoh format baris (bukan data asli). Tab ini menyediakan filter kelas, pencarian Image ID, dan pratinjau citra.")
        else:
            f1, f2 = st.columns(2, gap="medium")
            with f1:
                cls = st.selectbox("Filter kelas", ["Semua"] + CLASS_ORDER, key="viewer_cls")
            with f2:
                query = st.text_input("Cari Image ID", placeholder="Contoh: ISIC_0024306", key="viewer_search")

            view = df.copy()
            if cls != "Semua":
                view = view[view.diagnosis == cls]
            if query:
                view = view[view.image.str.contains(query, case=False, na=False)]

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.caption(f"{len(view):,} record ditemukan (menampilkan maksimal 1.000 baris).")
            st.dataframe(view.head(1000), use_container_width=True, hide_index=True)

            if len(view):
                selected = st.selectbox("Pilih gambar untuk preview", view.image.tolist()[:2000])
                im = load_image(selected, st.session_state.image_dir)
                if im:
                    st.image(
                        im,
                        caption=f"{selected} · {view.loc[view.image == selected, 'diagnosis'].iloc[0]}",
                        width=340,
                    )
                else:
                    st.warning("File gambar tidak ditemukan. Periksa path folder gambar.")
            st.markdown("</div>", unsafe_allow_html=True)


    with t_metadata:
        df = st.session_state.data_df
        if df is None:
            st.info("Muat dataset terlebih dahulu pada tab **Overview**.")
            st.markdown('<div class="panel"><div class="panel-title">Metadata klinis yang dipakai</div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                ("age_approx", "age", "Numerik (tahun)", "Nilai kosong diisi median data train, lalu z-score, plus penanda hilang"),
                ("sex", "sex", "Kategorikal (Female / Male)", "One-hot plus penanda hilang"),
                ("anatom_site_general", "site", "Kategorikal (9 lokasi)", "One-hot plus penanda hilang"),
                ("lesion_id", "lesion_id", "ID lesi", "Dipakai untuk mengelompokkan split, bukan fitur model"),
            ], columns=["Kolom CSV", "Nama di aplikasi", "Tipe", "Perlakuan"]), use_container_width=True, hide_index=True)
            st.caption("Vektor metadata model multimodal berisi 15 nilai: 1 usia, 2 jenis kelamin, 9 lokasi, dan 3 penanda hilang.")
        else:
            st.markdown('<div class="panel"><div class="panel-title">Kelengkapan Metadata Klinis</div>', unsafe_allow_html=True)
            fields = [c for c in ["age", "sex", "site"] if c in df.columns]
            comp = pd.DataFrame({
                "Metadata": fields,
                "Tersedia": [df[c].notna().sum() for c in fields],
                "Hilang": [df[c].isna().sum() for c in fields],
            })
            if len(comp):
                comp["Missing (%)"] = (comp["Hilang"] / len(df) * 100).round(2)
                st.dataframe(comp, use_container_width=True, hide_index=True)
                fig = px.bar(comp, x="Metadata", y="Missing (%)", color_discrete_sequence=["#079B8B"])
                fig.update_layout(height=240, margin=dict(l=8, r=8, t=16, b=8))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="panel"><div class="panel-title">Preview Metadata</div>', unsafe_allow_html=True)
            st.dataframe(df[["image", "diagnosis"] + fields].head(100), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with t_pre:
        df = st.session_state.data_df
        st.markdown('<div class="panel"><div class="panel-title">Pengaturan Preprocessing & Dataset Split</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3, gap="medium")
        with p1:
            size = st.selectbox("Image size", ["224 × 224", "256 × 256", "300 × 300"])
        with p2:
            norm = st.selectbox("Normalisasi", ["ImageNet mean/std", "Rescale 0–1"])
        with p3:
            aug = st.multiselect(
                "Augmentasi training",
                ["Horizontal flip", "Vertical flip", "Rotation ±15°", "Brightness/contrast"],
                default=["Horizontal flip", "Rotation ±15°"],
            )
        st.markdown(
            '<div class="soft-note">Augmentasi hanya diterapkan pada data training. '
            "Validation dan testing menggunakan preprocessing deterministik serta pengelompokan <code>lesion_id</code> bila tersedia.</div>",
            unsafe_allow_html=True,
        )
        if df is not None:
            if st.button("Buat split dataset 70/15/15", type="primary"):
                if not SKLEARN_AVAILABLE:
                    st.error("scikit-learn belum terpasang.")
                else:
                    try:
                        split = make_splits(df)
                        st.session_state.split_df = split
                        st.success("Split 70/15/15 berhasil dibuat.")
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.split_df is not None:
                split = st.session_state.split_df
                st.dataframe(
                    split["split"].value_counts().rename_axis("Split").reset_index(name="Jumlah"),
                    hide_index=True,
                )
                st.download_button(
                    "Download CSV hasil split",
                    split.to_csv(index=False).encode(),
                    "isic2019_3class_split.csv",
                    "text/csv",
                )
        else:
            st.info("Muat dataset terlebih dahulu untuk membuat split 70/15/15.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODEL CNN
# ============================================================
elif page == "Model CNN":
    page_header(
        "Model CNN – EfficientNet-B0",
        "Training model klasifikasi citra lesi kulit untuk MEL, NV, dan BCC.",
        badge=st.session_state.model_status,
    )
    if not TORCH_AVAILABLE:
        st.error("PyTorch/torchvision belum tersedia. Install dependency dari requirements.txt.")
    df = st.session_state.data_df
    if df is None:
        st.warning("Muat dataset ISIC 2019 terlebih dahulu melalui menu **Dataset**.")
        st.markdown('<div class="panel"><div class="panel-title">Ringkasan Model</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Backbone", "EfficientNet-B0, pretrained ImageNet"),
            ("Input", "224 × 224 RGB, normalisasi mean/std ImageNet"),
            ("Augmentasi (train)", "Flip horizontal/vertikal, rotasi 15°, color jitter"),
            ("Loss", "Cross-entropy berbobot kelas (dihitung dari data train)"),
            ("Optimizer", "AdamW (weight decay 1e-4), ReduceLROnPlateau"),
            ("Default", "10 epoch, batch 16, learning rate 1e-4"),
            ("Seleksi model", "Macro-F1 validasi terbaik"),
        ], columns=["Komponen", "Keterangan"]), use_container_width=True, hide_index=True)
    else:
        c1, c2 = st.columns([1.1, 1.0], gap="medium")
        with c1:
            st.markdown('<div class="panel"><div class="panel-title">Konfigurasi Training</div>', unsafe_allow_html=True)
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)
            batch = st.selectbox("Batch Size", [8, 16, 32], index=1)
            lr = st.select_slider(
                "Learning Rate",
                options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                value=0.0001,
                format_func=lambda x: f"{x:g}",
            )
            pretrained = st.checkbox("Gunakan pretrained ImageNet weights", value=True)
            seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel"><div class="panel-title">Arsitektur CNN</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="arch-flow">
                    <div class="arch-step"><b>Input Image</b><span>224 × 224 × 3 RGB</span></div>
                    <div class="arch-step"><b>EfficientNet-B0</b><span>Pretrained feature extractor backbone</span></div>
                    <div class="arch-step"><b>Global Average Pooling & Linear (3)</b><span>Klasifikasi probabilitas MEL · NV · BCC</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.split_df is None:
            if st.button("Buat split otomatis 70/15/15"):
                st.session_state.split_df = make_splits(df, seed=int(seed))
                st.rerun()

        split_df = st.session_state.split_df
        if split_df is not None:
            st.markdown('<div class="panel"><div class="panel-title">Ringkasan Split & Eksekusi Training</div>', unsafe_allow_html=True)
            st.dataframe(split_df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0), use_container_width=True)
            image_dir = st.session_state.image_dir
            missing_files = [x for x in split_df.image.head(1000) if get_image_path(x, image_dir) is None]
            if missing_files:
                st.warning(f"Tidak menemukan sebagian file gambar di folder: `{image_dir}`. Contoh ID: {missing_files[:3]}")

            if st.button(":material/play_arrow: Train Model", type="primary", use_container_width=True, disabled=not TORCH_AVAILABLE):
                if not SKLEARN_AVAILABLE:
                    st.error("scikit-learn belum terpasang.")
                else:
                    try:
                        random.seed(int(seed))
                        np.random.seed(int(seed))
                        torch.manual_seed(int(seed))
                        progress = st.progress(0, text="Menyiapkan training...")
                        model, history, device = train_cnn(
                            split_df, image_dir, int(epochs), int(batch), float(lr), pretrained, progress
                        )
                        st.session_state.trained_model = model
                        st.session_state.training_history = history
                        st.session_state.model_status = f"Training selesai ({device})"
                        model_path = MODEL_DIR / "dermasense_efficientnet_b0.pth"
                        torch.save({"state_dict": model.state_dict(), "classes": CLASS_ORDER}, model_path)
                        st.session_state.last_model_path = str(model_path)
                        st.success(f"Training selesai pada {device}. Best validation Macro-F1: {history.val_macro_f1.max():.4f}")
                    except Exception as e:
                        st.error(f"Training gagal: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.training_history is not None:
            hist = st.session_state.training_history
            st.markdown('<div class="panel"><div class="panel-title">Training History</div>', unsafe_allow_html=True)
            h1, h2 = st.columns(2, gap="medium")
            with h1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.epoch, y=hist.train_loss, name="Train loss", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_loss, name="Validation loss", mode="lines+markers"))
                fig.update_layout(title="Loss per epoch", height=300, margin=dict(l=8, r=8, t=36, b=8))
                st.plotly_chart(fig, use_container_width=True)
            with h2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.epoch, y=hist.train_accuracy, name="Train accuracy", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_accuracy, name="Validation accuracy", mode="lines+markers"))
                fig.update_layout(title="Accuracy per epoch", height=300, yaxis_range=[0, 1], margin=dict(l=8, r=8, t=36, b=8))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODEL MULTIMODAL
# ============================================================
elif page == "Model Multimodal":
    page_header(
        "Model Multimodal (Hybrid CNN-BiLSTM)",
        "Training dan simulasi missing metadata untuk klasifikasi MEL, NV, dan BCC.",
        badge=st.session_state.mm_status,
    )
    if not (TORCH_AVAILABLE and MULTIMODAL_LSTM_AVAILABLE):
        st.error("PyTorch atau modul `multimodal_lstm` belum tersedia.")
    df = st.session_state.data_df
    if df is None:
        st.warning("Muat dataset ISIC 2019 terlebih dahulu melalui menu **Dataset**.")
        st.markdown('<div class="panel"><div class="panel-title">Ringkasan Model Multimodal</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Branch citra", "EfficientNet-B0, peta fitur 7×7 dibaca sebagai 49 langkah urutan, lalu BiLSTM"),
            ("Branch metadata", "Usia, jenis kelamin, lokasi anatomi, dan penanda hilang, diproses MLP"),
            ("Fusi", "Concatenation fitur citra dan metadata, Dense, Softmax 3 kelas"),
            ("Simulasi missing", "Metadata disembunyikan pada tingkat 0, 25, 50, 75, atau 100%"),
            ("Pelatihan", "Loss, optimizer, dan seleksi model sama dengan model CNN"),
        ], columns=["Komponen", "Keterangan"]), use_container_width=True, hide_index=True)
    else:
        c1, c2 = st.columns([1.1, 1.0], gap="medium")
        with c1:
            st.markdown('<div class="panel"><div class="panel-title">Konfigurasi Hybrid CNN-BiLSTM + Metadata</div>', unsafe_allow_html=True)
            mm_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1, key="mm_epochs")
            mm_batch = st.selectbox("Batch Size", [8, 16, 32], index=1, key="mm_batch")
            mm_lr = st.select_slider(
                "Learning Rate",
                options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                value=0.0001,
                format_func=lambda x: f"{x:g}",
                key="mm_lr",
            )
            mm_pretrained = st.checkbox("Gunakan pretrained ImageNet weights", value=True, key="mm_pretrained")
            mm_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, key="mm_seed")
            mm_missing = st.radio(
                "Simulasi missing metadata",
                [0, 25, 50, 75, 100],
                format_func=lambda x: "Complete (0%)" if x == 0 else ("Image only (100%)" if x == 100 else f"{x}% missing"),
                horizontal=True,
                key="mm_missing_radio",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel"><div class="panel-title">Arsitektur Hybrid Multimodal</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="arch-flow">
                    <div class="arch-step"><b>Branch Citra</b><span>224×224×3 → EfficientNet-B0 → Urutan Fitur (7×7) → BiLSTM</span></div>
                    <div class="arch-step"><b>Branch Metadata Klinis</b><span>Age, Sex, Site + Missing Mask → MLP</span></div>
                    <div class="arch-step"><b>Fusion & Output</b><span>Concatenation Fusion → Dense → Softmax (MEL · NV · BCC)</span></div>
                </div>
                <div class="small-muted">Status model: <b>{st.session_state.mm_status}</b> · Missing rate terpilih: <b>{mm_missing}%</b></div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.split_df is None:
            if st.button("Buat split otomatis 70/15/15", key="mm_make_split"):
                st.session_state.split_df = make_splits(df, seed=int(mm_seed))
                st.rerun()

        split_df = st.session_state.split_df
        if split_df is not None:
            if st.button(":material/play_arrow: Train Hybrid CNN-BiLSTM Multimodal", type="primary", use_container_width=True):
                if not SKLEARN_AVAILABLE:
                    st.error("scikit-learn belum terpasang.")
                elif not MULTIMODAL_LSTM_AVAILABLE:
                    st.error("Modul multimodal_lstm tidak ditemukan.")
                else:
                    try:
                        random.seed(int(mm_seed))
                        np.random.seed(int(mm_seed))
                        torch.manual_seed(int(mm_seed))
                        prog_bar = st.progress(0, text="Menyiapkan training multimodal...")

                        def on_epoch(ep, total_ep, val_f1):
                            prog_bar.progress(ep / total_ep, text=f"Epoch {ep}/{total_ep} · val Macro-F1 {val_f1:.4f}")

                        mm_model, mm_prep, mm_hist, dev = mm_lstm.train_multimodal_hybrid(
                            frame=split_df,
                            image_dir=st.session_state.image_dir,
                            epochs=int(mm_epochs),
                            batch_size=int(mm_batch),
                            lr=float(mm_lr),
                            pretrained=mm_pretrained,
                            missing_rate=int(mm_missing),
                            progress_callback=on_epoch,
                        )
                        st.session_state.mm_model = mm_model
                        st.session_state.mm_prep = mm_prep
                        st.session_state.mm_history = mm_hist
                        st.session_state.mm_missing_rate = int(mm_missing)
                        st.session_state.mm_status = f"Training selesai ({dev})"

                        save_path = MODEL_DIR / "dermasense_hybrid_cnn_bilstm.pth"
                        torch.save(
                            {
                                "state_dict": mm_model.state_dict(),
                                "preprocessor": mm_prep,
                                "classes": CLASS_ORDER,
                            },
                            save_path,
                        )
                        st.session_state.mm_model_path = str(save_path)
                        st.success(
                            f"Training Hybrid CNN-BiLSTM selesai pada {dev}. "
                            f"Best val Macro-F1: {mm_hist['val_macro_f1'].max():.4f}"
                        )
                    except Exception as exc:
                        st.error(f"Training multimodal gagal: {exc}")

        if st.session_state.mm_history is not None:
            hist = st.session_state.mm_history
            st.markdown('<div class="panel"><div class="panel-title">Training History (Hybrid CNN-BiLSTM)</div>', unsafe_allow_html=True)
            h1, h2 = st.columns(2, gap="medium")
            with h1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="Train loss", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Validation loss", mode="lines+markers"))
                fig.update_layout(title="Loss per epoch", height=300, margin=dict(l=8, r=8, t=36, b=8))
                st.plotly_chart(fig, use_container_width=True)
            with h2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_accuracy"], name="Train accuracy", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_accuracy"], name="Validation accuracy", mode="lines+markers"))
                fig.update_layout(title="Accuracy per epoch", height=300, yaxis_range=[0, 1], margin=dict(l=8, r=8, t=36, b=8))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# EVALUATION
# ============================================================
elif page == "Evaluation":
    page_header(
        "Evaluasi Model",
        "Evaluasi model pada data testing yang tidak digunakan selama training.",
    )
    if st.session_state.split_df is None:
        st.warning("Split dataset belum tersedia. Buat split pada menu **Dataset** atau **Model CNN**.")
        st.markdown('<div class="panel"><div class="panel-title">Metrik yang akan ditampilkan</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Accuracy", "Proporsi prediksi benar"),
            ("Macro-F1", "Rata-rata F1 tiap kelas, metrik utama karena kelas tidak seimbang"),
            ("Precision / Recall", "Rata-rata makro per kelas"),
            ("Confusion matrix", "Sebaran prediksi terhadap label asli"),
            ("ROC-AUC (OvR)", "Kemampuan membedakan tiap kelas dari sisanya"),
            ("Uji missing metadata", "Model multimodal diuji pada 0, 25, 50, 75, dan 100% metadata hilang"),
        ], columns=["Metrik", "Keterangan"]), use_container_width=True, hide_index=True)
    else:
        eval_tabs = st.tabs([":material/image: Model CNN (EfficientNet-B0)", ":material/account_tree: Model Multimodal (Hybrid CNN-BiLSTM)"])

        with eval_tabs[0]:
            if st.session_state.trained_model is None:
                st.info("Belum ada model CNN terlatih. Selesaikan training pada menu **Model CNN** terlebih dahulu.")
            else:
                if st.button("Evaluasi CNN pada Test Set", type="primary", key="btn_eval_cnn"):
                    try:
                        with st.spinner("Menghitung metrik CNN pada test set..."):
                            result = evaluate_cnn(
                                st.session_state.trained_model,
                                st.session_state.split_df,
                                st.session_state.image_dir,
                            )
                            st.session_state.eval_result = result
                        st.success("Evaluasi CNN selesai.")
                    except Exception as e:
                        st.error(f"Evaluasi gagal: {e}")

                if st.session_state.eval_result:
                    render_eval_dashboard(st.session_state.eval_result, "EfficientNet-B0")

        with eval_tabs[1]:
            if st.session_state.mm_model is None:
                st.info("Belum ada model Multimodal terlatih. Selesaikan training pada menu **Model Multimodal** terlebih dahulu.")
            else:
                if st.button("Evaluasi Multimodal pada Test Set", type="primary", key="btn_eval_mm"):
                    eval_fn = getattr(mm_lstm, "evaluate_multimodal_hybrid", None) if mm_lstm else None
                    if callable(eval_fn):
                        try:
                            with st.spinner("Menghitung metrik Multimodal pada test set..."):
                                mm_res = eval_fn(
                                    st.session_state.mm_model,
                                    st.session_state.mm_prep,
                                    st.session_state.split_df,
                                    st.session_state.image_dir,
                                    missing_rate=st.session_state.mm_missing_rate,
                                )
                                st.session_state.mm_eval_result = mm_res
                            st.success("Evaluasi Multimodal selesai.")
                        except Exception as e:
                            st.error(f"Evaluasi multimodal gagal: {e}")
                    else:
                        st.warning(
                            "Fungsi `evaluate_multimodal_hybrid` belum tersedia di `multimodal_lstm.py`. "
                            "Tambahkan fungsi evaluasi tersebut di `multimodal_lstm.py` untuk mengaktifkan evaluasi otomatis di tab ini."
                        )

                if st.session_state.mm_eval_result:
                    render_eval_dashboard(st.session_state.mm_eval_result, "Hybrid CNN-BiLSTM")

                st.markdown('<div class="panel"><div class="panel-title">Uji Ketahanan terhadap Missing Metadata</div></div>', unsafe_allow_html=True)
                st.caption("Model terlatih yang sama diuji pada missing rate 0, 25, 50, 75, dan 100%.")
                rob_fn = getattr(mm_lstm, "evaluate_multimodal_hybrid", None) if mm_lstm else None
                if st.button("Jalankan uji semua missing rate", key="btn_eval_mm_rob", disabled=not callable(rob_fn)):
                    rob_rows = []
                    try:
                        with st.spinner("Menguji model pada 5 skenario missing metadata..."):
                            for rate in [0, 25, 50, 75, 100]:
                                res = rob_fn(
                                    st.session_state.mm_model,
                                    st.session_state.mm_prep,
                                    st.session_state.split_df,
                                    st.session_state.image_dir,
                                    missing_rate=rate,
                                )
                                rob_rows.append({
                                    "Missing rate (%)": rate,
                                    "Accuracy": res["accuracy"],
                                    "Macro-F1": res["macro_f1"],
                                    "Precision": res["precision"],
                                    "Recall": res["recall"],
                                })
                        st.session_state["mm_rob_df"] = pd.DataFrame(rob_rows)
                        st.session_state["mm_rob_for"] = id(st.session_state.mm_model)
                    except Exception as e:
                        st.error(f"Uji ketahanan gagal: {e}")
                rob_df = st.session_state.get("mm_rob_df")
                if rob_df is not None and st.session_state.get("mm_rob_for") == id(st.session_state.mm_model):
                    fig = px.line(rob_df, x="Missing rate (%)", y=["Accuracy", "Macro-F1"], markers=True)
                    fig.update_layout(height=320, margin=dict(l=8, r=8, t=16, b=8), yaxis_range=[0, 1],
                                      yaxis_title="Skor", legend_title_text="")
                    fig.update_xaxes(tickvals=[0, 25, 50, 75, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(rob_df.round(4), use_container_width=True, hide_index=True)

# ============================================================
# ABOUT
# ============================================================
else:
    page_header(
        "About & Methods",
        "Informasi penelitian dan batasan sistem DermaSense.",
    )
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">DermaSense</div>
            <p style="font-size:13.5px;color:#233B4A;line-height:1.6">
                Prototype penelitian klasifikasi lesi kulit menggunakan subset tiga kelas ISIC 2019:
            </p>
            <ul style="font-size:13.5px;color:#233B4A;line-height:1.7">
                <li><b>MEL</b> — Melanoma</li>
                <li><b>NV</b> — Melanocytic Nevus</li>
                <li><b>BCC</b> — Basal Cell Carcinoma</li>
            </ul>
            <p style="font-size:13.5px;color:#233B4A;line-height:1.6;margin-bottom:0">
                Model CNN menggunakan <b>EfficientNet-B0</b>, sedangkan arah pengembangan multimodal menggabungkan
                citra dermoskopi dengan metadata klinis melalui <b>Hybrid CNN-BiLSTM</b> serta menguji ketahanan terhadap <i>missing metadata</i>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "DermaSense merupakan prototipe penelitian akademik, bukan alat diagnosis medis. "
        "Hasil model tidak boleh digunakan sebagai pengganti pemeriksaan dokter atau tenaga kesehatan."
    )

st.markdown(
    '<div class="app-footer">DERMASENSE · ISIC 2019 · MEL / NV / BCC · RESEARCH PROTOTYPE · NOT FOR CLINICAL DIAGNOSIS</div>',
    unsafe_allow_html=True,
)