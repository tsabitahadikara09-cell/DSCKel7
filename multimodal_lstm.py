from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASS_ORDER = ["MEL", "NV", "BCC"]
SITE_OPTIONS = [
    "Head/Neck", "Anterior torso", "Posterior torso",
    "Lateral torso", "Upper extremity", "Lower extremity",
    "Palms/Soles", "Oral/Genital", "Other"
]


class MetadataPreprocessor:
    """Fit hanya pada data Train agar tidak terjadi data leakage."""

    def __init__(self):
        self.age_median = 50.0
        self.age_mean = 50.0
        self.age_std = 20.0
        self.sex_categories = ["Female", "Male"]
        self.site_categories = SITE_OPTIONS

    def fit(self, train_df: pd.DataFrame):
        if "age" in train_df.columns:
            ages = pd.to_numeric(train_df["age"], errors="coerce")
            if ages.notna().any():
                self.age_median = float(ages.median())
                clean_ages = ages.fillna(self.age_median)
                self.age_mean = float(clean_ages.mean())
                std_val = float(clean_ages.std())
                self.age_std = std_val if std_val > 1e-6 else 1.0
        return self

    def transform_row(self, row: pd.Series, missing_rate: int = 0, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(42)

        # 1. Age + missing indicator
        raw_age = pd.to_numeric(row.get("age", np.nan), errors="coerce")
        age_missing = pd.isna(raw_age)
        age_val = self.age_median if age_missing else float(raw_age)

        # 2. Sex (one-hot + missing)
        raw_sex = row.get("sex", None)
        sex_missing = pd.isna(raw_sex) or str(raw_sex).strip() == ""
        sex_str = "" if sex_missing else str(raw_sex).strip()

        # 3. Site (one-hot + missing)
        raw_site = row.get("site", None)
        site_missing = pd.isna(raw_site) or str(raw_site).strip() == ""
        site_str = "" if site_missing else str(raw_site).strip()

        # Simulasi missing metadata sesuai missing_rate (0..100%)
        drop_meta = (missing_rate >= 100) or (missing_rate > 0 and rng.random() < (missing_rate / 100.0))
        if drop_meta:
            age_val = self.age_median
            age_missing = True
            sex_str = ""
            sex_missing = True
            site_str = ""
            site_missing = True

        age_norm = (age_val - self.age_mean) / self.age_std
        sex_vec = [1.0 if (not sex_missing and sex_str.lower() == c.lower()) else 0.0 for c in self.sex_categories]
        site_vec = [1.0 if (not site_missing and site_str.lower() == c.lower()) else 0.0 for c in self.site_categories]
        mask_vec = [1.0 if age_missing else 0.0, 1.0 if sex_missing else 0.0, 1.0 if site_missing else 0.0]

        feat = np.array([age_norm] + sex_vec + site_vec + mask_vec, dtype=np.float32)
        return torch.tensor(feat, dtype=torch.float32)


class HybridCNNBiLSTMModel(nn.Module):
    """Hybrid EfficientNet-B0 + LSTM + Metadata MLP untuk klasifikasi MEL/NV/BCC."""

    def __init__(self, meta_dim: int, num_classes: int = 3, lstm_hidden: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        # Ambil feature map spasial sebelum pooling/classifier: (B, 1280, 7, 7)
        self.features = backbone.features
        self.proj = nn.Conv2d(1280, 128, kernel_size=1)

        # Perlakukan 49 posisi spasial (7x7) sebagai urutan sekuensial untuk LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        img_out_dim = lstm_hidden * 2

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(img_out_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
        fmap = self.proj(self.features(x_img))  # (B, 128, 7, 7)
        b, c, h, w = fmap.shape
        seq = fmap.flatten(2).transpose(1, 2)   # (B, 49, 128)
        lstm_out, _ = self.lstm(seq)            # (B, 49, 256)
        img_feat = lstm_out[:, -1, :]           # (B, 256)

        meta_feat = self.meta_mlp(x_meta)       # (B, 64)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)


    from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)


def resolve_image_path(image_id: str, image_dir: str) -> Path | None:
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


class MultimodalLesionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str,
        preprocessor: MetadataPreprocessor,
        train: bool = False,
        missing_rate: int = 0,
        seed: int = 42,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocessor = preprocessor
        self.train = train
        self.missing_rate = missing_rate
        self.rng = np.random.default_rng(seed)

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(15) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)
            if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img_path = resolve_image_path(str(row["image"]), self.image_dir)
        if img_path is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {row['image']}")

        img = Image.open(img_path).convert("RGB")
        x_img = self.img_transform(img)
        x_meta = self.preprocessor.transform_row(row, missing_rate=self.missing_rate, rng=self.rng)
        y = CLASS_ORDER.index(str(row["diagnosis"]).strip())
        return x_img, x_meta, y


def train_multimodal_hybrid(
    frame: pd.DataFrame,
    image_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    pretrained: bool = True,
    missing_rate: int = 0,
    progress_callback=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = frame[frame["split"] == "Train"].copy()
    val_df = frame[frame["split"] == "Validation"].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Split Train atau Validation kosong.")

    prep = MetadataPreprocessor().fit(train_df)
    sample_meta = prep.transform_row(train_df.iloc[0], missing_rate=0)
    meta_dim = int(sample_meta.numel())

    train_ds = MultimodalLesionDataset(train_df, image_dir, prep, train=True, missing_rate=missing_rate)
    val_ds = MultimodalLesionDataset(val_df, image_dir, prep, train=False, missing_rate=missing_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HybridCNNBiLSTMModel(meta_dim=meta_dim, num_classes=len(CLASS_ORDER), pretrained=pretrained).to(device)

    counts = train_df["diagnosis"].value_counts().reindex(CLASS_ORDER, fill_value=1).values
    weights = len(train_df) / (len(CLASS_ORDER) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    history = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        tr_loss_sum, tr_true, tr_pred = 0.0, [], []
        for x_img, x_meta, y in train_loader:
            x_img, x_meta, y = x_img.to(device), x_meta.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_img, x_meta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * len(y)
            tr_true.extend(y.detach().cpu().numpy())
            tr_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

        model.eval()
        va_loss_sum, va_true, va_pred = 0.0, [], []
        with torch.no_grad():
            for x_img, x_meta, y in val_loader:
                x_img, x_meta, y = x_img.to(device), x_meta.to(device), y.to(device)
                logits = model(x_img, x_meta)
                loss = criterion(logits, y)

                va_loss_sum += loss.item() * len(y)
                va_true.extend(y.cpu().numpy())
                va_pred.extend(logits.argmax(dim=1).cpu().numpy())

        tr_loss = tr_loss_sum / len(train_ds)
        va_loss = va_loss_sum / len(val_ds)
        va_f1 = f1_score(va_true, va_pred, average="macro", zero_division=0)
        scheduler.step(va_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_accuracy": accuracy_score(tr_true, tr_pred),
            "val_accuracy": accuracy_score(va_true, va_pred),
            "val_macro_f1": va_f1,
        })

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, prep, pd.DataFrame(history), device


def evaluate_multimodal_hybrid(
    model: nn.Module,
    preprocessor: MetadataPreprocessor,
    frame: pd.DataFrame,
    image_dir: str,
    batch_size: int = 32,
    missing_rate: int = 0,
):
    device = next(model.parameters()).device
    test_df = frame[frame["split"] == "Test"].copy()
    if len(test_df) == 0:
        raise ValueError("Test split kosong.")

    ds = MultimodalLesionDataset(test_df, image_dir, preprocessor, train=False, missing_rate=missing_rate)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for x_img, x_meta, y in loader:
            logits = model(x_img.to(device), x_meta.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(prob.argmax(axis=1))
            y_prob.extend(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "report": classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=CLASS_ORDER, output_dict=True, zero_division=0
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }