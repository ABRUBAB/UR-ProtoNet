"""Dataset loaders, patient-level splits, and episodic data generators for UR-ProtoNet.

Handles NIH ChestX-ray14, Kermany CXR, and Stanford CheXpert benchmarks with patient-leakage-free
splitting, U-Ones uncertainty label mapping, and deterministic episodic few-shot sampling.
"""

import os
import sys
import glob
import random
from typing import Tuple, List, Optional, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from config import cfg

DEVICE = torch.device(cfg.device if hasattr(cfg, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))
AMP_DEVICE = "cuda" if DEVICE.type == "cuda" else "cpu"
AMP_ENABLED = (DEVICE.type == "cuda") and getattr(cfg, "use_amp", True)


def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------------
# Image Transforms
# ------------------------------------------------------------------
if HAS_TORCHVISION:
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])
else:
    def _fallback_tf(img: Image.Image, train: bool = False) -> torch.Tensor:
        resized = img.resize((cfg.image_size, cfg.image_size), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = (arr - np.array(cfg.mean)) / np.array(cfg.std)
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    train_tf = lambda img: _fallback_tf(img, train=True)
    eval_tf = lambda img: _fallback_tf(img, train=False)


# ------------------------------------------------------------------
# 1.1 CXR Dataset Class
# ------------------------------------------------------------------
class CXRDataset(Dataset):
    """Generic chest radiograph dataset loaded from a DataFrame containing filepath and label."""
    def __init__(self, df: pd.DataFrame, train: bool = True, return_meta: bool = False):
        self.df = df.reset_index(drop=True)
        self.return_meta = return_meta
        self.tf = train_tf if train else eval_tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fp = row["filepath"]
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (cfg.image_size, cfg.image_size), 0)
        x = self.tf(img)
        y = int(row["label"])
        if self.return_meta:
            return x, y, str(fp), str(row.get("patient_id", idx))
        return x, y


# ------------------------------------------------------------------
# 1.2 Patient-level splitting
# ------------------------------------------------------------------
def patient_level_split(df: pd.DataFrame, group_col: str, test_size: float = 0.2,
                        val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset strictly by patient ID to prevent cross-partition leakage."""
    groups = df[group_col].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=groups))
    train_val = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1.0 - test_size),
        random_state=seed
    )
    tv_groups = train_val[group_col].values
    tr2, va2 = next(gss2.split(train_val, groups=tv_groups))
    train = train_val.iloc[tr2].reset_index(drop=True)
    val = train_val.iloc[va2].reset_index(drop=True)
    return train, val, test


# ------------------------------------------------------------------
# 1.3 NIH ChestX-ray14 Loader
# ------------------------------------------------------------------
def find_nih_root(custom_root: Optional[str] = None) -> Optional[str]:
    candidates = [
        custom_root or cfg.nih_root,
        "/kaggle/input/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized",
        "/kaggle/input/nih-chest-xrays/data",
        "data/nih",
    ]
    for p in glob.glob("/kaggle/input/*nih*") + glob.glob("data/*nih*"):
        candidates.append(p)
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "Data_Entry_2017.csv")):
            return c
    return None


def map_nih_pneumonia(labels_str: str) -> int:
    s = str(labels_str)
    if "Pneumonia" in s:
        return 1
    if "No Finding" in s:
        return 0
    return -1


def load_nih_df(nih_root: Optional[str] = None) -> Optional[pd.DataFrame]:
    root = find_nih_root(nih_root)
    if not root:
        return None
    labels_path = os.path.join(root, "Data_Entry_2017.csv")
    if not os.path.exists(labels_path):
        return None
    df_raw = pd.read_csv(labels_path)
    sample_name = df_raw["Image Index"].iloc[0]
    matches = glob.glob(os.path.join(root, "**", sample_name), recursive=True)
    images_root = os.path.dirname(matches[0]) if matches else root

    df_raw["filepath"] = df_raw["Image Index"].apply(lambda x: os.path.join(images_root, str(x)))
    df_raw["label"] = df_raw["Finding Labels"].apply(map_nih_pneumonia)
    df = df_raw[df_raw["label"] >= 0].copy()
    df = df[df["filepath"].apply(os.path.exists)]
    df["dataset"] = "NIH"
    df["patient_id"] = df["Patient ID"].astype(str)
    return df[["filepath", "label", "dataset", "patient_id"]].reset_index(drop=True)


# ------------------------------------------------------------------
# 1.4 CXR Loader
# ------------------------------------------------------------------
def _walk_class_dir(dirpath: str, label: int, dataset_tag: str = "CXR") -> List[Dict]:
    rows = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                fp = os.path.join(root, f)
                rows.append({
                    "filepath": fp,
                    "label": label,
                    "dataset": dataset_tag,
                    "patient_id": os.path.splitext(os.path.basename(fp))[0]
                })
    return rows


def load_cxr_df(cxr_root: Optional[str] = None) -> Optional[pd.DataFrame]:
    root = cxr_root or cfg.cxr_root
    if not root or not os.path.isdir(root):
        return None
    class_map = {"normal": 0, "pneumonia": 1}
    rows = []
    for cls_folder in os.listdir(root):
        label = class_map.get(cls_folder.lower())
        cls_path = os.path.join(root, cls_folder)
        if label is not None and os.path.isdir(cls_path):
            rows.extend(_walk_class_dir(cls_path, label))
    if not rows:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(root, split)
            if os.path.isdir(split_dir):
                for cls_folder in os.listdir(split_dir):
                    label = class_map.get(cls_folder.lower())
                    cls_path = os.path.join(split_dir, cls_folder)
                    if label is not None and os.path.isdir(cls_path):
                        rows.extend(_walk_class_dir(cls_path, label))
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates(subset="filepath").reset_index(drop=True)


# ------------------------------------------------------------------
# 1.5 CheXpert Loader (U-Ones Mapping)
# ------------------------------------------------------------------
def find_chexpert_root(custom_root: Optional[str] = None) -> Optional[str]:
    candidates = [
        custom_root or cfg.chexpert_root,
        "/kaggle/input/datasets/ashery/chexpert/CheXpert-v1.0-small",
        "/kaggle/input/datasets/ashery/chexpert",
        "/kaggle/input/chexpert-v10-small",
        "/kaggle/input/chexpert-v1.0-small",
        "data/chexpert",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    return None


def resolve_chexpert_paths(df_c: pd.DataFrame, chexpert_root: str) -> pd.Series:
    sample = str(df_c["Path"].iloc[0]).replace("\\", "/")
    parts = sample.split("/")
    for strip_n in range(len(parts) - 1):
        stripped = "/".join(parts[strip_n:])
        test_fp = os.path.join(chexpert_root, stripped)
        if os.path.exists(test_fp):
            def _resolve(p, _sn=strip_n, _base=chexpert_root):
                p_parts = str(p).replace("\\", "/").split("/")
                return os.path.join(_base, "/".join(p_parts[_sn:]))
            return df_c["Path"].apply(_resolve)
    return df_c["Path"].apply(lambda p: os.path.join(chexpert_root, str(p).replace("\\", "/")))


def load_chexpert_df(chexpert_root: Optional[str] = None, max_per_class: int = 5000,
                     seed: int = 42) -> Optional[pd.DataFrame]:
    root = find_chexpert_root(chexpert_root)
    if not root:
        return None
    dfs = []
    for name in ["train.csv", "valid.csv"]:
        p = os.path.join(root, name)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    df_c = pd.concat(dfs, ignore_index=True)
    if "Frontal/Lateral" in df_c.columns:
        df_c = df_c[df_c["Frontal/Lateral"] == "Frontal"].copy()

    df_c["label"] = -1
    for col in ["Pneumonia", "Consolidation", "Lung Opacity"]:
        if col in df_c.columns:
            vals = df_c[col].fillna(0)
            df_c.loc[(vals == 1.0) | (vals == -1.0), "label"] = 1
    no_find = df_c.get("No Finding", pd.Series(0, index=df_c.index))
    df_c.loc[(no_find == 1.0) & (df_c["label"] == -1), "label"] = 0
    df_c = df_c[df_c["label"] >= 0].copy()

    df_c["filepath"] = resolve_chexpert_paths(df_c, root)
    df_c = df_c[df_c["filepath"].apply(os.path.exists)].copy()
    df_c["patient_id"] = df_c["Path"].apply(
        lambda p: next((x for x in str(p).replace("\\", "/").split("/") if x.startswith("patient")),
                       os.path.basename(str(p))))
    df_c["dataset"] = "CheXpert"
    res = df_c[["filepath", "label", "dataset", "patient_id"]].reset_index(drop=True)

    # Subsample balanced
    subs = []
    for cls in [0, 1]:
        cdf = res[res["label"] == cls]
        if len(cdf) > max_per_class:
            cdf = cdf.sample(max_per_class, random_state=seed)
        subs.append(cdf)
    return pd.concat(subs).reset_index(drop=True)


# ------------------------------------------------------------------
# 1.6 Episode Sampling & EpisodeDataset
# ------------------------------------------------------------------
def sample_episode(df: pd.DataFrame, n_way: int, k_shot: int, q_queries: int,
                   rng: np.random.RandomState) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classes = sorted(df["label"].unique().tolist())
    chosen = rng.choice(classes, size=n_way, replace=False)
    s_rows, q_rows = [], []
    for c in chosen:
        c_df = df[df["label"] == c]
        need = k_shot + q_queries
        rs = int(rng.randint(0, 2**31 - 1))
        sampled = c_df.sample(need, replace=len(c_df) < need, random_state=rs)
        s_rows.append(sampled.iloc[:k_shot])
        q_rows.append(sampled.iloc[k_shot:])
    return (pd.concat(s_rows).reset_index(drop=True),
            pd.concat(q_rows).reset_index(drop=True))


class EpisodeDataset(Dataset):
    """Generates few-shot N-way K-shot episodes with deterministic seeding per epoch."""
    def __init__(self, df: pd.DataFrame, episodes: int, n_way: int, k_shot: int,
                 q_queries: int, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.episodes = episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.base_rng = np.random.RandomState(seed)
        self._seeds = self.base_rng.randint(0, 2**31 - 1, size=episodes * 100)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return self.episodes

    def _load_batch(self, df_sub: pd.DataFrame, train: bool = False):
        ds = CXRDataset(df_sub, train=train, return_meta=False)
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

    def __getitem__(self, idx: int):
        si = (self._epoch * self.episodes + idx) % len(self._seeds)
        rng = np.random.RandomState(int(self._seeds[si]))
        s_df, q_df = sample_episode(self.df, self.n_way, self.k_shot, self.q_queries, rng)
        s_x, s_y = self._load_batch(s_df, train=False)
        q_x, q_y = self._load_batch(q_df, train=False)
        return s_x, s_y, q_x, q_y


def sample_support_df(source_df: pd.DataFrame, k_shot: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for cls in [0, 1]:
        cdf = source_df[source_df["label"] == cls]
        rows.append(cdf.sample(k_shot, replace=len(cdf) < k_shot,
                               random_state=int(rng.randint(1, 10**9))))
    return pd.concat(rows).reset_index(drop=True)


@torch.no_grad()
def build_support_tensors(support_df: pd.DataFrame, device: Optional[torch.device] = None):
    dev = device or DEVICE
    ds = CXRDataset(support_df, train=False, return_meta=False)
    dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
    for x, y in dl:
        return x.to(dev), y.to(dev)


if __name__ == "__main__":
    print("=" * 70)
    print("UR-ProtoNet Data Subsystem Diagnostics")
    print("=" * 70)
    nih = find_nih_root()
    print(f"NIH root resolved: {nih}")
    chex = find_chexpert_root()
    print(f"CheXpert root resolved: {chex}")
    print(f"Device: {DEVICE} (AMP enabled: {AMP_ENABLED})")
    print("=" * 70)