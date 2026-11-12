"""Dataset discovery, label definitions, preprocessing and an in-memory uint8 image cache.

All four corpora are preprocessed once (percentile clipping -> CLAHE -> 224x224) and stored as
uint8 arrays so that episodic training reads images from RAM instead of decoding JPEG/PNG files
inside every episode.
"""

import os
import re
import gzip
import json
import math
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

IMG_EXT = (".png", ".jpg", ".jpeg")
INPUT_ROOT = os.environ.get("URP_INPUT", "/kaggle/input")


# ----------------------------------------------------------------------------------------------
# Root discovery
# ----------------------------------------------------------------------------------------------
def _walk_dirs(root: str, max_depth: int = 6, max_children: int = 500):
    """os.walk that never descends into huge directories (e.g. CheXpert's 65k patient folders)."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base_depth >= max_depth or len(dirs) > max_children:
            dirs[:] = []
        yield d, dirs, files


def find_file(name: str, root: str = INPUT_ROOT, max_depth: int = 7) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    for d, _, files in _walk_dirs(root, max_depth):
        if name in files:
            return os.path.join(d, name)
    return None


def find_dir_with(marker_files, root: str = INPUT_ROOT, must_contain_dirs=(), max_depth: int = 6,
                  name_hint: Optional[str] = None) -> Optional[str]:
    """First directory under `root` that contains all `marker_files` (and sub-dirs)."""
    hits = []
    for d, dirs, files in _walk_dirs(root, max_depth):
        fs = set(files)
        ds = {x.lower() for x in dirs}
        if all(m in fs for m in marker_files) and all(x.lower() in ds for x in must_contain_dirs):
            hits.append(d)
    if name_hint:
        hinted = [h for h in hits if name_hint.lower() in h.lower()]
        if hinted:
            return hinted[0]
    return hits[0] if hits else None


def find_cxr_root(root: str = INPUT_ROOT) -> Optional[str]:
    for d, dirs, _ in _walk_dirs(root, 5):
        low = {x.lower() for x in dirs}
        if "normal" in low and "pneumonia" in low and "cxr" in d.lower():
            return d
    for d, dirs, _ in _walk_dirs(root, 5):
        low = {x.lower() for x in dirs}
        if "normal" in low and "pneumonia" in low:
            return d
    return None


def discover_roots(root: str = INPUT_ROOT) -> Dict[str, Optional[str]]:
    r = {
        "nih": find_dir_with(["Data_Entry_2017.csv"], root),
        "cxr": find_cxr_root(root),
        "chexpert": find_dir_with(["train.csv", "valid.csv"], root, must_contain_dirs=("train",), name_hint="chexpert"),
        "siim_img": find_dir_with(["meta.csv"], root, must_contain_dirs=("train",), name_hint="siim"),
        "siim_lbl": find_dir_with(["train_study_level.csv", "train_image_level.csv"], root),
        "cache": find_dir_with(["urp_cache_index.json"], root),
    }
    return r


def index_images(root: str) -> Dict[str, str]:
    """basename -> full path for every image file below root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                out.setdefault(f, os.path.join(d, f))
    return out


# ----------------------------------------------------------------------------------------------
# Tables (one row per image): key, path, label, group, dataset
# ----------------------------------------------------------------------------------------------
def _read_list_gz(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f)


def nih_table(nih_root: str, mem_normal: int = 8000, seed: int = 42) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(nih_root, "Data_Entry_2017.csv"))
    labels = raw["Finding Labels"].astype(str)
    lab = np.where(labels.str.contains("Pneumonia"), 1, np.where(labels.str.contains("No Finding"), 0, -1))
    raw["label"] = lab
    df = raw[raw["label"] >= 0].copy()
    idx = index_images(nih_root)
    df["path"] = df["Image Index"].map(idx)
    df = df[df["path"].notna()].reset_index(drop=True)
    df["key"] = df["Image Index"]
    df["group"] = "nih_" + df["Patient ID"].astype(str)
    df["dataset"] = "NIH"
    # Stage II memory membership: 8,000 Normal sampled uniformly at random, all Pneumonia retained
    normal_idx = df.index[df["label"] == 0]
    rng = np.random.RandomState(seed)
    keep = rng.choice(normal_idx, size=min(mem_normal, len(normal_idx)), replace=False)
    df["in_memory"] = 0
    df.loc[keep, "in_memory"] = 1
    df.loc[df["label"] == 1, "in_memory"] = 1
    # Patient-level monitoring split for Stage I curves
    pats = np.array(sorted(df["group"].unique()))
    rng2 = np.random.RandomState(seed + 1)
    val_pats = set(rng2.choice(pats, size=int(0.1 * len(pats)), replace=False))
    df["split"] = np.where(df["group"].isin(val_pats), "val", "train")
    return df[["key", "path", "label", "group", "dataset", "in_memory", "split"]]


NIH14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
         "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def nih14_table(nih_root: str, seed: int = 42) -> pd.DataFrame:
    """Every ChestX-ray14 image with its 14 finding labels (multi-label Stage I), patient-level 90/10 split."""
    raw = pd.read_csv(os.path.join(nih_root, "Data_Entry_2017.csv"))
    labels = raw["Finding Labels"].astype(str)
    idx = index_images(nih_root)
    df = pd.DataFrame({"key": raw["Image Index"].values, "path": raw["Image Index"].map(idx).values,
                       "group": ("nih_" + raw["Patient ID"].astype(str)).values, "dataset": "NIH"})
    parts = labels.str.split("|")
    for n in NIH14:
        df["y_" + n] = parts.map(lambda p, n=n: int(n in p)).values
    df["label"] = np.where(df["y_Pneumonia"].values == 1, 1, np.where(labels.str.contains("No Finding").values, 0, -1))
    df = df[df["path"].notna()].reset_index(drop=True)
    pats = np.array(sorted(df["group"].unique()))
    rng = np.random.RandomState(seed + 1)
    val_pats = set(rng.choice(pats, size=int(0.1 * len(pats)), replace=False))
    df["split"] = np.where(df["group"].isin(val_pats), "val", "train")
    return df


_CXR_PAT_RULES = [
    (re.compile(r"^(person\d+)_", re.I), lambda m: "k_" + m.group(1).lower()),
    (re.compile(r"^(?:bacteria|virus|normal)-(\d+)-\d+", re.I), lambda m: "m_" + m.group(1)),
    (re.compile(r"^(?:normal2-)?im-(\d+)-\d+", re.I), lambda m: "im_" + m.group(1)),
]


def cxr_patient_id(fname: str) -> str:
    s = os.path.splitext(os.path.basename(fname))[0]
    for rx, fn in _CXR_PAT_RULES:
        m = rx.match(s)
        if m:
            return fn(m)
    return "img_" + s.lower()


def cxr_table(cxr_root: str, assets_dir: str) -> pd.DataFrame:
    rows = []
    for cls_dir in os.listdir(cxr_root):
        lab = {"normal": 0, "pneumonia": 1}.get(cls_dir.lower())
        if lab is None:
            continue
        for d, _, files in os.walk(os.path.join(cxr_root, cls_dir)):
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    full = os.path.join(d, f)
                    rel = os.path.relpath(full, cxr_root).replace("\\", "/")
                    rows.append({"key": rel, "path": full, "label": lab})
    df = pd.DataFrame(rows).drop_duplicates("key").reset_index(drop=True)
    tr = set(_read_list_gz(os.path.join(assets_dir, "cxr_train.csv.gz"))["key"])
    va = set(_read_list_gz(os.path.join(assets_dir, "cxr_val.csv.gz"))["key"])
    df["split"] = np.where(df["key"].isin(tr), "train", np.where(df["key"].isin(va), "val", "test"))
    df["group"] = df["key"].map(cxr_patient_id)
    df["dataset"] = "CXR"
    return df[["key", "path", "label", "group", "dataset", "split"]]


NIH_NAME = re.compile(r"^\d{8}_\d{3}\.png$")


def cxr_clean_flags(df: pd.DataFrame, nih_overlap_keys=None, exclude_nih_from_train: bool = False) -> pd.DataFrame:
    """Strictly held-out evaluation subsets of the CXR corpus.

    The aggregated CXR corpus contains re-encoded copies of the same radiograph under different file
    names and includes NIH-derived images. An evaluation image is kept only if (i) no image in the
    earlier partitions has the same perceptual hash (dHash on the preprocessed 224 px image), (ii) its
    parsed patient identifier does not occur in those partitions and (iii) it is not an NIH file
    (NIH images are used for Stage I and the memory bank). Val is checked against train; test is
    checked against train and val.
    """
    df = df.copy()
    fname = df["key"].astype(str).str.split("/").str[-1]
    nih = fname.str.match(NIH_NAME)
    if nih_overlap_keys is not None:  # perceptual-hash duplicates of NIH ChestX-ray14 images
        nih = nih | df["key"].isin(set(nih_overlap_keys))
    tr = df["split"] == "train"
    va = df["split"] == "val"
    te = df["split"] == "test"
    h_tr, p_tr = set(df.loc[tr, "dhash"]), set(df.loc[tr, "group"])
    h_trva, p_trva = h_tr | set(df.loc[va, "dhash"]), p_tr | set(df.loc[va, "group"])
    clean = np.ones(len(df), dtype=bool)
    clean[va.values] = (~df.loc[va, "dhash"].isin(h_tr) & ~df.loc[va, "group"].isin(p_tr) & ~nih[va]).values
    clean[te.values] = (~df.loc[te, "dhash"].isin(h_trva) & ~df.loc[te, "group"].isin(p_trva) & ~nih[te]).values
    df["nih_origin"] = nih.astype(int)
    df["clean"] = clean.astype(int)
    # split used for evaluation / model selection: strict subsets of val and test
    df["eval_split"] = np.where(tr, "train", np.where(df["clean"] == 1, df["split"], "excluded"))
    if exclude_nih_from_train:
        df.loc[tr.values & nih.values, "eval_split"] = "excluded"
    return df


def chexpert_labels(df_c: pd.DataFrame) -> pd.Series:
    """U-Ones mapping: Pneumonia / Consolidation / Lung Opacity in {1, -1} -> 1; No Finding -> 0."""
    lab = pd.Series(-1, index=df_c.index)
    for col in ["Pneumonia", "Consolidation", "Lung Opacity"]:
        if col in df_c.columns:
            v = df_c[col].fillna(0)
            lab[(v == 1.0) | (v == -1.0)] = 1
    nf = df_c["No Finding"] if "No Finding" in df_c.columns else pd.Series(0, index=df_c.index)
    lab[(nf == 1.0) & (lab == -1)] = 0
    return lab


def chexpert_table(cx_root: str, assets_dir: str, pool_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    dfs = [pd.read_csv(os.path.join(cx_root, n)) for n in ("train.csv", "valid.csv")
           if os.path.exists(os.path.join(cx_root, n))]
    d = pd.concat(dfs, ignore_index=True)
    if "Frontal/Lateral" in d.columns:
        d = d[d["Frontal/Lateral"] == "Frontal"].copy()
    d["label"] = chexpert_labels(d)
    d = d[d["label"] >= 0].copy()
    # "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg" -> "train/patient00001/..."
    d["key"] = d["Path"].astype(str).str.replace("\\", "/", regex=False).str.replace(r"^[^/]*CheXpert[^/]*/", "", regex=True)
    d["path"] = d["key"].map(lambda k: os.path.join(cx_root, k))
    d["group"] = d["key"].str.extract(r"(patient\d+)")[0]
    d["study"] = d["key"].str.extract(r"(patient\d+/study\d+)")[0]
    q = _read_list_gz(os.path.join(assets_dir, "chexpert_query.csv.gz"))  # key,label (published query set)
    qkeys = set(q["key"])
    query = d[d["key"].isin(qkeys)].copy()
    query["role"] = "query"
    # sanity: labels recomputed here must equal the published query labels
    lab_map = dict(zip(q["key"], q["label"]))
    query["label_published"] = query["key"].map(lab_map)
    qpat = set(query["group"])
    rest = d[~d["group"].isin(qpat)]
    pools = []
    for c in (0, 1):
        cdf = rest[rest["label"] == c]
        pools.append(cdf.sample(min(pool_per_class, len(cdf)), random_state=seed))
    pool = pd.concat(pools).copy()
    pool["role"] = "pool"
    out = pd.concat([query, pool], ignore_index=True)
    out = out[out["path"].map(os.path.exists)].reset_index(drop=True)
    out["dataset"] = "CheXpert"
    return out[["key", "path", "label", "group", "study", "role", "dataset", "label_published"]]


def siim_table(img_root: str, lbl_root: str, pool_studies_per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    """SIIM-FISABIO-RSNA (BIMCV + MIDRC): Negative for Pneumonia -> 0, any pneumonia appearance -> 1."""
    il = pd.read_csv(os.path.join(lbl_root, "train_image_level.csv"))
    il = il.drop(columns=[c for c in ("label", "boxes") if c in il.columns])  # box strings, not used
    sl = pd.read_csv(os.path.join(lbl_root, "train_study_level.csv"))
    sl["StudyInstanceUID"] = sl["id"].str.replace("_study", "", regex=False)
    sl["label"] = (sl["Negative for Pneumonia"] != 1).astype(int)
    il["image_id"] = il["id"].str.replace("_image", "", regex=False)
    df = il.merge(sl[["StudyInstanceUID", "label"]], on="StudyInstanceUID", how="inner")
    idx = index_images(os.path.join(img_root, "train"))
    df["path"] = df["image_id"].map(lambda i: idx.get(i + ".png") or idx.get(i + ".jpg"))
    df = df[df["path"].notna()].reset_index(drop=True)
    df["key"] = df["image_id"]
    df["group"] = df["StudyInstanceUID"]
    rng = np.random.RandomState(seed)
    pool_studies = set()
    st = df.groupby("group")["label"].max()
    for c in (0, 1):
        ids = st[st == c].index.values
        pool_studies |= set(rng.choice(ids, size=min(pool_studies_per_class, len(ids)), replace=False))
    df["role"] = np.where(df["group"].isin(pool_studies), "pool", "query")
    df["dataset"] = "SIIM"
    return df[["key", "path", "label", "group", "role", "dataset"]]


# ----------------------------------------------------------------------------------------------
# Preprocessing (Section 3.3): percentile clip -> 8-bit rescale -> CLAHE(2.0, 8x8) -> 224x224
# ----------------------------------------------------------------------------------------------
def preprocess_image(path: str, size: int = 224) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    m = max(h, w)
    if m > 1024:  # speed only: CLAHE is applied at <=1024 px on the long side
        s = 1024.0 / m
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    img = (np.clip((img - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _pp_worker(args):
    i, path, size = args
    try:
        a = preprocess_image(path, size)
    except Exception:
        a = None
    return i, a


def dhash(a: np.ndarray) -> str:
    small = cv2.resize(a, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)


def build_cache(df: pd.DataFrame, name: str, out_dir: str, size: int = 224, procs: Optional[int] = None) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    arr = np.lib.format.open_memmap(os.path.join(out_dir, f"{name}.npy"), mode="w+", dtype=np.uint8,
                                    shape=(n, size, size))
    ok = np.zeros(n, dtype=bool)
    hashes = [""] * n
    jobs = [(i, p, size) for i, p in enumerate(df["path"].tolist())]
    with Pool(procs or os.cpu_count()) as pool:
        for k, (i, a) in enumerate(pool.imap_unordered(_pp_worker, jobs, chunksize=64)):
            if a is not None:
                arr[i] = a
                ok[i] = True
                hashes[i] = dhash(a)
            if (k + 1) % 5000 == 0:
                print(f"  [{name}] {k + 1}/{n}", flush=True)
    arr.flush()
    del arr
    out = df.copy().reset_index(drop=True)
    out["cache_row"] = np.arange(n)
    out["ok"] = ok.astype(int)
    out["dhash"] = hashes
    out.drop(columns=["path"]).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    print(f"  [{name}] done: {ok.sum()}/{n} readable", flush=True)
    return out


def load_cache(cache_dir: str, name: str, in_ram: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    arr = np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r")
    df = pd.read_csv(os.path.join(cache_dir, f"{name}.csv"), low_memory=False)
    if in_ram:
        arr = np.array(arr)  # force a real copy into RAM (ascontiguousarray would keep the memmap)
    keep = df["ok"] == 1
    return arr, df[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# GPU input pipeline and augmentation (flip p=0.5, rotation +-10 deg, translation +-5 %)
# ----------------------------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gpu_augment(x: torch.Tensor, max_rot_deg: float = 10.0, max_shift: float = 0.05) -> torch.Tensor:
    b = x.size(0)
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2.0   # normalised coords span 2
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2.0
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c * flip, -s, tx], -1), torch.stack([s * flip, c, ty], -1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _gauss1d(sigma: float, k: int, device) -> torch.Tensor:
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gpu_augment_dg(x: torch.Tensor, max_rot_deg: float = 10.0, max_shift: float = 0.05) -> torch.Tensor:
    """Domain-generalisation augmentation, x in [0, 1] with shape [B, 1, H, W]; parameters are drawn per image.

    Geometric: flip, rotation +-10 deg, translation +-5 % (as gpu_augment) and a zoom-in crop that keeps
    75-100 % of the field of view with +-10 % aspect jitter. Photometric, imitating acquisition and
    post-processing differences between sites: gamma (p 0.5, exp U(-0.4, 0.4)), contrast x U(0.75, 1.25) and
    brightness +-0.08 about the image mean (p 0.5), Gaussian blur <-> unsharp masking (p 0.5, sigma 1.5,
    amount U(-1, 1)), resolution loss by 0.5-0.9x down/up-sampling (p 0.25), Gaussian noise sd <= 0.03 (p 0.25)."""
    b, dev = x.size(0), x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2.0
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2.0
    sc = 0.75 + 0.25 * torch.rand(b, device=dev)
    ar = torch.exp((torch.rand(b, device=dev) * 2 - 1) * 0.1)
    sx, sy = sc * ar, sc / ar
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c * flip * sx, -s * sy, tx], -1), torch.stack([s * flip * sx, c * sy, ty], -1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def on(p):
        return (torch.rand(b, device=dev) < p).view(b, 1, 1, 1)

    def uni(lo, hi):
        return (lo + (hi - lo) * torch.rand(b, device=dev)).view(b, 1, 1, 1)

    x = torch.where(on(0.5), x.clamp(1e-4, 1.0) ** torch.exp(uni(-0.4, 0.4)), x)                 # gamma
    m = x.mean(dim=(2, 3), keepdim=True)
    x = torch.where(on(0.5), (x - m) * uni(0.75, 1.25) + m + uni(-0.08, 0.08), x)                # contrast / brightness
    k = _gauss1d(1.5, 7, dev)
    xb = F.conv2d(F.pad(x, (3, 3, 3, 3), mode="replicate"), k.view(1, 1, 1, 7))
    xb = F.conv2d(xb, k.view(1, 1, 7, 1))
    x = torch.where(on(0.5), x + uni(-1.0, 1.0) * (x - xb), x)                                   # blur <-> sharpen
    sel = torch.rand(b, device=dev) < 0.25                                                       # resolution loss
    if bool(sel.any()):
        h, w = x.shape[-2:]
        r = 0.5 + 0.4 * float(torch.rand(1).item())
        low = F.interpolate(x[sel], size=(max(8, int(h * r)), max(8, int(w * r))), mode="bilinear",
                            align_corners=False, antialias=True)
        x = x.clone()
        x[sel] = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    x = torch.where(on(0.25), x + torch.randn_like(x) * uni(0.0, 0.03), x)                       # noise
    return x.clamp(0.0, 1.0)


def to_input(u8, device, augment: bool = False, aug_mode: str = "basic") -> torch.Tensor:
    """uint8 [B,H,W] (numpy or tensor) -> normalised float [B,3,H,W] on device."""
    x = torch.as_tensor(np.asarray(u8)).to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
    if augment:
        x = gpu_augment_dg(x) if aug_mode == "dg" else gpu_augment(x)
    x = x.expand(-1, 3, -1, -1)
    return (x - _MEAN.to(device)) / _STD.to(device)


def iter_batches(arr: np.ndarray, rows: np.ndarray, bs: int):
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        yield r, arr[r]


def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
