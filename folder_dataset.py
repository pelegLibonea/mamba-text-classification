"""
Folder-based dataset for two-stage text classification.

Stage 1: Binary classification (medical vs no_medical)
Stage 2A: Medical leaf class classification (multi-class)
Stage 2B: No-medical leaf class classification (multi-class)

Expects a directory structure:
    data_dir/
        medical/
            class_folder1/*.txt    (e.g., "01=Cardiology")
            class_folder2/*.txt
        no_medical/
            class_folder3/*.txt
            ...

Folder names can use "code=title", "code_title", "name.cat", or plain "name".
"""

import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Placeholder text returned for empty / unreadable files.
EMPTY_TEXT = "[EMPTY]"


# ---------------------------------------------------------------------------
# Folder parsing & text reading
# ---------------------------------------------------------------------------

def parse_class_folder(folder_name: str) -> Dict[str, str]:
    """Parse a class folder name into code/title/raw components."""
    raw = folder_name
    name = folder_name.replace(".cat", "").strip()
    code, title = name, name
    if "=" in name:
        a, b = name.split("=", 1)
        code, title = a.strip(), b.strip()
    elif "_" in name:
        a, b = name.split("_", 1)
        code, title = a.strip(), b.strip()
    return {"code": code, "title": title, "folder_name": raw}


@lru_cache(maxsize=200000)
def read_text_cached(path_str: str) -> str:
    """Read and clean a text file, caching the result."""
    try:
        txt = Path(path_str).read_text(encoding="utf-8", errors="ignore")
        txt = txt.replace("\x00", " ").strip()
        if not txt:
            return EMPTY_TEXT
        cleaned = [c if c.isprintable() or c in "\n\t\r " else " " for c in txt]
        out = "".join(cleaned).strip()
        return out if out else EMPTY_TEXT
    except Exception:
        return EMPTY_TEXT


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_samples(
    data_dir: Path,
    medical_ignore_folders: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Walk through data_dir/medical and data_dir/no_medical,
    collecting text samples from class subfolders.

    Args:
        data_dir: Root directory containing ``medical`` and ``no_medical`` subdirs.
        medical_ignore_folders: Folder-name patterns to skip inside the medical branch.

    Returns:
        DataFrame with columns: text_path, branch, leaf_folder, leaf_code, leaf_title
    """
    rows: List[dict] = []
    ignored_count = 0

    for branch in ("medical", "no_medical"):
        bdir = data_dir / branch
        if not bdir.exists():
            continue

        for class_folder in sorted(bdir.iterdir()):
            if not class_folder.is_dir():
                continue
            info = parse_class_folder(class_folder.name)

            # Skip ignored folders for medical branch
            if branch == "medical" and medical_ignore_folders:
                folder_title = info["title"].lower().strip()
                folder_name = class_folder.name.lower().strip()
                skip = False
                for pattern in medical_ignore_folders:
                    if pattern.lower().strip() in folder_title or pattern.lower().strip() in folder_name:
                        skip = True
                        break
                if skip:
                    skipped = len(list(class_folder.glob("*.txt")))
                    ignored_count += skipped
                    print(f"  Ignoring medical folder: {class_folder.name} ({skipped} samples)")
                    continue

            for txt_path in class_folder.glob("*.txt"):
                try:
                    if not txt_path.read_text(encoding="utf-8", errors="ignore").strip():
                        continue
                except Exception:
                    continue

                rows.append({
                    "text_path": str(txt_path),
                    "branch": branch,
                    "leaf_folder": class_folder.name,
                    "leaf_code": info["code"],
                    "leaf_title": info["title"],
                })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(
            f"Collected samples: {len(df)} | ignored: {ignored_count} "
            f"| leaf classes: {df['leaf_title'].nunique()} "
            f"| branches: {df['branch'].value_counts().to_dict()}"
        )
    else:
        print("No samples collected.")
    return df


# ---------------------------------------------------------------------------
# Filtering / capping
# ---------------------------------------------------------------------------

def filter_min_samples(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Remove classes with fewer than *min_samples*."""
    vc = df["leaf_title"].value_counts()
    keep = vc[vc >= min_samples].index
    drop = vc[vc < min_samples]
    out = df[df["leaf_title"].isin(keep)].copy().reset_index(drop=True)
    if len(drop) > 0:
        print(f"  Dropped {len(drop)} classes with < {min_samples} samples")
    return out


def cap_overrepresented_classes(df: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    """Down-sample classes with more than *max_samples*."""
    vc = df["leaf_title"].value_counts()
    if (vc > max_samples).sum() == 0:
        return df

    print(f"  Capping {(vc > max_samples).sum()} classes with > {max_samples} samples")
    parts = []
    for cls in vc.index:
        sub = df[df["leaf_title"] == cls]
        if len(sub) > max_samples:
            sub = sub.sample(n=max_samples, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    print(f"  Samples: {len(df)} -> {len(out)} (removed {len(df) - len(out)})")
    return out


def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Create a sorted label -> id mapping."""
    labels = sorted(df["leaf_title"].unique().tolist())
    return {name: i for i, name in enumerate(labels)}


# ---------------------------------------------------------------------------
# Two-stage splitting
# ---------------------------------------------------------------------------

BRANCH_TO_ID = {"medical": 0, "no_medical": 1}


def prepare_two_stage_splits(
    data_dir: Path,
    val_ratio: float = 0.10,
    seed: int = 42,
    min_samples_stage1: int = 1,
    min_samples_stage2: int = 50,
    max_samples: Optional[int] = None,
    medical_ignore_folders: Tuple[str, ...] = (),
) -> dict:
    """
    Collect data and produce train/val splits for all three stages.

    Returns a dict with keys:
        stage1_train, stage1_val,
        med_train, med_val, med_labels, med_to_id,
        non_train, non_val, non_labels, non_to_id
    """
    df_all = collect_samples(data_dir, medical_ignore_folders=medical_ignore_folders)
    if len(df_all) == 0:
        raise SystemExit("No samples found.")

    df_med = df_all[df_all["branch"] == "medical"].copy().reset_index(drop=True)
    df_non = df_all[df_all["branch"] == "no_medical"].copy().reset_index(drop=True)

    # Stage 1 filtering (lenient)
    df_med_s1 = filter_min_samples(df_med, min_samples_stage1)
    df_non_s1 = filter_min_samples(df_non, min_samples_stage1)
    df_s1 = pd.concat([df_med_s1, df_non_s1], ignore_index=True)

    df_tr_s1, df_va_s1 = train_test_split(
        df_s1, test_size=val_ratio, random_state=seed, stratify=df_s1["branch"]
    )
    df_tr_s1 = df_tr_s1.reset_index(drop=True)
    df_va_s1 = df_va_s1.reset_index(drop=True)
    print(f"Stage1 splits: {len(df_tr_s1)} train, {len(df_va_s1)} val")

    # Stage 2 filtering (stricter)
    df_med_f = filter_min_samples(df_med, min_samples_stage2)
    df_non_f = filter_min_samples(df_non, min_samples_stage2)
    if max_samples is not None:
        df_med_f = cap_overrepresented_classes(df_med_f, max_samples, seed)
        df_non_f = cap_overrepresented_classes(df_non_f, max_samples, seed)

    def _split(df_branch):
        tr, va = train_test_split(
            df_branch, test_size=val_ratio, random_state=seed,
            stratify=df_branch["leaf_title"],
        )
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    df_tr_med, df_va_med = _split(df_med_f)
    df_tr_non, df_va_non = _split(df_non_f)

    med_labels = sorted(df_tr_med["leaf_title"].unique().tolist())
    non_labels = sorted(df_tr_non["leaf_title"].unique().tolist())
    med_to_id = {n: i for i, n in enumerate(med_labels)}
    non_to_id = {n: i for i, n in enumerate(non_labels)}

    print(f"Stage2 medical: {len(med_labels)} classes, {len(df_tr_med)} train, {len(df_va_med)} val")
    print(f"Stage2 no_medical: {len(non_labels)} classes, {len(df_tr_non)} train, {len(df_va_non)} val")

    return {
        "stage1_train": df_tr_s1, "stage1_val": df_va_s1,
        "med_train": df_tr_med, "med_val": df_va_med,
        "med_labels": med_labels, "med_to_id": med_to_id,
        "non_train": df_tr_non, "non_val": df_va_non,
        "non_labels": non_labels, "non_to_id": non_to_id,
    }


# ---------------------------------------------------------------------------
# Text preloading helper
# ---------------------------------------------------------------------------

def preload_texts(df: pd.DataFrame) -> Dict[str, str]:
    """Read all text files referenced in *df* into a dict keyed by path."""
    out: Dict[str, str] = {}
    for p in df["text_path"]:
        if p not in out:
            out[p] = read_text_cached(p)
    return out


# ---------------------------------------------------------------------------
# Tokenisation helper (head-tail strategy)
# ---------------------------------------------------------------------------

def tokenize_head_tail(
    text: str,
    tokenizer,
    max_len: int = 512,
    head_tokens: int = 360,
) -> dict:
    """
    Tokenize using a head-tail strategy: keep the first *head_tokens* and
    the last *(max_len - head_tokens)* tokens from a long document.
    """
    enc = tokenizer(text, truncation=False, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) <= max_len:
        attn = [1] * len(ids)
    else:
        tail_len = max_len - head_tokens
        ids = ids[:head_tokens] + ids[-tail_len:]
        attn = [1] * max_len
    return {"input_ids": ids, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """Dataset for Stage 1: binary classification (medical vs no_medical)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 512,
        head_tokens: int = 360,
        split: str = "train",
        preloaded_texts: Optional[Dict[str, str]] = None,
        header_mask_prob: float = 0.0,
        header_lines_to_mask: int = 5,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_tokens = head_tokens
        self.is_train = (split == "train")
        self.preloaded_texts = preloaded_texts
        self.header_mask_prob = header_mask_prob
        self.header_lines_to_mask = header_lines_to_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preloaded_texts is not None:
            text = self.preloaded_texts.get(row["text_path"], read_text_cached(row["text_path"]))
        else:
            text = read_text_cached(row["text_path"])

        # Header masking augmentation (training only)
        if self.is_train and self.header_mask_prob > 0 and random.random() < self.header_mask_prob:
            lines = text.split("\n")
            if len(lines) > 3:
                n_mask = random.randint(1, min(self.header_lines_to_mask, len(lines) - 1))
                lines[:n_mask] = [""] * n_mask
                text = "\n".join(lines)

        tok = tokenize_head_tail(text, self.tokenizer, self.max_len, self.head_tokens)
        y = BRANCH_TO_ID[row["branch"]]

        return {
            "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
        }


class Stage2Dataset(Dataset):
    """Dataset for Stage 2: leaf-class multi-class classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_id: Dict[str, int],
        tokenizer,
        max_len: int = 512,
        head_tokens: int = 360,
        split: str = "train",
        preloaded_texts: Optional[Dict[str, str]] = None,
        header_mask_prob: float = 0.0,
        header_lines_to_mask: int = 5,
    ):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_tokens = head_tokens
        self.is_train = (split == "train")
        self.preloaded_texts = preloaded_texts
        self.header_mask_prob = header_mask_prob
        self.header_lines_to_mask = header_lines_to_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preloaded_texts is not None:
            text = self.preloaded_texts.get(row["text_path"], read_text_cached(row["text_path"]))
        else:
            text = read_text_cached(row["text_path"])

        # Header masking augmentation (training only)
        if self.is_train and self.header_mask_prob > 0 and random.random() < self.header_mask_prob:
            lines = text.split("\n")
            if len(lines) > 3:
                n_mask = random.randint(1, min(self.header_lines_to_mask, len(lines) - 1))
                lines[:n_mask] = [""] * n_mask
                text = "\n".join(lines)

        tok = tokenize_head_tail(text, self.tokenizer, self.max_len, self.head_tokens)
        y = self.label_to_id[row["leaf_title"]]

        return {
            "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collation & padding
# ---------------------------------------------------------------------------

def collate_pad(batch, pad_id: int = 0, max_len: int = 512):
    """Pad variable-length sequences and clamp token ids."""
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    y = torch.stack([b["y"] for b in batch])

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attn = nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    if input_ids.size(1) > max_len:
        input_ids = input_ids[:, :max_len]
        attn = attn[:, :max_len]

    return {"input_ids": input_ids, "attention_mask": attn, "y": y}


# ---------------------------------------------------------------------------
# Weighted sampler & class weights
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels: np.ndarray, power: float = 0.75):
    """Create a WeightedRandomSampler from an integer label array."""
    class_counts = np.bincount(labels)
    weights = 1.0 / (class_counts.astype(np.float64) ** power)
    sample_weights = weights[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def make_class_weights(
    labels: np.ndarray,
    power: float = 0.5,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute per-class weights for the loss function."""
    counts = np.bincount(labels).astype(np.float64)
    counts = np.maximum(counts, 1)
    w = (1.0 / counts) ** power
    w = w / w.mean()
    w = np.clip(w, 0, max_weight)
    return torch.tensor(w, dtype=torch.float32)
