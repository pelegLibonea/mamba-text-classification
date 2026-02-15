"""
Folder-based dataset for multi-class text classification.

Expects a directory structure like:
    data_dir/
        branch1/
            class_folder1/*.txt
            class_folder2/*.txt
        branch2/
            class_folder3/*.txt
            ...

Each class folder name is parsed to extract a code and title.
Folder names can use formats like:
    "code=title", "code_title", "name.cat", or just "name"
"""

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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
            return "[EMPTY]"
        cleaned = [c if c.isprintable() or c in "\n\t\r " else " " for c in txt]
        out = "".join(cleaned).strip()
        return out if out else "[EMPTY]"
    except Exception:
        return "[EMPTY]"


def collect_samples(
    data_dir: Path,
    ignore_folders: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Walk through data_dir, collecting text samples from class subfolders.

    Supports either a flat structure (data_dir/class_folder/*.txt) or
    a branched structure (data_dir/branch/class_folder/*.txt).

    Args:
        data_dir: Root directory containing class folders (or branch folders).
        ignore_folders: Tuple of folder name patterns to skip.

    Returns:
        DataFrame with columns: text_path, branch, leaf_folder, leaf_code, leaf_title
    """
    rows = []
    ignored_count = 0

    # Detect structure: if subdirectories contain .txt files directly, it's flat
    # If subdirectories contain further subdirectories with .txt files, it's branched
    has_branches = False
    for item in data_dir.iterdir():
        if item.is_dir():
            for sub in item.iterdir():
                if sub.is_dir():
                    has_branches = True
                    break
            break

    if has_branches:
        branches = [d.name for d in data_dir.iterdir() if d.is_dir()]
    else:
        branches = ["."]  # flat structure, single pseudo-branch

    for branch in branches:
        bdir = data_dir / branch if branch != "." else data_dir
        if not bdir.exists():
            continue

        for class_folder in sorted(bdir.iterdir()):
            if not class_folder.is_dir():
                continue
            info = parse_class_folder(class_folder.name)

            # Check if folder should be ignored
            if ignore_folders:
                folder_title = info["title"].lower().strip()
                folder_name = class_folder.name.lower().strip()
                skip_folder = False
                for ignore_pattern in ignore_folders:
                    ignore_lower = ignore_pattern.lower().strip()
                    if ignore_lower in folder_title or ignore_lower in folder_name:
                        skip_folder = True
                        break
                if skip_folder:
                    skipped = len(list(class_folder.glob("*.txt")))
                    ignored_count += skipped
                    print(f"  Ignoring folder: {class_folder.name} ({skipped} samples)")
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
            f"| leaf classes: {df['leaf_title'].nunique()}"
        )
    else:
        print("No samples collected.")
    return df


def filter_min_samples(
    df: pd.DataFrame,
    min_samples: int,
) -> pd.DataFrame:
    """Filter out classes with fewer than min_samples."""
    vc = df["leaf_title"].value_counts()
    keep = vc[vc >= min_samples].index
    drop = vc[vc < min_samples]
    out = df[df["leaf_title"].isin(keep)].copy().reset_index(drop=True)

    if len(drop) > 0:
        print(f"  Dropped {len(drop)} classes with < {min_samples} samples")

    return out


def cap_overrepresented_classes(
    df: pd.DataFrame,
    max_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap overrepresented classes to max_samples."""
    vc = df["leaf_title"].value_counts()
    classes_over_cap = (vc > max_samples).sum()

    if classes_over_cap == 0:
        return df

    print(f"  Capping {classes_over_cap} classes with > {max_samples} samples")

    capped_dfs = []
    for class_name in vc.index:
        class_df = df[df["leaf_title"] == class_name]
        if len(class_df) > max_samples:
            class_df = class_df.sample(n=max_samples, random_state=seed)
        capped_dfs.append(class_df)

    out = pd.concat(capped_dfs, ignore_index=True)
    print(f"  Samples: {len(df)} -> {len(out)} (removed {len(df) - len(out)})")
    return out


def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Create a sorted label -> id mapping from the DataFrame."""
    labels = sorted(df["leaf_title"].unique().tolist())
    return {name: i for i, name in enumerate(labels)}


def prepare_splits(
    data_dir: Path,
    val_ratio: float = 0.10,
    seed: int = 42,
    min_samples: int = 10,
    max_samples: Optional[int] = None,
    ignore_folders: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Collect data from folders, filter, optionally cap, and split into train/val.

    Returns:
        (train_df, val_df, label_map)
    """
    df = collect_samples(data_dir, ignore_folders=ignore_folders)
    if len(df) == 0:
        raise SystemExit("No samples found in data directory.")

    df = filter_min_samples(df, min_samples)
    if len(df) == 0:
        raise SystemExit("No classes with enough samples after filtering.")

    if max_samples is not None:
        df = cap_overrepresented_classes(df, max_samples, seed)

    label_map = build_label_map(df)
    print(f"  Classes: {len(label_map)}")

    df_train, df_val = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df["leaf_title"]
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")
    return df_train, df_val, label_map


class FolderTextDataset(Dataset):
    """
    PyTorch Dataset that reads text files from paths in a DataFrame
    and tokenizes them for the Mamba model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_map: Dict[str, int],
        tokenizer,
        max_len: int = 512,
        preload: bool = False,
    ):
        self.df = df
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts: Optional[List[str]] = None

        if preload:
            self.texts = [read_text_cached(p) for p in self.df["text_path"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label_map[row["leaf_title"]]

        if self.texts is not None:
            text = self.texts[idx]
        else:
            text = read_text_cached(row["text_path"])

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=False,
        )

        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def make_weighted_sampler(
    df: pd.DataFrame,
    label_map: Dict[str, int],
    power: float = 0.75,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a weighted random sampler for class imbalance."""
    labels = df["leaf_title"].map(label_map).values
    class_counts = np.bincount(labels, minlength=len(label_map))
    # Inverse frequency weighting with power smoothing
    weights = 1.0 / (class_counts.astype(np.float64) ** power)
    sample_weights = weights[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
