"""
Two-stage (three-model) training pipeline using the Mamba model with
folder-based data loading.

    Stage 1  – binary classifier  (medical vs no_medical)
    Stage 2A – medical leaf classifier  (multi-class)
    Stage 2B – no_medical leaf classifier (multi-class)

Key features (ported from the XLM-RoBERTa two-stage pipeline):
  - Focal Loss for hard-example mining
  - ReduceLROnPlateau + cosine warmup
  - Embedding-level mixup augmentation
  - Header-masking augmentation
  - Class-balanced sampling & class weights
  - Resume training from checkpoint
  - Early stopping with patience

Usage:
    python multi_class_trainer.py                       # uses default data path
    python multi_class_trainer.py --data_dir /other/path [options]

Data directory layout (default: /media/user/Libonea AI/class):
    data_dir/
        medical/
            class_folder1/*.txt
            class_folder2/*.txt
        no_medical/
            class_folder3/*.txt
            ...
"""

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from folder_dataset import (
    BRANCH_TO_ID,
    Stage1Dataset,
    Stage2Dataset,
    collate_pad,
    make_class_weights,
    make_weighted_sampler,
    preload_texts,
    prepare_two_stage_splits,
)
from mamba.model import MambaTextClassification
from utils import compute_multiclass_metrics


# ---------------------------------------------------------------------------
# Focal Loss  (Focal Loss for Dense Object Detection, Lin et al. 2017)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing and per-class alpha weights."""

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.zeros_like(inputs)
                smooth.fill_(self.label_smoothing / (n_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            ce_loss = -(smooth * log_p).sum(dim=-1)
            p_t = (smooth * p).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_p, targets, reduction="none")
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(x, y, alpha=0.2):
    """Mixup at the embedding level."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(logits, y_a, y_b, lam, criterion):
    """Mixed loss for mixup: weighted sum of losses against both targets."""
    # Compute per-sample losses
    orig_reduction = getattr(criterion, "reduction", "mean")
    criterion.reduction = "none"
    loss_a = criterion(logits, y_a)
    loss_b = criterion(logits, y_b)
    criterion.reduction = orig_reduction
    return (lam * loss_a + (1 - lam) * loss_b).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy and macro-F1 on a validation loader."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        y = batch["y"]
        logits = model(input_ids).logits
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(y)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import accuracy_score, f1_score
    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training loop (matches the XLM-R pipeline's train_loop_improved)
# ---------------------------------------------------------------------------

def train_loop(
    stage_name: str,
    model,
    loader_tr: DataLoader,
    loader_va: DataLoader,
    class_weight,
    epochs: int,
    patience: int,
    ckpt_path: Path,
    output_dir: Path,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    grad_accum: int,
    label_smoothing: float,
    device: str,
    num_classes: int,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.5,
    use_reduce_lr: bool = True,
    reduce_lr_factor: float = 0.6,
    reduce_lr_patience: int = 3,
    min_lr: float = 1e-7,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    use_bf16: bool = False,
    resume_from: Optional[Path] = None,
    labels: Optional[List[str]] = None,
):
    """
    Custom training loop with:
      - Focal Loss or label-smoothed CE
      - Cosine LR schedule with warmup
      - ReduceLROnPlateau
      - Embedding-level mixup
      - Early stopping & checkpointing
      - Resume from checkpoint
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = math.ceil(len(loader_tr) / grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    cosine_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    reduce_lr_scheduler = None
    if use_reduce_lr:
        reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=reduce_lr_factor,
            patience=reduce_lr_patience, min_lr=min_lr,
        )

    # Build loss function
    cw = class_weight.to(device) if class_weight is not None else None
    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma, alpha=cw,
                              label_smoothing=label_smoothing)
        print(f"  Using Focal Loss (gamma={focal_gamma}, smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        print(f"  Using CrossEntropyLoss (smoothing={label_smoothing})")

    use_fp16 = (device == "cuda" and not use_bf16)
    ac_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    best_f1 = -1.0
    start_epoch = 1
    no_improve = 0
    history: List[dict] = []

    # Resume from checkpoint
    if resume_from and resume_from.exists():
        print(f"  Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        best_f1 = ckpt.get("best_f1", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best_f1={best_f1:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Starting {stage_name} | epochs={epochs} patience={patience} classes={num_classes}")
    if use_mixup:
        print(f"  Mixup: alpha={mixup_alpha}, prob={mixup_prob}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        pbar = tqdm(enumerate(loader_tr), total=len(loader_tr),
                    desc=f"[{stage_name}] {epoch}/{epochs}")
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=ac_dtype, enabled=(device == "cuda")):
                do_mixup = use_mixup and random.random() < mixup_prob

                if do_mixup:
                    emb = model.forward_embeddings(input_ids)
                    mixed, y_a, y_b, lam = mixup_data(emb, y, mixup_alpha)
                    logits = model.forward_head(mixed)
                    loss = mixup_criterion(logits, y_a, y_b, lam, criterion)
                else:
                    logits = model(input_ids).logits
                    loss = criterion(logits, y)

            loss = loss / grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                cosine_scheduler.step()

            running_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        # ── evaluate ──
        scores = evaluate(model, loader_va, device)
        avg_loss = running_loss / len(loader_tr)
        current_lr = optimizer.param_groups[-1]["lr"]

        print(f"[{stage_name}] epoch={epoch} loss={avg_loss:.4f} "
              f"val_F1={scores['macro_f1']:.4f} val_acc={scores['acc']:.4f} "
              f"lr={current_lr:.2e}")

        if reduce_lr_scheduler:
            reduce_lr_scheduler.step(scores["macro_f1"])

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "val_f1": scores["macro_f1"], "val_acc": scores["acc"],
            "lr": current_lr,
        })

        # Save history
        hist_path = output_dir / f"{stage_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # ── checkpoint ──
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            no_improve = 0
            torch.save({"model": model.state_dict(), "best_f1": best_f1,
                         "epoch": epoch, "history": history}, ckpt_path)
            print(f"[{stage_name}] New best! macro-F1: {best_f1:.4f}")
        else:
            no_improve += 1
            torch.save({"model": model.state_dict(), "best_f1": best_f1,
                         "epoch": epoch, "history": history},
                        ckpt_path.parent / f"{ckpt_path.stem}_latest.pt")
            if no_improve >= patience:
                print(f"[{stage_name}] Early stop at epoch {epoch}. "
                      f"Best F1: {best_f1:.4f}")
                break

    print(f"  History saved to {hist_path}")
    return best_f1


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def _make_loader_common(ds_tr, ds_va, y_tr, tokenizer, batch_size, cfg_ns):
    """Shared logic for building train/val DataLoaders."""
    sampler = (make_weighted_sampler(y_tr, cfg_ns.sampler_power)
               if cfg_ns.use_weighted_sampler else None)

    pad_fn = partial(collate_pad, pad_id=tokenizer.pad_token_id,
                     max_len=cfg_ns.max_len)
    kw = dict(num_workers=cfg_ns.num_workers, pin_memory=cfg_ns.pin_memory,
              collate_fn=pad_fn)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size,
                           shuffle=(sampler is None), sampler=sampler,
                           drop_last=True, **kw)
    loader_va = DataLoader(ds_va, batch_size=batch_size * 2,
                           shuffle=False, **kw)

    cw = (make_class_weights(y_tr, cfg_ns.class_weight_power,
                             cfg_ns.max_class_weight)
          if cfg_ns.use_class_weights else None)
    return loader_tr, loader_va, cw


def make_loader_stage1(df_tr, df_va, tokenizer, batch_size, cfg_ns):
    """Build DataLoaders for Stage 1 (binary)."""
    preloaded = None
    if cfg_ns.preload_text:
        import pandas as pd
        preloaded = preload_texts(pd.concat([df_tr, df_va]))

    ds_tr = Stage1Dataset(df_tr, tokenizer, cfg_ns.max_len, cfg_ns.head_tokens,
                          split="train", preloaded_texts=preloaded,
                          header_mask_prob=cfg_ns.header_mask_prob,
                          header_lines_to_mask=cfg_ns.header_lines_to_mask)
    ds_va = Stage1Dataset(df_va, tokenizer, cfg_ns.max_len, cfg_ns.head_tokens,
                          split="dev", preloaded_texts=preloaded)

    y_tr = np.array([BRANCH_TO_ID[b] for b in df_tr["branch"]], dtype=np.int64)
    return _make_loader_common(ds_tr, ds_va, y_tr, tokenizer, batch_size, cfg_ns)


def make_loader_stage2(df_tr, df_va, label_to_id, tokenizer, batch_size, cfg_ns):
    """Build DataLoaders for Stage 2 (multi-class)."""
    preloaded = None
    if cfg_ns.preload_text:
        import pandas as pd
        preloaded = preload_texts(pd.concat([df_tr, df_va]))

    ds_tr = Stage2Dataset(df_tr, label_to_id, tokenizer, cfg_ns.max_len,
                          cfg_ns.head_tokens, split="train",
                          preloaded_texts=preloaded,
                          header_mask_prob=cfg_ns.header_mask_prob,
                          header_lines_to_mask=cfg_ns.header_lines_to_mask)
    ds_va = Stage2Dataset(df_va, label_to_id, tokenizer, cfg_ns.max_len,
                          cfg_ns.head_tokens, split="dev",
                          preloaded_texts=preloaded)

    y_tr = np.array([label_to_id[x] for x in df_tr["leaf_title"]], dtype=np.int64)
    return _make_loader_common(ds_tr, ds_va, y_tr, tokenizer, batch_size, cfg_ns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Two-stage Mamba text classification")
    p.add_argument("--data_dir", type=str, default="/media/user/Libonea AI/class")
    p.add_argument("--output_dir", type=str, default="runs/twostage_mamba")
    p.add_argument("--pretrained_model", type=str, default="state-spaces/mamba-130m")
    p.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")

    # Splits
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--min_samples_stage1", type=int, default=1)
    p.add_argument("--min_samples_stage2", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=1500)

    # Token window
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--head_tokens", type=int, default=360)

    # Training
    p.add_argument("--batch_size_stage1", type=int, default=32)
    p.add_argument("--batch_size_stage2", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs_stage1", type=int, default=50)
    p.add_argument("--epochs_medical", type=int, default=60)
    p.add_argument("--epochs_no_medical", type=int, default=60)
    p.add_argument("--patience_stage1", type=int, default=3)
    p.add_argument("--patience_medical", type=int, default=8)
    p.add_argument("--patience_no_medical", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Loss
    p.add_argument("--use_focal_loss", action="store_true", default=True)
    p.add_argument("--focal_gamma", type=float, default=2.5)

    # ReduceLROnPlateau
    p.add_argument("--use_reduce_lr", action="store_true", default=True)
    p.add_argument("--reduce_lr_factor", type=float, default=0.6)
    p.add_argument("--reduce_lr_patience", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-7)

    # Sampling / class weights
    p.add_argument("--use_weighted_sampler", action="store_true", default=True)
    p.add_argument("--sampler_power", type=float, default=0.75)
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--max_class_weight", type=float, default=10.0)

    # Augmentation
    p.add_argument("--header_mask_prob", type=float, default=0.20)
    p.add_argument("--header_lines_to_mask", type=int, default=5)
    p.add_argument("--use_mixup", action="store_true", default=True)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--mixup_prob", type=float, default=0.5)

    # Dropout per stage
    p.add_argument("--dropout_stage1", type=float, default=0.15)
    p.add_argument("--dropout_medical", type=float, default=0.20)
    p.add_argument("--dropout_no_medical", type=float, default=0.15)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--preload_text", action="store_true", default=True)

    # Skip stage 1
    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--stage1_ckpt", type=str, default=None)

    # Ignore folders
    p.add_argument("--medical_ignore_folders", nargs="*", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    print(f"DEVICE: {device} | bf16: {use_bf16} | fp16: {device == 'cuda' and not use_bf16}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "label_maps").mkdir(exist_ok=True)

    # ── tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── data splits ────────────────────────────────────────────────────
    print("\nPreparing data splits...")
    splits = prepare_two_stage_splits(
        data_dir=data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_samples_stage1=args.min_samples_stage1,
        min_samples_stage2=args.min_samples_stage2,
        max_samples=args.max_samples,
        medical_ignore_folders=tuple(args.medical_ignore_folders),
    )

    # Save label maps
    for key, fname in [("med_labels", "medical_leaf_labels.json"),
                       ("non_labels", "no_medical_leaf_labels.json")]:
        with open(output_dir / "label_maps" / fname, "w", encoding="utf-8") as f:
            json.dump(splits[key], f, ensure_ascii=False, indent=2)

    # common train_loop kwargs
    common_kw = dict(
        output_dir=output_dir,
        lr=args.lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, max_grad_norm=1.0,
        grad_accum=args.grad_accum, label_smoothing=args.label_smoothing,
        device=device,
        use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
        use_reduce_lr=args.use_reduce_lr,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        min_lr=args.min_lr,
        use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob, use_bf16=use_bf16,
    )

    # ================================================================
    # STAGE 1 – binary classifier (medical vs no_medical)
    # ================================================================
    if args.skip_stage1:
        print("\n" + "=" * 70)
        print("STAGE 1: SKIPPED (using existing checkpoint)")
        print("=" * 70)
        best_stage1 = None
    else:
        print("\n" + "=" * 70)
        print("STAGE 1: Binary classifier (medical vs no_medical)")
        print("=" * 70)

        loader_tr_s1, loader_va_s1, w_s1 = make_loader_stage1(
            splits["stage1_train"], splits["stage1_val"],
            tokenizer, args.batch_size_stage1, args,
        )

        model_s1 = MambaTextClassification.from_pretrained(
            args.pretrained_model, num_classes=2, dropout=args.dropout_stage1)
        ckpt_s1 = output_dir / "checkpoints" / "stage1_best.pt"
        resume_s1 = output_dir / "checkpoints" / "stage1_best_latest.pt"

        best_stage1 = train_loop(
            stage_name="stage1_branch", model=model_s1,
            loader_tr=loader_tr_s1, loader_va=loader_va_s1,
            class_weight=w_s1,
            epochs=args.epochs_stage1, patience=args.patience_stage1,
            ckpt_path=ckpt_s1, num_classes=2,
            labels=["medical", "no_medical"],
            resume_from=resume_s1 if resume_s1.exists() else None,
            **common_kw,
        )
        del model_s1
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"\n  Stage 1 complete! Best macro-F1: {best_stage1:.4f}")

    # ================================================================
    # STAGE 2A – medical leaf classifier
    # ================================================================
    print("\n" + "=" * 70)
    print("STAGE 2A: MEDICAL leaf classifier")
    print("=" * 70)

    med_labels = splits["med_labels"]
    med_to_id = splits["med_to_id"]

    loader_tr_med, loader_va_med, w_med = make_loader_stage2(
        splits["med_train"], splits["med_val"], med_to_id,
        tokenizer, args.batch_size_stage2, args,
    )

    model_med = MambaTextClassification.from_pretrained(
        args.pretrained_model, num_classes=len(med_labels),
        dropout=args.dropout_medical)
    ckpt_med = output_dir / "checkpoints" / "stage2_medical_best.pt"
    resume_med = output_dir / "checkpoints" / "stage2_medical_best_latest.pt"

    best_med = train_loop(
        stage_name="stage2_medical", model=model_med,
        loader_tr=loader_tr_med, loader_va=loader_va_med,
        class_weight=w_med,
        epochs=args.epochs_medical, patience=args.patience_medical,
        ckpt_path=ckpt_med, num_classes=len(med_labels),
        labels=med_labels,
        resume_from=resume_med if resume_med.exists() else None,
        **common_kw,
    )
    del model_med
    if device == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # STAGE 2B – no_medical leaf classifier
    # ================================================================
    print("\n" + "=" * 70)
    print("STAGE 2B: NO_MEDICAL leaf classifier")
    print("=" * 70)

    non_labels = splits["non_labels"]
    non_to_id = splits["non_to_id"]

    loader_tr_non, loader_va_non, w_non = make_loader_stage2(
        splits["non_train"], splits["non_val"], non_to_id,
        tokenizer, args.batch_size_stage2, args,
    )

    model_non = MambaTextClassification.from_pretrained(
        args.pretrained_model, num_classes=len(non_labels),
        dropout=args.dropout_no_medical)
    ckpt_non = output_dir / "checkpoints" / "stage2_no_medical_best.pt"
    resume_non = output_dir / "checkpoints" / "stage2_no_medical_best_latest.pt"

    best_non = train_loop(
        stage_name="stage2_no_medical", model=model_non,
        loader_tr=loader_tr_non, loader_va=loader_va_non,
        class_weight=w_non,
        epochs=args.epochs_no_medical, patience=args.patience_no_medical,
        ckpt_path=ckpt_non, num_classes=len(non_labels),
        labels=non_labels,
        resume_from=resume_non if resume_non.exists() else None,
        **common_kw,
    )
    del model_non
    if device == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    if best_stage1 is not None:
        print(f"Stage1 BINARY best macro-F1:      {best_stage1:.4f}")
    else:
        print("Stage1 BINARY: SKIPPED (using existing checkpoint)")
    print(f"Stage2 MEDICAL best macro-F1:     {best_med:.4f}")
    print(f"Stage2 NO_MEDICAL best macro-F1:  {best_non:.4f}")
    print(f"Outputs: {output_dir}")

    summary = {
        "stage1_binary": {
            "best_f1": best_stage1 if best_stage1 is not None else "skipped",
            "train_samples": len(splits["stage1_train"]),
            "val_samples": len(splits["stage1_val"]),
            "skipped": args.skip_stage1,
        },
        "medical": {
            "best_f1": best_med,
            "num_classes": len(med_labels),
            "train_samples": len(splits["med_train"]),
            "val_samples": len(splits["med_val"]),
        },
        "no_medical": {
            "best_f1": best_non,
            "num_classes": len(non_labels),
            "train_samples": len(splits["non_train"]),
            "val_samples": len(splits["non_val"]),
        },
        "config": {
            "pretrained_model": args.pretrained_model,
            "batch_size_stage1": args.batch_size_stage1,
            "batch_size_stage2": args.batch_size_stage2,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "sampler_power": args.sampler_power,
            "class_weight_power": args.class_weight_power,
            "use_mixup": args.use_mixup,
            "dropout_stage1": args.dropout_stage1,
            "dropout_medical": args.dropout_medical,
            "dropout_no_medical": args.dropout_no_medical,
            "medical_ignore_folders": args.medical_ignore_folders,
        },
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved training summary to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
