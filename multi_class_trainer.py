"""
Multi-class text classification training script using the Mamba model
with folder-based data loading.

Usage:
    python multi_class_trainer.py --data_dir /path/to/data [options]

Data directory structure:
    data_dir/
        branch1/              (e.g., "medical", "no_medical")
            class_folder1/    (e.g., "01=Cardiology", "02_Neurology")
                doc1.txt
                doc2.txt
            class_folder2/
                ...
        branch2/
            ...

    Or flat structure:
    data_dir/
        class_folder1/
            doc1.txt
        class_folder2/
            ...
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments

from cfg.config import MultiClassConfig
from folder_dataset import (
    FolderTextDataset,
    make_weighted_sampler,
    prepare_splits,
)
from mamba.head import MambaClassificationHead
from mamba.model import MambaTextClassification
from mamba.trainer import MambaTrainer
from utils import compute_multiclass_metrics


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-class text classification training with Mamba"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing class folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/multiclass",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--pretrained_model", type=str, default="state-spaces/mamba-130m",
        help="Pretrained Mamba model name or path",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b",
        help="Tokenizer name or path",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum token length")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--min_samples", type=int, default=10, help="Minimum samples per class")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per class (cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--warmup_ratio", type=float, default=0.08, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use weighted random sampler")
    parser.add_argument("--sampler_power", type=float, default=0.75, help="Sampler inverse-frequency power")
    parser.add_argument(
        "--ignore_folders", nargs="*", default=[],
        help="Folder name patterns to ignore during data collection",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    return parser.parse_args()


def main():
    args = parse_args()
    from pathlib import Path

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Prepare data splits and label map
    print("\nPreparing data splits...")
    ignore_folders = tuple(args.ignore_folders)
    df_train, df_val, label_map = prepare_splits(
        data_dir=data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        ignore_folders=ignore_folders,
    )

    num_classes = len(label_map)
    id2label = {v: k for k, v in label_map.items()}
    print(f"Number of classes: {num_classes}")

    # Save label map
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved label map to {label_map_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FolderTextDataset(
        df=df_train,
        label_map=label_map,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    val_dataset = FolderTextDataset(
        df=df_val,
        label_map=label_map,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Load model with correct number of classes
    print(f"\nLoading pretrained model: {args.pretrained_model}")
    model = MambaTextClassification.from_pretrained(
        args.pretrained_model,
        num_classes=num_classes,
    )
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        label_smoothing_factor=args.label_smoothing,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    # Initialize trainer
    trainer = MambaTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_multiclass_metrics,
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / "final_model"))

    # Save id2label mapping alongside the model
    with open(output_dir / "final_model" / "id2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete. Model saved to {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
