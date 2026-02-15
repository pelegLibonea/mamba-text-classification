"""
Example script demonstrating multi-class text classification with Mamba.
This example uses the AG News dataset (4 classes: World, Sports, Business, Sci/Tech)
"""

from mamba.model import MambaTextClassification
from dataset import MultiClassDataset
from utils import compute_metrics
from mamba.trainer import MambaTrainer

import os
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments

# Configuration
BASE_MODEL_NAME = "state-spaces/mamba-130m"

# Load AG News dataset (4-class news topic classification)
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")

# Detect number of classes (efficient method)
try:
    # Try to get from ClassLabel feature (most efficient)
    if hasattr(dataset["train"].features["label"], 'num_classes'):
        num_classes = dataset["train"].features["label"].num_classes
    else:
        num_classes = len(set(dataset["train"]["label"]))
except (AttributeError, KeyError):
    num_classes = len(set(dataset["train"]["label"]))

print(f"Number of classes: {num_classes}")
print(f"Classes: World (0), Sports (1), Business (2), Sci/Tech (3)")

# Load the Mamba model with correct number of classes
print("\nInitializing Mamba model...")
model = MambaTextClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_classes=num_classes
)
model.to("cuda")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare datasets
print("\nPreparing datasets...")
dataset_wrapper = MultiClassDataset(dataset, tokenizer, num_classes=num_classes)
train_dataset = dataset_wrapper.return_train_dataset()
test_dataset, eval_dataset = dataset_wrapper.return_test_dataset(eval_ratio=0.1)

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="mamba_multiclass_agnews",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # Use more epochs for real training
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    logging_strategy="steps",
    logging_steps=100,
    push_to_hub=False,
    load_best_model_at_end=True,
)

# Initialize trainer
print("\nInitializing trainer...")
trainer = MambaTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics
)

# Train
print("\nStarting training...")
trainer.train()

# Save model
output_dir = training_args.output_dir
model.save_pretrained(output_dir, base_model_name=BASE_MODEL_NAME)
print(f"\nModel saved to {output_dir}")

# Test loading
print("\nTesting model loading...")
loaded_model = MambaTextClassification.load_pretrained_local(output_dir, device="cuda")
print("Model loaded successfully!")

# Example prediction
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
test_text = "The stock market reached new highs today as tech companies reported strong earnings."
prediction = loaded_model.predict(test_text, tokenizer, id2label=id2label)
print(f"\nExample prediction:")
print(f"Text: {test_text}")
print(f"Predicted class: {prediction}")
