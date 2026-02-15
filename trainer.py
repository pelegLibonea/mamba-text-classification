from mamba.head import MambaClassificationHead
from mamba.model import MambaTextClassification
from dataset import ImdbDataset
from utils import preprocess_function, compute_metrics
from mamba.trainer import MambaTrainer

import os
import random
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import Trainer
from transformers import AutoTokenizer, TrainingArguments

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token, write_permission=True)

# Load dataset - default to IMDB for backward compatibility
imdb = load_dataset("imdb")

# Detect number of classes from dataset
# For IMDB: 2 classes (positive/negative)
# For other datasets, you can modify this
num_classes = len(set(imdb["train"]["label"]))
print(f"Detected {num_classes} classes in dataset")

# Load the Mamba model from a pretrained model with configurable num_classes.
model = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m", 
    num_classes=num_classes
)
model.to("cuda")

# Load the tokenizer of the Mamba model from the gpt-neox-20b model.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# Set the pad token id to the eos token id in the tokenizer.
tokenizer.pad_token_id = tokenizer.eos_token_id

imdbDataset = ImdbDataset(imdb, tokenizer)
train_dataset = imdbDataset.return_train_dataset()
test_dataset, eval_dataset = imdbDataset.return_test_dataset(eval_ratio=0.1)

# Define training arguments in the TrainingArguments class.
# More details about supported parameters can be found at: https://huggingface.co/docs/transformers/main_classes/trainer
training_args = TrainingArguments(
    output_dir="mamba_text_classification",  # Output folder name
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Number of training samples per device
    per_device_eval_batch_size=16,  # Number of evaluation samples per device
    num_train_epochs=1,  # Number of training epochs
    warmup_ratio=0.01,  # Ratio of increasing LR during warmup
    lr_scheduler_type="cosine",  # Type of scheduler to decrease LR
    report_to="none",  # "wandb" if you want to log results
    evaluation_strategy="steps",  # Determine the metric for evaluation after each step
    eval_steps=0.1,  # Number of steps between evaluation batches
    save_strategy="steps",  # Determine when to save checkpoints
    save_steps=0.1,  # Number of steps between saving checkpoints
    logging_strategy="steps",  # Determine when to log information
    logging_steps=1,  # Number of steps between logging
    push_to_hub=False,  # Changed to False by default to avoid requiring token
    load_best_model_at_end=True,  # Load the model with the best evaluation result during training
)

# Initialize the MambaTrainer class to perform the model training process.
trainer = MambaTrainer(
    model=model,  # Model to train
    train_dataset=train_dataset,  # Training data
    eval_dataset=eval_dataset,  # Evaluation data
    tokenizer=tokenizer,  # Tokenizer used to encode data
    args=training_args,  # Pre-defined training parameters
    compute_metrics=compute_metrics  # Function to calculate performance metrics for evaluation
)

# Start the training process by calling the train() function on the trainer class.
trainer.train()

# Save the model after training (similar to reference code's save mechanism)
output_dir = training_args.output_dir
model.save_pretrained(output_dir)
print(f"\nModel saved to {output_dir}")
print(f"Number of classes: {num_classes}")