# Multi-Class Classification Guide

This guide explains how to use the enhanced multi-class classification features in this Mamba text classification implementation.

## Overview

The codebase now supports:
- **Binary classification** (e.g., IMDB sentiment: positive/negative)
- **Multi-class classification** (e.g., AG News: World/Sports/Business/Sci-Tech)
- **Any number of classes** (2 to N classes)

## Key Features

### 1. Configurable Number of Classes

The `MambaTextClassification` model now accepts a `num_classes` parameter:

```python
from mamba.model import MambaTextClassification

# Binary classification (default)
model_binary = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m",
    num_classes=2
)

# Multi-class classification (e.g., 4 classes)
model_multi = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m",
    num_classes=4
)
```

### 2. Model Saving and Loading

Save trained models with their configuration:

```python
# After training
model.save_pretrained("./my_model")
```

This saves:
- `pytorch_model.bin`: Model weights
- `config.json`: Configuration including number of classes

Load a saved model:

```python
model = MambaTextClassification.load_pretrained_local(
    "./my_model",
    device="cuda"
)
```

### 3. Multi-Class Metrics

The `compute_metrics` function now returns:
- **Accuracy**: Overall classification accuracy
- **F1 Macro**: Unweighted average F1 across all classes
- **F1 Weighted**: Weighted average F1 (accounts for class imbalance)

### 4. Dataset Support

Two dataset wrapper classes are available:

#### ImdbDataset (Binary)
```python
from dataset import ImdbDataset
from datasets import load_dataset

imdb = load_dataset("imdb")
dataset_wrapper = ImdbDataset(imdb, tokenizer)
train_dataset = dataset_wrapper.return_train_dataset()
```

#### MultiClassDataset (N classes)
```python
from dataset import MultiClassDataset
from datasets import load_dataset

dataset = load_dataset("ag_news")  # 4 classes
dataset_wrapper = MultiClassDataset(dataset, tokenizer, num_classes=4)
train_dataset = dataset_wrapper.return_train_dataset()
```

## Complete Example: Training on AG News (4 classes)

```python
from mamba.model import MambaTextClassification
from dataset import MultiClassDataset
from utils import compute_metrics
from mamba.trainer import MambaTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments

# 1. Load dataset
dataset = load_dataset("ag_news")
num_classes = 4

# 2. Initialize model
model = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m",
    num_classes=num_classes
)
model.to("cuda")

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# 4. Prepare datasets
dataset_wrapper = MultiClassDataset(dataset, tokenizer, num_classes=num_classes)
train_dataset = dataset_wrapper.return_train_dataset()
test_dataset, eval_dataset = dataset_wrapper.return_test_dataset(eval_ratio=0.1)

# 5. Configure training
training_args = TrainingArguments(
    output_dir="mamba_agnews",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    load_best_model_at_end=True,
)

# 6. Initialize trainer
trainer = MambaTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# 8. Save model
model.save_pretrained("./mamba_agnews_final")

# 9. Make predictions
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
text = "The stock market hit new highs today."
prediction = model.predict(text, tokenizer, id2label=id2label)
print(f"Predicted class: {prediction}")  # Expected: "Business"
```

## Backward Compatibility

The changes are fully backward compatible with existing binary classification code:

```python
# Old code still works (defaults to num_classes=2)
model = MambaTextClassification.from_pretrained("state-spaces/mamba-130m")
```

## Custom Datasets

To use your own dataset, ensure it has:
1. A `text` field containing the text to classify
2. A `label` field containing integer labels (0 to num_classes-1)

```python
from datasets import Dataset

# Create custom dataset
data = {
    "text": ["text 1", "text 2", "text 3"],
    "label": [0, 1, 2]  # 3 classes
}
custom_dataset = Dataset.from_dict(data)

# Wrap it
dataset_wrapper = MultiClassDataset(custom_dataset, tokenizer, num_classes=3)
```

## Tips for Multi-Class Training

1. **Class Imbalance**: The F1 weighted metric helps evaluate performance on imbalanced datasets
2. **Learning Rate**: You may need to adjust learning rate for different numbers of classes
3. **Epochs**: More classes may require more training epochs to converge
4. **Batch Size**: Adjust based on your GPU memory and number of classes

## Comparison with Reference Code

The implementation mimics the reference code's patterns:

| Feature | Reference Code | This Implementation |
|---------|---------------|---------------------|
| Model saving | ✓ Custom save mechanism | ✓ `save_pretrained()` |
| Class config | ✓ Stored in checkpoint | ✓ Stored in config.json |
| F1 metrics | ✓ Macro & weighted | ✓ Macro & weighted |
| Multi-stage | ✓ Stage 1 + Stage 2 | ✓ Single model, extensible |
| Label maps | ✓ JSON label maps | ✓ id2label dict |

## Running the Example

A complete working example is provided in `example_multiclass.py`:

```bash
python example_multiclass.py
```

This will:
1. Download AG News dataset (4 classes)
2. Train a Mamba model for 1 epoch
3. Save the trained model
4. Load it back and make a prediction

## Troubleshooting

**Q: "No module named 'sklearn'"**  
A: Install scikit-learn: `pip install scikit-learn`

**Q: Model size increases with more classes?**  
A: Yes, the classification head size scales with num_classes, but the backbone remains the same size.

**Q: Can I use this for sequence labeling?**  
A: No, this is for text classification (document-level). For token-level classification, you'd need a different architecture.

## References

- Original Mamba paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
- Reference implementation: Medical document classification system (two-stage architecture)
