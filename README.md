# Text Classification using Mamba with Multi-Class Support

## Overview
This project performs text classification using the Mamba Model. It now supports both **binary classification** (e.g., IMDB sentiment analysis) and **multi-class classification** (e.g., topic classification, medical document classification).

The implementation follows best practices from the reference code for handling multi-class scenarios, including:
- Configurable number of classes
- Model saving/loading with class configuration
- F1 score metrics (macro and weighted) alongside accuracy
- Support for custom datasets

## Dataset
By default, the project uses the IMDB dataset for binary sentiment classification:
- 50,000 movie reviews split into 25k training and 25k testing
- Each review is labeled as either positive or negative

However, you can easily adapt it to any multi-class text classification task.

## Installation
To run the project locally, follow these steps:

1. Clone this repository:
```
git clone https://github.com/pelegLibonea/mamba-text-classification.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Basic Training (Binary Classification - IMDB)
```bash
cd mamba-text-classification
python trainer.py
```

### Multi-Class Classification
To use your own multi-class dataset, modify `trainer.py`:

```python
from datasets import load_dataset
from mamba.model import MambaTextClassification
from dataset import MultiClassDataset

# Load your dataset (must have 'text' and 'label' fields)
dataset = load_dataset("your_dataset_name")

# Detect number of classes
num_classes = len(set(dataset["train"]["label"]))

# Initialize model with correct number of classes
model = MambaTextClassification.from_pretrained(
    "state-spaces/mamba-130m",
    num_classes=num_classes
)

# Use MultiClassDataset wrapper
dataset_wrapper = MultiClassDataset(dataset, tokenizer, num_classes=num_classes)
train_dataset = dataset_wrapper.return_train_dataset()
test_dataset, eval_dataset = dataset_wrapper.return_test_dataset(eval_ratio=0.1)
```

### Saving and Loading Models
The model now includes save/load functionality similar to the reference code:

```python
# Save model after training
model.save_pretrained("./my_model")

# Load model later
model = MambaTextClassification.load_pretrained_local("./my_model", device="cuda")
```

### Metrics
The training now reports multiple metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Macro**: Unweighted mean F1 score across all classes
- **F1 Weighted**: Weighted mean F1 score (accounts for class imbalance)

## History of my training
| Step | Training Loss | Validation Loss | Accuracy |
|------|---------------|-----------------|----------|
| 625  | 0.020500      | 0.246246        | 0.928000 |
| 1250 | 0.671000      | 0.195849        | 0.940800 |
| 1875 | 0.596100      | 0.266093        | 0.934400 |
| 2500 | 0.016700      | 0.217099        | 0.941200 |
| 3125 | 0.000700      | 0.209536        | 0.944800 |
| 3750 | 2.680700      | 0.188751        | 0.949200 |
| 4375 | 0.015500      | 0.224948        | 0.950000 |
| 5000 | 0.002100      | 0.199092        | 0.952800 |
| 5625 | 0.013400      | 0.192042        | 0.952400 |
| 6250 | 0.152500      | 0.190083        | 0.953600 |

**Note**: You can check my model on hugging face hub in the link: https://huggingface.co/vubacktracking/mamba_text_classification

## Key Features
- ✅ **Multi-class support**: Configurable number of output classes
- ✅ **Flexible dataset handling**: Works with any text classification dataset
- ✅ **Advanced metrics**: F1 scores (macro/weighted) in addition to accuracy
- ✅ **Model persistence**: Save and load trained models with configurations
- ✅ **Backward compatible**: Still works with binary classification (IMDB)

## Dependencies
- Python 3.x
- PyTorch
- Transformers
- Mamba-SSM
- scikit-learn (for F1 scores)
- Other dependencies listed in requirements.txt

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The IMDB dataset: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- Mamba: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
- Reference implementation for multi-class training patterns