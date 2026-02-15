# Text Classification using Mamba

## Overview
This project performs text classification using the Mamba Model. It supports both:
- **Binary classification** (e.g., IMDB sentiment analysis — positive/negative)
- **Multi-class classification** from a folder-based dataset structure

## Dataset

### IMDB (Binary)
The IMDB dataset consists of 50,000 movie reviews, split evenly into 25k for training and 25k for testing. Each review is labeled as either positive or negative.

### Folder-Based (Multi-Class)
For multi-class classification, organize your data as follows:
```
data_dir/
    branch1/                    # e.g., "medical"
        class_folder1/          # e.g., "01=Cardiology" or "01_Cardiology"
            doc1.txt
            doc2.txt
        class_folder2/
            ...
    branch2/                    # e.g., "no_medical"
        ...
```
Or a flat structure (no branches):
```
data_dir/
    class_folder1/
        doc1.txt
    class_folder2/
        ...
```
Folder names can use `code=title`, `code_title`, or `name.cat` formats.

## Installation
To run the project locally, follow these steps:

1. Clone this repository:
```
git clone https://github.com/VuBacktracking/mamba-text-classification.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Binary Classification (IMDB)
```
cd mamba-text-classification
python trainer.py
```

### Multi-Class Classification (Folder-Based)
```
python multi_class_trainer.py \
    --data_dir /path/to/your/data \
    --output_dir runs/my_experiment \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-5 \
    --min_samples 10 \
    --max_samples 1500 \
    --use_weighted_sampler \
    --ignore_folders "noisy_folder1" "noisy_folder2"
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Root directory with class subfolders |
| `--output_dir` | `runs/multiclass` | Output directory for checkpoints |
| `--pretrained_model` | `state-spaces/mamba-130m` | Pretrained Mamba model |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 16 | Batch size per device |
| `--lr` | 5e-5 | Learning rate |
| `--min_samples` | 10 | Minimum samples per class (filter rare) |
| `--max_samples` | None | Cap overrepresented classes |
| `--val_ratio` | 0.10 | Validation split ratio |
| `--use_weighted_sampler` | False | Use weighted sampling for imbalance |
| `--ignore_folders` | [] | Folder patterns to skip |

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

## Dependencies
- Python 3.x
- Other dependencies listed in requirements.txt

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The IMDB dataset: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- Mamba: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)