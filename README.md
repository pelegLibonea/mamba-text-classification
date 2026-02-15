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

### Two-Stage Classification (Folder-Based)

The two-stage pipeline trains three models:
1. **Stage 1** – Binary classifier (medical vs no\_medical)
2. **Stage 2A** – Medical leaf classifier (multi-class)
3. **Stage 2B** – No-medical leaf classifier (multi-class)

Features: Focal Loss, ReduceLROnPlateau + cosine warmup, embedding-level
mixup, header-masking augmentation, class-balanced sampling & class weights,
per-stage dropout, early stopping, and resume-from-checkpoint.

```
python multi_class_trainer.py \
    --output_dir runs/twostage_mamba \
    --epochs_stage1 50 \
    --epochs_medical 60 \
    --epochs_no_medical 60 \
    --batch_size_stage2 16 \
    --lr 5e-5 \
    --min_samples_stage2 50 \
    --max_samples 1500 \
    --medical_ignore_folders "noisy_folder1" "noisy_folder2"
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `/media/user/Libonea AI/class` | Root directory with `medical/` and `no_medical/` subfolders |
| `--output_dir` | `runs/twostage_mamba` | Output directory for checkpoints |
| `--pretrained_model` | `state-spaces/mamba-130m` | Pretrained Mamba model |
| `--epochs_stage1` | 50 | Epochs for Stage 1 (binary) |
| `--epochs_medical` | 60 | Epochs for Stage 2A (medical) |
| `--epochs_no_medical` | 60 | Epochs for Stage 2B (no\_medical) |
| `--batch_size_stage1` | 32 | Stage 1 batch size |
| `--batch_size_stage2` | 16 | Stage 2 batch size |
| `--lr` | 5e-5 | Learning rate |
| `--min_samples_stage1` | 1 | Min samples per class for Stage 1 |
| `--min_samples_stage2` | 50 | Min samples per class for Stage 2 |
| `--max_samples` | 1500 | Cap overrepresented classes |
| `--use_focal_loss` | True | Use Focal Loss for hard-example mining |
| `--focal_gamma` | 2.5 | Focal Loss gamma |
| `--use_reduce_lr` | True | Use ReduceLROnPlateau scheduler |
| `--use_weighted_sampler` | True | Use weighted sampling for imbalance |
| `--use_class_weights` | True | Use class weights in loss |
| `--use_mixup` | True | Enable embedding-level mixup augmentation |
| `--dropout_stage1` | 0.15 | Dropout for Stage 1 classifier head |
| `--dropout_medical` | 0.20 | Dropout for Stage 2A classifier head |
| `--dropout_no_medical` | 0.15 | Dropout for Stage 2B classifier head |
| `--skip_stage1` | False | Skip Stage 1 (reuse checkpoint) |
| `--medical_ignore_folders` | [] | Medical folder patterns to skip |

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