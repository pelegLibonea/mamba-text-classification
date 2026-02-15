import json
from dataclasses import dataclass , field , asdict
from pathlib import Path
from typing import Tuple, Optional

class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple:int = 8
    tie_embeddings: bool = True

    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass
class TwoStageConfig:
    """Configuration for two-stage folder-based classification training.

    Stage 1: Binary classifier (medical vs no_medical)
    Stage 2A: Medical leaf classifier (multi-class)
    Stage 2B: No-medical leaf classifier (multi-class)
    """

    # Paths
    DATA_DIR: Path = Path("/media/user/Libonea AI/class")
    OUTDIR: Path = Path("runs/twostage_mamba")

    # Pretrained Mamba model
    PRETRAINED_MODEL: str = "state-spaces/mamba-130m"
    TOKENIZER_NAME: str = "EleutherAI/gpt-neox-20b"

    # Splits
    SEED: int = 42
    VAL_RATIO: float = 0.10

    # Filter rare classes
    MIN_SAMPLES_PER_CLASS_STAGE1: int = 1
    MIN_SAMPLES_PER_CLASS_STAGE2: int = 50

    # Cap overrepresented classes (int for fixed cap, None to disable)
    MAX_SAMPLES_PER_CLASS: Optional[int] = 1500

    # Token window
    MAX_LEN: int = 512
    HEAD_TOKENS: int = 360

    # Training – batch / accumulation
    BATCH_SIZE_STAGE1: int = 32
    BATCH_SIZE_STAGE2: int = 16
    GRAD_ACCUM: int = 4

    # Stage 1 specific
    EPOCHS_STAGE1: int = 50
    PATIENCE_STAGE1: int = 3

    # Stage 2 specific
    EPOCHS_MEDICAL: int = 60
    EPOCHS_NO_MEDICAL: int = 60
    PATIENCE_MEDICAL: int = 8
    PATIENCE_NO_MEDICAL: int = 8

    # Learning rates
    LR: float = 5e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.08
    MAX_GRAD_NORM: float = 1.0

    # DataLoader
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 4
    PIN_MEMORY: bool = True
    PRELOAD_TEXT: bool = True

    # Class imbalance
    USE_WEIGHTED_SAMPLER: bool = True
    SAMPLER_POWER: float = 0.75
    USE_CLASS_WEIGHTS: bool = True
    CLASS_WEIGHT_POWER: float = 0.5
    MAX_CLASS_WEIGHT: float = 10.0

    # Focal Loss
    USE_FOCAL_LOSS: bool = True
    FOCAL_GAMMA: float = 2.5

    # Label smoothing
    LABEL_SMOOTHING: float = 0.1

    # ReduceLROnPlateau
    USE_REDUCE_LR: bool = True
    REDUCE_LR_FACTOR: float = 0.6
    REDUCE_LR_PATIENCE: int = 3
    MIN_LR: float = 1e-7

    # Augmentation
    USE_HEADER_MASKING: bool = True
    HEADER_MASK_PROB: float = 0.20
    HEADER_LINES_TO_MASK: int = 5
    USE_WORD_DROPOUT: bool = True
    WORD_DROPOUT_PROB: float = 0.05

    # Mixup
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.2
    MIXUP_PROB: float = 0.5

    # Dropout for classification head
    DROPOUT_STAGE1: float = 0.15
    DROPOUT_MEDICAL: float = 0.20
    DROPOUT_NO_MEDICAL: float = 0.15

    # Skip stage 1 (use existing checkpoint)
    SKIP_STAGE1: bool = False
    STAGE1_CKPT: Optional[Path] = None

    # Folders to ignore in medical branch (noisy/catch-all categories)
    MEDICAL_IGNORE_FOLDERS: Tuple[str, ...] = (
        "מסמכים רפואיים",
        "רפואי שונות",
        "בדיקות במרפאה",
        "ביקורים בקופות חולים",
        "הפניות",
        "דימות",
        "אשפוזים",
        "other medical",
    )

    # Pooling strategy
    POOLING: str = "mean"

    # Precision
    USE_BF16_IF_AVAILABLE: bool = True
