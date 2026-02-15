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
class MultiClassConfig:
    """Configuration for multi-class folder-based classification training."""

    # Paths
    DATA_DIR: Path = Path("data")
    OUTDIR: Path = Path("runs/multiclass")

    # Splits
    SEED: int = 42
    VAL_RATIO: float = 0.10

    # Filter rare classes
    MIN_SAMPLES_PER_CLASS: int = 10

    # Cap overrepresented classes ("auto" uses 2nd highest count, int for fixed cap, None to disable)
    MAX_SAMPLES_PER_CLASS: Optional[int] = None

    # Token window
    MAX_LEN: int = 512

    # Training
    BATCH_SIZE: int = 16
    GRAD_ACCUM: int = 4
    EPOCHS: int = 50
    PATIENCE: int = 5
    LR: float = 5e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.08
    MAX_GRAD_NORM: float = 1.0

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    # Label smoothing
    LABEL_SMOOTHING: float = 0.1

    # Weighted sampling for class imbalance
    USE_WEIGHTED_SAMPLER: bool = True
    SAMPLER_POWER: float = 0.75

    # Folders to ignore (noisy/catch-all categories)
    IGNORE_FOLDERS: Tuple[str, ...] = ()

    # Pretrained model
    PRETRAINED_MODEL: str = "state-spaces/mamba-130m"
    TOKENIZER_NAME: str = "EleutherAI/gpt-neox-20b"
