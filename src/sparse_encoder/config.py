from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class ModelCfg:
    backbone_id: str
    pooling_strategy: str = "max"
    model_name: str = "SPLADE"
    language: str = "en"
    license: str = "mit"


@dataclass
class DataCfg:
    train_name: str
    train_split: str = "train"
    eval_name: str = ""
    eval_split: str = "dev"
    train_select_rows: Optional[int] = None
    eval_select_rows: Optional[int] = None
    # NEW: any length 1..8, values in 0..7 (after sorting by teacher score)
    negatives_indices: List[int] = field(default_factory=lambda: [0, 1, 4, 7])
    label_min: float = 1.0


@dataclass
class TrainCfg:
    run_name: str = "sparse-encoder"
    output_dir: str = "models/sparse-encoder"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    fp16: bool = False
    bf16: bool = False
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "dot_mrr@10"


@dataclass
class LossCfg:
    query_regularizer_weight: float = 5e-5
    document_regularizer_weight: float = 3e-5
    margin_weight: float = 0.05  # λ_MSE (focuses on Recall)
    kl_weight: float = 1.0  # λ_KL (focuses on Precision)
    kl_temperature: float = 2.0


@dataclass
class HubCfg:
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    private: bool = True


@dataclass
class AppCfg:
    seed: int = 42
    model: ModelCfg = None  # type: ignore
    data: DataCfg = None  # type: ignore
    train: TrainCfg = None  # type: ignore
    loss: LossCfg = None  # type: ignore
    hub: HubCfg = None  # type: ignore


def load_config(path: str | Path) -> AppCfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return AppCfg(
        seed=raw.get("seed", 42),
        model=ModelCfg(**raw["model"]),
        data=DataCfg(**raw["data"]),
        train=TrainCfg(**raw["train"]),
        loss=LossCfg(**raw["loss"]),
        hub=HubCfg(**raw.get("hub", {})),
    )
