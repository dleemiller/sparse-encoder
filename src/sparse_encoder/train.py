from __future__ import annotations
import os, random
import numpy as np
import torch
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.evaluation import (
    SparseInformationRetrievalEvaluator,
)
from .config import AppCfg
from .model import build_model, build_loss
from .data import load_train_dataset, load_eval_corpus


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: AppCfg):
    set_seed(cfg.seed)
    os.environ.setdefault("WANDB_MODE", "disabled")

    # Data
    train_ds = load_train_dataset(cfg.data)
    queries, corpus, relevant = load_eval_corpus(cfg.data)

    evaluator = SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        batch_size=cfg.train.per_device_eval_batch_size,
        corpus_chunk_size=2048,
        show_progress_bar=False,
    )

    # Model + loss
    model = build_model(cfg.model)
    loss = build_loss(model, cfg.loss)

    # Args
    args = SparseEncoderTrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        optim=cfg.train.optim,
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        eval_strategy=cfg.train.eval_strategy,
        save_strategy=cfg.train.save_strategy,
        logging_strategy=cfg.train.logging_strategy,
        save_total_limit=cfg.train.save_total_limit,
        run_name=cfg.train.run_name,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        push_to_hub=False,
    )

    # Build dynamic column list based on how many negative_* were emitted
    k = len(cfg.data.negatives_indices)
    neg_cols = [f"negative_{i}" for i in range(1, k + 1)]
    cols = ["query", "positive"] + neg_cols + ["label"]

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds.select_columns(cols),
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    model.save_pretrained("./final")
    return model
