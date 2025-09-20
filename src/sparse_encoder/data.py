from __future__ import annotations

import hashlib

import torch
from datasets import Dataset, load_dataset
from sentence_transformers.sparse_encoder import SparseEncoder

from .config import DataCfg

# Load teacher model once at module level
teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _extract_negatives_and_scores(row) -> tuple[list[str], float, list[float]]:
    """
    Extract negatives and raw scores from dataset.
    Dataset has: 51 scores (pos at index 0, neg 1-50), negative_1..negative_50
    """
    # Get negative texts
    neg_texts = []
    for i in range(1, 51):
        neg_key = f"negative_{i}"
        if neg_key in row:
            neg_texts.append(row[neg_key])

    # Get raw scores: pos score at index 0, neg scores at indices 1+
    all_scores = row["label"]
    pos_score = all_scores[0]
    neg_scores = all_scores[1 : len(neg_texts) + 1]

    return neg_texts, pos_score, neg_scores


def add_kl_labels(dataset: Dataset) -> Dataset:
    """
    Add KL labels by computing raw teacher scores on the fly.
    Following the exact pattern from SparseDistillKLDivLoss documentation.
    """

    def compute_kl_labels(batch):
        # Encode with teacher model
        emb_queries = teacher_model.encode(batch["query"])
        emb_positives = teacher_model.encode(batch["positive"])

        # Handle multiple negatives - collect first negative for each example

        # Compute raw similarity scores (not margins!)
        pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)

        all_scores = [pos_scores]
        for i in range(1, 9):
            emb_negatives = teacher_model.encode(batch[f"negative_{i}"])
            all_scores.append(teacher_model.similarity_pairwise(emb_queries, emb_negatives))

        # Stack scores as [pos_scores, neg_scores] for KL loss
        kl_labels = torch.stack(all_scores, dim=1)

        return {"kl_label": kl_labels}

    return dataset.map(compute_kl_labels, batched=True)


def load_train_dataset(cfg: DataCfg) -> Dataset:
    ds = load_dataset(cfg.train_name, split=cfg.train_split)
    if cfg.train_select_rows is not None:
        ds = ds.shuffle(seed=42).select(range(cfg.train_select_rows))

    # Always add KL labels using the global teacher model
    ds = add_kl_labels(ds)

    rows = []
    for row in ds:
        neg_texts, pos_score, neg_scores = _extract_negatives_and_scores(row)

        # Select which negatives to use based on config indices
        selected_negs = [neg_texts[i] for i in cfg.negatives_pick_indices]
        selected_neg_scores = [neg_scores[i] for i in cfg.negatives_pick_indices]

        # Compute margins for MarginMSE: pos_score - each neg_score
        margin_labels = [pos_score] + selected_neg_scores

        # Build example
        example = {
            # "query_id": row.get("query_id"),
            "query": row["query"],
            "positive": row["positive"],
            "label": margin_labels + row["kl_label"],  # Margins for MarginMSE loss
        }
        for j, text in enumerate(selected_negs, start=1):
            example[f"negative_{j}"] = text

        rows.append(example)

    dataset = Dataset.from_list(rows)

    print(dataset[0])

    return dataset


def load_eval_corpus(
    cfg: DataCfg,
) -> tuple[dict[int, str], dict[str, str], dict[int, list[str]]]:
    eval_ds = load_dataset(cfg.eval_name, split=cfg.eval_split)
    if cfg.eval_select_rows is not None:
        eval_ds = eval_ds.select(range(cfg.eval_select_rows))

    queries = dict(zip(eval_ds["query_id"], eval_ds["query"], strict=False))

    corpus: dict[str, str] = {}
    for row in eval_ds:
        for pos in row["positives"]:
            corpus[md5(pos)] = pos
        for neg in row["negatives"]:
            corpus[md5(neg)] = neg

    relevant = dict(
        zip(
            eval_ds["query_id"],
            [[md5(pos) for pos in positives] for positives in eval_ds["positives"]],
            strict=False,
        )
    )
    return queries, corpus, relevant
