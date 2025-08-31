from __future__ import annotations
import hashlib
from datasets import load_dataset, Dataset
import numpy as np
from typing import Dict, List, Tuple
from .config import DataCfg


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _validate_and_normalize_indices(idxs: List[int], max_count: int) -> List[int]:
    if not idxs:
        raise ValueError("data.negatives_indices must have at least 1 index (1..8).")
    if len(idxs) > max_count:
        raise ValueError(
            f"data.negatives_indices has {len(idxs)} items but only {max_count} negatives exist."
        )
    for i in idxs:
        if not (0 <= i < max_count):
            raise ValueError(
                f"Index {i} is out of bounds for {max_count} negatives (valid: 0..{max_count - 1})."
            )
    return list(idxs)


def _extract_negatives_and_scores(row) -> tuple[list[str], list[float], float | None]:
    """
    Tries to be robust across the common distillation datasets:
      - rows provide 8 text negatives as negative_1..negative_8
      - teacher scores are in one of: 'label', 'scores', 'neg_scores', etc.
      - sometimes a positive score is included (length 9); we ignore it for margin MSE
    Returns: (neg_texts[8], neg_scores[8], pos_score or None)
    """
    # 1) negatives text
    neg_texts = [row[f"negative_{i}"] for i in range(1, 9) if f"negative_{i}" in row]
    if len(neg_texts) != 8:
        # fallback: some variants might store negatives as a list
        if "negatives" in row and isinstance(row["negatives"], list):
            neg_texts = row["negatives"][:8]
        else:
            raise KeyError(
                "Could not find 8 negative texts (negative_1..negative_8 or a 'negatives' list)."
            )

    # 2) teacher scores
    pos_score = None
    scores = None
    # most common fields to check
    for field in ("label", "scores", "neg_scores"):
        if field in row:
            val = row[field]
            if isinstance(val, (list, tuple, np.ndarray)):
                scores = list(map(float, val))
                break

    if scores is None:
        raise KeyError(
            "Could not find teacher scores in row (expected one of: label, scores, neg_scores)."
        )

    # Heuristics:
    # - If length == 8: assume these are the 8 negative scores (most common)
    # - If length >= 9: assume first (or one) corresponds to the positive; we keep only 8 neg scores
    if len(scores) == 8:
        neg_scores = scores
    elif len(scores) >= 9:
        # Try to detect which element is the positive score:
        # Many datasets store [pos, neg1..neg8]. We'll assume that and drop the first.
        pos_score = scores[0]
        neg_scores = scores[1:9]
    else:
        raise ValueError(
            f"Unexpected teacher score length: {len(scores)} (expected 8 or >=9)."
        )

    if len(neg_scores) != 8:
        raise ValueError("After normalization, neg_scores must have length 8.")

    return neg_texts, neg_scores, pos_score


def load_train_dataset(cfg: DataCfg) -> Dataset:
    ds = load_dataset(cfg.train_name, split=cfg.train_split)
    if cfg.train_select_rows is not None:
        ds = ds.shuffle(seed=42).select(range(cfg.train_select_rows))

    rows = []
    for row in ds:
        neg_texts, neg_scores, _pos_score = _extract_negatives_and_scores(row)

        # Sort negatives by teacher score (ascending: easier→harder; reverse if you prefer hardest-first)
        pairs = sorted(zip(neg_texts, neg_scores), key=lambda x: x[1])
        neg_sorted = [p[0] for p in pairs]
        score_sorted = [max(float(p[1]), float(cfg.label_min)) for p in pairs]

        # Validate indices and select K
        idxs = _validate_and_normalize_indices(cfg.negatives_indices, max_count=8)
        selected_negs = [neg_sorted[i] for i in idxs]
        selected_scores = [score_sorted[i] for i in idxs]

        # Emit exactly K columns: negative_1..negative_K
        example = {
            "query_id": row.get("query_id"),
            "query": row["query"],
            "positive": row["positive"]
            if "positive" in row
            else row.get("positives", [""])[0],
            "label": selected_scores,  # length K — matches number of negative_* columns
        }
        for j, text in enumerate(selected_negs, start=1):
            example[f"negative_{j}"] = text

        rows.append(example)

    return Dataset.from_list(rows)


def load_eval_corpus(
    cfg: DataCfg,
) -> Tuple[Dict[int, str], Dict[str, str], Dict[int, List[str]]]:
    eval_ds = load_dataset(cfg.eval_name, split=cfg.eval_split)
    if cfg.eval_select_rows is not None:
        eval_ds = eval_ds.select(range(cfg.eval_select_rows))

    queries = dict(zip(eval_ds["query_id"], eval_ds["query"]))

    corpus: Dict[str, str] = {}
    for row in eval_ds:
        for pos in row["positives"]:
            corpus[md5(pos)] = pos
        for neg in row["negatives"]:
            corpus[md5(neg)] = neg

    relevant = dict(
        zip(
            eval_ds["query_id"],
            [[md5(pos) for pos in positives] for positives in eval_ds["positives"]],
        )
    )
    return queries, corpus, relevant
