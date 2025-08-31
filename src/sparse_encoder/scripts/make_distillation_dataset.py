#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd

NEG_COLS = [f"negative_{i}" for i in range(1, 51)]  # exact schema for triplet-50


def rrf_scores(bm25_order: List[str], ce_scores: Dict[str, float], k: int = 60) -> Dict[str, float]:
    """RRF fuse BM25 order (rank by index) with CE ranking (from logits)."""
    bm25_rank = {doc_id: i + 1 for i, doc_id in enumerate(bm25_order)}  # 1-based
    ce_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(
        sorted(ce_scores.items(), key=lambda x: x[1], reverse=True)
    )}
    big = len(bm25_order) + 1
    fused = {}
    for doc_id in bm25_order:
        rb = bm25_rank.get(doc_id, big)
        rc = ce_rank.get(doc_id, big)
        fused[doc_id] = 1.0 / (k + rb) + 1.0 / (k + rc)
    return fused


def batch_pairs_for_ce_triplet50(
    batch: List[Dict[str, Any]], use_pos: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[int, Optional[int]]]]:
    """
    Build (query, passage) pairs and backrefs.
    backrefs: (row_idx, None) => positive; (row_idx, j) => negative_j (0..49)
    """
    pairs: List[Tuple[str, str]] = []
    backrefs: List[Tuple[int, Optional[int]]] = []
    for i, row in enumerate(batch):
        q = row["query"]
        if use_pos:
            pairs.append((q, row["positive"]))
            backrefs.append((i, None))
        for j, col in enumerate(NEG_COLS):
            pairs.append((q, row[col]))
            backrefs.append((i, j))
    return pairs, backrefs


def score_with_ce_logits(model: CrossEncoder, pairs: List[Tuple[str, str]], batch_size: int) -> np.ndarray:
    """
    Return raw CE logits (no sigmoid/softmax), dtype float32, shape [len(pairs)].
    """
    scores = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="CrossEncoder scoring"):
        chunk = pairs[start : start + batch_size]
        # CrossEncoder.predict with activation_fn=None yields raw logits
        s = model.predict(chunk, convert_to_numpy=True, show_progress_bar=False)
        s = s.astype(np.float32, copy=False)
        scores.append(s)
    return np.concatenate(scores, axis=0)


def make_row_out(
    row: Dict[str, Any],
    pos_logit: Optional[float],
    neg_logits: List[float],
    rrf_k: int,
    topk_negatives: Optional[int],
    teacher_id: str,
) -> Dict[str, Any]:
    # IDs for the 50 negatives by original BM25 rank
    ids = [f"neg-{i+1}" for i in range(50)]  # 1..50
    ce_map = {doc_id: float(s) for doc_id, s in zip(ids, neg_logits)}

    # BM25 order = original column order
    bm25_order = list(ids)

    # RRF fuse (BM25 ⨁ CE logits ranking)
    fused = rrf_scores(bm25_order, ce_map, k=rrf_k)
    fused_sorted = sorted(ids, key=lambda d: fused[d], reverse=True)

    if topk_negatives is not None:
        fused_sorted = fused_sorted[:topk_negatives]

    # Resort texts + align logits
    neg_texts = [row[f"negative_{int(doc_id.split('-')[1])}"] for doc_id in fused_sorted]
    neg_logits_rrf = [ce_map[doc_id] for doc_id in fused_sorted]

    return {
        "query": row["query"],
        "positive": row["positive"],
        # negatives in RRF order (possibly truncated)
        "negatives": neg_texts,
        # logits-only, format: [pos_logit] + neg_logits_rrf
        "scores_raw": ([pos_logit] if pos_logit is not None else []) + neg_logits_rrf,
        "scores_meta": {
            "ce_model": teacher_id,
            "scores_format": "scores_raw[0] = positive logit (if present), then negatives in RRF order",
            "rrf_k": rrf_k,
            "source_dataset": "sentence-transformers/msmarco-bm25:triplet-50",
            "value_type": "logit",
        },
        # convenience (aligned to `negatives`)
        "negatives_ce_logits_rrf": neg_logits_rrf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="sentence-transformers/msmarco-bm25")
    ap.add_argument("--split", default="triplet-50")
    ap.add_argument("--teacher", default="tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--topk_negatives", type=int, default=8, help="0 or <0 = keep all 50")
    ap.add_argument("--output_dir", default="out/msmarco-triplet50-modernbert")
    ap.add_argument("--save_parquet", action="store_true")
    args = ap.parse_args()

    if args.topk_negatives is not None and args.topk_negatives <= 0:
        args.topk_negatives = None

    ds = load_dataset(args.dataset_id, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    # ensure expected columns exist
    missing = [c for c in ["query", "positive", *NEG_COLS] if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"Dataset split missing expected columns: {missing}")

    # CE with NO activation -> raw logits
    ce = CrossEncoder(args.teacher)  # activation_fn=None by default

    outputs: List[Dict[str, Any]] = []
    ROW_CHUNK = 512  # tune for memory/throughput

    for start in range(0, len(ds), ROW_CHUNK):
        batch = ds[start : start + ROW_CHUNK]
        pairs, backrefs = batch_pairs_for_ce_triplet50(batch, use_pos=True)
        logits = score_with_ce_logits(ce, pairs, batch_size=args.batch_size)

        pos_logits: Dict[int, float] = {}
        neg_logits_map: Dict[int, List[float]] = {i: [] for i in range(len(batch))}
        for s, (i, j) in zip(logits, backrefs):
            if j is None:
                pos_logits[i] = float(s)
            else:
                neg_logits_map[i].append(float(s))

        for i, row in enumerate(batch):
            out = make_row_out(
                row=row,
                pos_logit=pos_logits.get(i),
                neg_logits=neg_logits_map[i],   # len == 50, aligned with NEG_COLS
                rrf_k=args.rrf_k,
                topk_negatives=args.topk_negatives,
                teacher_id=args.teacher,
            )
            outputs.append(out)

    out_ds = Dataset.from_list(outputs)
    os.makedirs(args.output_dir, exist_ok=True)
    out_ds.save_to_disk(args.output_dir)

    if args.save_parquet:
        pd.DataFrame(outputs).to_parquet(os.path.join(args.output_dir, "distill.parquet"), index=False)

    print(f"Saved {len(out_ds)} rows to {args.output_dir}")
    if args.save_parquet:
        print(f"Parquet: {os.path.join(args.output_dir, 'distill.parquet')}")


if __name__ == "__main__":
    main()

