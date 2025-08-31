#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# Exact schema for the 'triplet-50' config
NEG_COLS = [f"negative_{i}" for i in range(1, 51)]


def rrf_scores(
    bm25_order: List[str], ce_scores: Dict[str, float], k: int = 60
) -> Dict[str, float]:
    """RRF fuse BM25 order (rank by column index) with CE logits ranking."""
    bm25_rank = {doc_id: i + 1 for i, doc_id in enumerate(bm25_order)}  # 1-based
    ce_rank = {
        doc_id: i + 1
        for i, (doc_id, _) in enumerate(
            sorted(ce_scores.items(), key=lambda x: x[1], reverse=True)
        )
    }
    fused = {}
    big = len(bm25_order) + 1
    for doc_id in bm25_order:
        rb = bm25_rank.get(doc_id, big)
        rc = ce_rank.get(doc_id, big)
        fused[doc_id] = 1.0 / (k + rb) + 1.0 / (k + rc)
    return fused


def build_pairs_for_rows(
    batch_ds,
) -> Tuple[List[Tuple[str, str]], List[Tuple[int, Optional[int]]]]:
    """
    Build CE (query, passage) pairs for a batch of FULL rows.
    backrefs: (row_idx, None) => positive; (row_idx, j) => negative_j (0..49)
    """
    pairs, backrefs = [], []
    n = len(batch_ds)
    for i in range(n):
        row = batch_ds[i]
        q = row["query"]
        # positive
        pairs.append((q, row["positive"]))
        backrefs.append((i, None))
        # 50 negatives in BM25 order
        for j, col in enumerate(NEG_COLS):
            pairs.append((q, row[col]))
            backrefs.append((i, j))
    return pairs, backrefs


def make_row_out(
    row: Dict[str, Any],
    pos_logit: float,
    neg_logits: List[float],
    rrf_k: int,
    topk_negatives: Optional[int],
    teacher_id: str,
) -> Dict[str, Any]:
    ids = [f"neg-{i + 1}" for i in range(50)]  # 1..50
    ce_map = {doc_id: float(s) for doc_id, s in zip(ids, neg_logits)}
    bm25_order = list(ids)

    fused = rrf_scores(bm25_order, ce_map, k=rrf_k)
    fused_sorted = sorted(ids, key=lambda d: fused[d], reverse=True)
    if topk_negatives is not None:
        fused_sorted = fused_sorted[:topk_negatives]

    neg_texts = [
        row[f"negative_{int(doc_id.split('-')[1])}"] for doc_id in fused_sorted
    ]
    neg_logits_rrf = [ce_map[doc_id] for doc_id in fused_sorted]

    return {
        "query": row["query"],
        "positive": row["positive"],
        "negatives": neg_texts,
        "scores_raw": [float(pos_logit)] + neg_logits_rrf,  # logits only
        "scores_meta": {
            "ce_model": teacher_id,
            "scores_format": "scores_raw[0] = positive logit, then negatives in RRF order",
            "rrf_k": rrf_k,
            "source_dataset": "sentence-transformers/msmarco-bm25:triplet-50",
            "value_type": "logit",
        },
        "negatives_ce_logits_rrf": neg_logits_rrf,
    }


def write_parquet_shards(
    rows_iter, shard_size: int, out_dir: str, basename: str = "msmarco"
):
    os.makedirs(out_dir, exist_ok=True)
    buf: List[Dict[str, Any]] = []
    shard_idx = 0
    for row in rows_iter:
        buf.append(row)
        if len(buf) >= shard_size:
            shard_idx += 1
            path = os.path.join(out_dir, f"{basename}-{shard_idx:05d}.parquet")
            pd.DataFrame(buf).to_parquet(path, index=False)
            buf.clear()
    if buf:
        shard_idx += 1
        path = os.path.join(out_dir, f"{basename}-{shard_idx:05d}.parquet")
        pd.DataFrame(buf).to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="sentence-transformers/msmarco-bm25")
    ap.add_argument(
        "--config", default="triplet-50", help="HF dataset config (e.g., triplet-50)"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--teacher", default="tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss"
    )
    ap.add_argument(
        "--row_batch_size",
        type=int,
        default=32,
        help="Rows per batch. Inferences per batch = 51 × row_batch_size.",
    )
    ap.add_argument(
        "--shard_size",
        type=int,
        default=1024 * 32,
        help="Rows per parquet shard (default 32k).",
    )
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument(
        "--topk_negatives", type=int, default=8, help="0 or <0 keeps all 50 after RRF"
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--output_dir", default="out/msmarco-triplet50-modernbert-parquet")
    args = ap.parse_args()

    # Interpret topk=0 or <0 as "keep all 50"
    topk = (
        None
        if args.topk_negatives is not None and args.topk_negatives <= 0
        else args.topk_negatives
    )

    # Perf toggles for Ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load dataset (strings split)
    ds: Dataset = load_dataset(args.dataset_id, args.config, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    missing = [c for c in ["query", "positive", *NEG_COLS] if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"Dataset missing expected columns: {missing}")

    # Load CE teacher with NO extra truncation beyond the model's own max context.
    ce = CrossEncoder(args.teacher, device="cuda")  # no manual max_length clamp
    # Align to model's native max context length so we don't add our own shorter cap.
    model_max = getattr(ce.model.config, "max_position_embeddings", None)
    if model_max is None or (isinstance(model_max, int) and model_max <= 0):
        model_max = ce.tokenizer.model_max_length  # last resort from tokenizer
    # CrossEncoder stores this on `self.max_length`; update it post-init.
    if isinstance(model_max, int) and model_max > 0:
        ce.max_length = int(model_max)
        ce.tokenizer.model_max_length = int(model_max)

    # Use FP16 inference; if you prefer BF16, switch dtype in autocast below.
    ce.model.half()

    total_rows = len(ds)
    row_bs = args.row_batch_size
    print(
        f"Total rows: {total_rows} | row_batch_size: {row_bs} | shard_size: {args.shard_size}"
    )

    def generate_rows():
        for start in tqdm(range(0, total_rows, row_bs)):
            end = min(total_rows, start + row_bs)
            batch_ds = ds.select(range(start, end))  # FULL rows only

            # 1) Build all pairs for THIS ROW BATCH (exactly 51 × n_rows)
            pairs, backrefs = build_pairs_for_rows(batch_ds)
            assert len(pairs) == 51 * len(batch_ds), (
                "Row batch was split; pairs count mismatch."
            )

            # 2) Single predict call that covers the ENTIRE row-batch (no partials)
            #    If this OOMs, lower --row_batch_size (by design, no try/except).
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", dtype=torch.float16),
            ):
                logits = ce.predict(
                    pairs,
                    batch_size=len(pairs),  # EXACTLY one internal batch = one row-batch
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32, copy=False)

            # 3) Split logits back into per-row pos / negs
            pos_logits: Dict[int, float] = {}
            neg_logits_map: Dict[int, List[float]] = {
                i: [] for i in range(len(batch_ds))
            }
            for s, (i, j) in zip(logits, backrefs):
                if j is None:
                    pos_logits[i] = float(s)
                else:
                    neg_logits_map[i].append(float(s))
            # Sanity: each row must have exactly 50 neg logits
            for i in range(len(batch_ds)):
                assert len(neg_logits_map[i]) == 50, (
                    "Row broken during predict; neg logits missing."
                )

            # 4) Yield output rows
            for i in range(len(batch_ds)):
                row = batch_ds[i]
                yield make_row_out(
                    row=row,
                    pos_logit=pos_logits[i],
                    neg_logits=neg_logits_map[i],  # length 50
                    rrf_k=args.rrf_k,
                    topk_negatives=topk,
                    teacher_id=args.teacher,
                )

    # 5) Rolling parquet shards
    write_parquet_shards(
        generate_rows(),
        shard_size=args.shard_size,
        out_dir=args.output_dir,
        basename="msmarco",
    )

    print(f"Done. Parquet shards written to: {args.output_dir}")


if __name__ == "__main__":
    main()
