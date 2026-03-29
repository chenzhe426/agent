#!/usr/bin/env python3
"""
fix_financebench_dataset.py – Convert FinanceBench JSONL to EvalSample-compatible format
with proper gold_doc_ids for scoring.

Usage:
    python -m evals.scripts.fix_financebench_dataset \
        --input evals/data/financebench_v1_subset_3docs.jsonl \
        --output evals/data/financebench_v1_subset_3docs_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_dataset(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_doc_mapping() -> dict[str, int]:
    """Build doc_name -> internal doc_id mapping."""
    from app.db.bootstrap import init_db
    from app.db import get_all_documents
    init_db()
    docs = get_all_documents()
    return {d["title"]: d["id"] for d in docs}


def convert_sample(sample: dict, doc_map: dict[str, int]) -> dict:
    """Convert FinanceBench sample to EvalSample-compatible format."""
    eval_id = sample.get("eval_id", "")
    gold_doc_name = sample.get("gold_doc_name", "")
    gold_doc_id = doc_map.get(gold_doc_name)

    gold_chunk_ids = sample.get("gold_chunk_ids") or []

    return {
        # Required top-level fields for score_eval
        "id": eval_id,
        "question": sample.get("question", ""),
        "retrieval": {
            "label_status": "labeled_doc" if gold_doc_name else "unlabeled",
            "gold_doc_ids": [gold_doc_id] if gold_doc_id is not None else [],
            "gold_chunk_ids": gold_chunk_ids,
            "hard_negative_chunk_ids": [],
        },
        "answer": {
            "gold_answer": sample.get("gold_answer", ""),
            "must_include": sample.get("must_include", []),
            "must_not_include": sample.get("must_not_include", []),
        },
        "evaluation": {
            "expected_behavior": sample.get("expected_behavior", "answer"),
            "scoring_type": "rule+llm",
        },
        "metadata": {
            "source": "financebench",
        },
        # Keep original FinanceBench fields for reference
        "gold_doc_name": gold_doc_name,
        "gold_pages": sample.get("gold_pages", []),
        "gold_evidence_texts": sample.get("gold_evidence_texts") or [],
        "gold_chunk_status": sample.get("gold_chunk_status", "unresolved"),
        "gold_chunk_match_method": sample.get("gold_chunk_match_method", ""),
        "gold_answer": sample.get("gold_answer", ""),
        "expected_behavior": sample.get("expected_behavior", "answer"),
        "must_include": sample.get("must_include", []),
        "company": sample.get("company", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FinanceBench JSONL to EvalSample format")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples = load_dataset(args.input)
    print(f"Loaded {len(samples)} samples")

    doc_map = build_doc_mapping()
    print(f"Doc mapping: {doc_map}")

    converted = []
    for s in samples:
        c = convert_sample(s, doc_map)
        converted.append(c)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Written {len(converted)} converted samples to {args.output}")


if __name__ == "__main__":
    main()
