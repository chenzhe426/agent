#!/usr/bin/env python3
"""
apply_gold.py – Apply gold label annotations back into the dataset JSONL.

Supports incremental annotation: update one or more cases at a time.
Backs up the original file before overwriting.

Usage:
    # Chunk-level labeling
    python evals/scripts/apply_gold.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --case kb_0001 \\
        --gold-chunk 1 --gold-chunk 10 --gold-chunk 2 \\
        --label-status labeled_chunk

    # Doc-level labeling
    python evals/scripts/apply_gold.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --case kb_0002 \\
        --gold-doc 1 --gold-doc 3 \\
        --label-status labeled_doc

    # Mark as unanswerable (gold may be empty)
    python evals/scripts/apply_gold.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --case kb_0003 \\
        --label-status unanswerable

    # Clear gold and reset to unlabeled
    python evals/scripts/apply_gold.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --case kb_0001 \\
        --label-status unlabeled --clear-gold

    # Dry run (no backup, no write)
    python evals/scripts/apply_gold.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --case kb_0001 --gold-chunk 1 \\
        --label-status labeled_chunk \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def save_samples(path: Path, samples: list[dict[str, Any]], backup: bool = True) -> None:
    if backup:
        bak_path = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, bak_path)
        print(f"[apply_gold] Backed up original → {bak_path}")

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def apply_gold(
    samples: list[dict[str, Any]],
    case_id: str,
    label_status: str | None,
    gold_chunks: list[int],
    gold_docs: list[int],
    clear_gold: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Apply gold labels to the sample matching case_id.
    Returns (updated_samples, old_sample or None).
    """
    target_idx = None
    old_sample = None
    for i, s in enumerate(samples):
        if s.get("id") == case_id:
            target_idx = i
            old_sample = dict(s)
            break

    if target_idx is None:
        return samples, None

    sample = samples[target_idx]

    # Determine new label_status
    new_status = label_status
    if new_status is None:
        new_status = sample.get("retrieval", {}).get("label_status", "unlabeled")

    # Validate combination
    if new_status in {"labeled_chunk", "labeled_doc"}:
        effective_chunks = gold_chunks if gold_chunks else sample.get("retrieval", {}).get("gold_chunk_ids", [])
        effective_docs = gold_docs if gold_docs else sample.get("retrieval", {}).get("gold_doc_ids", [])
        if not effective_chunks and not effective_docs and not clear_gold:
            raise ValueError(
                f"label_status is '{new_status}' but gold_chunk_ids and gold_doc_ids "
                f"are both empty. Provide --gold-chunk / --gold-doc, or use --clear-gold."
            )

    # Apply
    if "retrieval" not in sample:
        sample["retrieval"] = {}

    sample["retrieval"]["label_status"] = new_status

    if clear_gold:
        sample["retrieval"]["gold_chunk_ids"] = []
        sample["retrieval"]["gold_doc_ids"] = []
    else:
        if gold_chunks:
            sample["retrieval"]["gold_chunk_ids"] = gold_chunks
        if gold_docs:
            sample["retrieval"]["gold_doc_ids"] = gold_docs

    # Normalize empty lists to [] (JSON default)
    if not sample["retrieval"].get("gold_chunk_ids"):
        sample["retrieval"]["gold_chunk_ids"] = []
    if not sample["retrieval"].get("gold_doc_ids"):
        sample["retrieval"]["gold_doc_ids"] = []

    samples[target_idx] = sample
    return samples, old_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply gold annotations to dataset JSONL")
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--case", required=True,
        help="Case ID to annotate",
    )
    parser.add_argument(
        "--label-status",
        choices={"unlabeled", "labeled_doc", "labeled_chunk", "unanswerable"},
        help="New label_status value",
    )
    parser.add_argument(
        "--gold-chunk", type=int, action="append", default=[],
        dest="gold_chunks",
        help="gold_chunk_ids entries (can be repeated)",
    )
    parser.add_argument(
        "--gold-doc", type=int, action="append", default=[],
        dest="gold_docs",
        help="gold_doc_ids entries (can be repeated)",
    )
    parser.add_argument(
        "--clear-gold",
        action="store_true",
        help="Clear all gold_chunk_ids and gold_doc_ids before applying",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backup file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(dataset_path)
    sample_ids = [s.get("id") for s in samples]
    if args.case not in sample_ids:
        print(f"ERROR: case '{args.case}' not found in dataset.", file=sys.stderr)
        print(f"  Available IDs: {sample_ids}", file=sys.stderr)
        sys.exit(1)

    try:
        updated, old = apply_gold(
            samples=samples,
            case_id=args.case,
            label_status=args.label_status,
            gold_chunks=args.gold_chunks,
            gold_docs=args.gold_docs,
            clear_gold=args.clear_gold,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Show diff
    new_sample = next(s for s in updated if s["id"] == args.case)
    print(f"[apply_gold] Case: {args.case}")
    print(f"  label_status : {old.get('retrieval', {}).get('label_status', '?')} → {new_sample['retrieval']['label_status']}")
    print(f"  gold_chunk_ids: {old.get('retrieval', {}).get('gold_chunk_ids', [])} → {new_sample['retrieval']['gold_chunk_ids']}")
    print(f"  gold_doc_ids  : {old.get('retrieval', {}).get('gold_doc_ids', [])} → {new_sample['retrieval']['gold_doc_ids']}")

    if args.dry_run:
        print("\n[apply_gold] Dry run – no file written.")
        return

    save_samples(dataset_path, updated, backup=not args.no_backup)
    print(f"[apply_gold] Done – {dataset_path}")


if __name__ == "__main__":
    main()
