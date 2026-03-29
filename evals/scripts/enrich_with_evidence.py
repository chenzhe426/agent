#!/usr/bin/env python3
"""
enrich_with_evidence.py – Add gold_evidence_texts to eval run results.

Reads an eval run JSONL (from run_financebench_eval.py), looks up each sample's
gold_chunk_ids in the KB, and adds the corresponding chunk text as
gold_evidence_texts.

Usage:
    python evals/scripts/enrich_with_evidence.py \\
        --input evals/runs/run_latest.jsonl \\
        --output evals/runs/run_latest_with_evidence.jsonl

    # Or in-place:
    python evals/scripts/enrich_with_evidence.py \\
        --input evals/runs/run_latest.jsonl \\
        --output evals/runs/run_latest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db.bootstrap import init_db
from app.db import get_chunks_by_ids


def _get_gold_chunk_ids(record: dict) -> list[int]:
    """Extract gold_chunk_ids from an eval run record."""
    # Try top-level gold_chunk_ids first (from run_financebench_eval.py)
    ids = record.get("gold_chunk_ids")
    if ids:
        return [int(i) for i in ids if i is not None]

    # Fallback to retrieval.gold_chunk_ids
    retrieval = record.get("retrieval", {})
    ids = retrieval.get("gold_chunk_ids")
    if ids:
        return [int(i) for i in ids if i is not None]

    return []


def _get_chunk_text_field(chunk: dict) -> str:
    """Get the best available text field from a chunk dict."""
    return (
        chunk.get("chunk_text")
        or chunk.get("content")
        or chunk.get("search_text")
        or ""
    )


def enrich_record(record: dict, chunk_map: dict[int, dict]) -> dict:
    """Add gold_evidence_texts to a single eval run record."""
    chunk_ids = _get_gold_chunk_ids(record)
    if not chunk_ids:
        return record

    texts: list[str] = []
    for cid in chunk_ids:
        chunk = chunk_map.get(cid)
        if chunk:
            text = _get_chunk_text_field(chunk)
            if text:
                texts.append(text)

    record["gold_evidence_texts"] = texts
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich eval run results with gold_evidence_texts from KB",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input eval run JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL (can be same as input for in-place)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load eval run
    records: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"[enrich_with_evidence] Loaded {len(records)} records from {args.input}")

    # Collect all unique gold_chunk_ids
    all_ids: set[int] = set()
    for record in records:
        for cid in _get_gold_chunk_ids(record):
            all_ids.add(cid)

    print(f"[enrich_with_evidence] Total unique gold_chunk_ids to fetch: {len(all_ids)}")

    if not all_ids:
        print("[WARN] No gold_chunk_ids found in records, writing as-is")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    # Fetch chunks from KB
    init_db()
    chunks = get_chunks_by_ids(all_ids)
    print(f"[enrich_with_evidence] Fetched {len(chunks)} chunks from KB")

    chunk_map: dict[int, dict] = {c["id"]: c for c in chunks}

    # Enrich records
    enriched = []
    n_with_evidence = 0
    for record in records:
        chunk_ids = _get_gold_chunk_ids(record)
        enriched_record = enrich_record(record, chunk_map)
        has_texts = bool(enriched_record.get("gold_evidence_texts"))
        if has_texts:
            n_with_evidence += 1
        enriched.append(enriched_record)

    print(f"[enrich_with_evidence] Records with evidence texts: {n_with_evidence}/{len(records)}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in enriched:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[enrich_with_evidence] Written to: {args.output}")

    # Show sample
    for record in enriched:
        texts = record.get("gold_evidence_texts", [])
        if texts:
            print(f"\n[Sample] id={record.get('id')}, evidence_count={len(texts)}")
            print(f"  First evidence ({len(texts[0])} chars): {texts[0][:200]!r}...")
            break


if __name__ == "__main__":
    main()
