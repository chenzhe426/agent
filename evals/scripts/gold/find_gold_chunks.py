#!/usr/bin/env python3
"""
find_gold_chunks.py – Find gold chunk IDs for evaluation samples by matching
evidence text against the indexed document_chunks table in MySQL.

Two modes:
  1. Use each sample's gold_answer as the evidence text.
  2. Use explicit --evidence "text" arguments (can be repeated).

The script:
  a) FULLTEXT searches MySQL document_chunks for candidate chunks.
  b) Computes token-overlap coverage between evidence text and each candidate's chunk_text.
  c) Returns the best-matching chunk_id(s) per evidence text.

Usage:
    # Auto: use gold_answer from each eval sample
    python evals/scripts/gold/find_gold_chunks.py \
        --dataset evals/datasets/kb_eval_seed.jsonl \
        --output evals/reports/found_gold_chunks.json

    # Explicit evidence texts
    python evals/scripts/gold/find_gold_chunks.py \
        --evidence "The company reported revenue of $100M" \
        --evidence "Revenue grew 20% year over year" \
        --output evals/reports/found_gold_chunks.json

    # Limit to specific samples
    python evals/scripts/gold/find_gold_chunks.py \
        --dataset evals/datasets/kb_eval_seed.jsonl \
        --case kb_0001 --case kb_0002 \
        --output evals/reports/found_gold_chunks.json

    # Show chunk text previews for the matched chunks
    python evals/scripts/gold/find_gold_chunks.py \
        --dataset evals/datasets/kb_eval_seed.jsonl \
        --output evals/reports/found_gold_chunks.json --show-chunks
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.dataset import load_dataset
from app.db.repositories.chunk_repository import search_chunks_fulltext, get_chunks_by_ids


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase whitespace tokenization."""
    if not text:
        return set()
    return set(text.lower().split())


def _coverage(evidence: str, chunk_text: str) -> float:
    """
    Token-overlap coverage: what fraction of evidence tokens appear in chunk_text.
    Returns 0.0–1.0.
    """
    ev_tokens = _tokenize(evidence)
    ch_tokens = _tokenize(chunk_text)
    if not ev_tokens:
        return 0.0
    overlap = len(ev_tokens & ch_tokens)
    return overlap / len(ev_tokens)


def _extract_key_sentences(text: str, max_sentences: int = 3) -> list[str]:
    """
    Split text into sentences and return up to max_sentences.
    Tries Chinese/English punctuation boundaries.
    """
    if not text:
        return []
    # Split on Chinese 。！？ or English .!? followed by space or end
    parts = re.split(r'(?<=[。！？.!?])\s+', text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned[:max_sentences]


def _best_chunks_for_evidence(
    evidence: str,
    top_k: int = 10,
    coverage_threshold: float = 0.3,
    document_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Find the best matching chunks for a single evidence text.

    1. FULLTEXT search to get candidates.
    2. Compute coverage of evidence against each candidate's chunk_text.
    3. Return top matches sorted by coverage descending.
    """
    if not evidence or not evidence.strip():
        return []

    candidates = search_chunks_fulltext(evidence, limit=top_k * 2)
    if document_ids is not None:
        candidates = [c for c in candidates if c.get("document_id") in document_ids]

    results: list[dict[str, Any]] = []
    for chunk in candidates[:top_k * 2]:
        chunk_id = chunk.get("id")
        if chunk_id is None:
            continue

        # Fetch full chunk_text for coverage computation
        full_chunks = get_chunks_by_ids([chunk_id])
        if not full_chunks:
            continue
        full = full_chunks[0]
        chunk_text = full.get("chunk_text", "") or ""

        cov = _coverage(evidence, chunk_text)
        if cov < coverage_threshold:
            continue

        results.append({
            "chunk_id": int(chunk_id),
            "document_id": chunk.get("document_id"),
            "chunk_index": chunk.get("chunk_index"),
            "title": chunk.get("title", ""),
            "section_title": chunk.get("section_title", ""),
            "coverage": round(cov, 4),
            "chunk_text_preview": (chunk_text or "")[:300],
        })

    # Sort by coverage descending
    results.sort(key=lambda x: x["coverage"], reverse=True)
    return results[:top_k]


def find_gold_chunks_for_sample(
    sample: dict[str, Any],
    coverage_threshold: float = 0.3,
    max_evidence_sentences: int = 3,
    use_gold_answer: bool = True,
    explicit_evidence: list[str] | None = None,
) -> dict[str, Any]:
    """
    Find gold chunk IDs for a single evaluation sample.

    Returns dict with:
      - sample_id
      - question
      - gold_answer (if used)
      - evidence_texts_used: list of evidence strings
      - matched_chunks: list of {chunk_id, document_id, coverage, ...}
      - gold_chunk_ids: deduplicated list of matched chunk IDs
    """
    sample_id = sample.get("id", "?")
    question = sample.get("question", {}).get("user_query", "") if isinstance(sample.get("question"), dict) else (sample.get("question") or "")
    gold_answer = sample.get("answer", {}).get("gold_answer", "") if isinstance(sample.get("answer"), dict) else ""

    # Determine evidence texts
    if explicit_evidence:
        evidence_texts = explicit_evidence
    elif use_gold_answer and gold_answer:
        evidence_texts = _extract_key_sentences(gold_answer, max_sentences=max_evidence_sentences)
    else:
        evidence_texts = []

    matched_chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[int] = set()

    for ev in evidence_texts:
        best = _best_chunks_for_evidence(
            evidence=ev,
            top_k=5,
            coverage_threshold=coverage_threshold,
        )
        for chunk in best:
            cid = chunk.get("chunk_id")
            if cid is not None and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                matched_chunks.append({**chunk, "matched_evidence": ev[:100]})

    gold_chunk_ids = sorted(seen_chunk_ids)

    return {
        "sample_id": sample_id,
        "question": question[:80],
        "gold_answer": gold_answer[:100] if gold_answer else "",
        "evidence_texts_used": evidence_texts,
        "matched_chunks": matched_chunks,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_doc_ids": sorted({
            c.get("document_id") for c in matched_chunks
            if c.get("document_id") is not None
        }),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find gold chunk IDs by matching evidence text against indexed chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        help="Path to JSONL evaluation dataset (for gold_answer extraction)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help="Filter to specific case IDs (can be repeated)",
    )
    parser.add_argument(
        "--evidence",
        type=str,
        action="append",
        default=[],
        dest="evidence_texts",
        help="Explicit evidence text strings (can be repeated; overrides --dataset)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.3,
        dest="coverage_threshold",
        help="Minimum token-overlap coverage to accept a chunk (default: 0.3)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        dest="max_sentences",
        help="Max sentences to extract from gold_answer as evidence (default: 3)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        dest="show_chunks",
        help="Include chunk_text previews in output",
    )
    args = parser.parse_args()

    # Build samples list
    if args.evidence_texts:
        # Explicit evidence mode: single pseudo-sample
        samples = [{
            "id": "explicit_evidence",
            "question": {"user_query": "(explicit evidence mode)"},
            "answer": {"gold_answer": ""},
        }]
        use_gold_answer = False
        explicit_ev = args.evidence_texts
    elif args.dataset:
        try:
            dataset_samples = load_dataset(args.dataset)
        except Exception as e:
            print(f"ERROR loading dataset: {e}", file=sys.stderr)
            sys.exit(1)
        samples = [s.to_dict() for s in dataset_samples]
        if args.cases:
            samples = [s for s in samples if s.get("id") in set(args.cases)]
            if not samples:
                print(f"ERROR: none of the specified cases found in dataset.", file=sys.stderr)
                sys.exit(1)
        use_gold_answer = True
        explicit_ev = None
    else:
        print("ERROR: must provide --dataset or --evidence", file=sys.stderr)
        sys.exit(1)

    results: list[dict[str, Any]] = []
    for sample in samples:
        result = find_gold_chunks_for_sample(
            sample=sample,
            coverage_threshold=args.coverage_threshold,
            max_evidence_sentences=args.max_sentences,
            use_gold_answer=use_gold_answer,
            explicit_evidence=explicit_ev,
        )

        # Optionally strip chunk_text previews
        if not args.show_chunks:
            for chunk in result.get("matched_chunks", []):
                chunk.pop("chunk_text_preview", None)

        results.append(result)

        # Print summary line
        sample_id = result["sample_id"]
        gold_ids = result["gold_chunk_ids"]
        n_chunks = len(result["matched_chunks"])
        ev_preview = (result["evidence_texts_used"][0][:40] if result["evidence_texts_used"] else "(none)")
        print(
            f"  [{sample_id}] "
            f"gold_chunk_ids={gold_ids} "
            f"({n_chunks} candidates) "
            f"ev={ev_preview}..."
        )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "coverage_threshold": args.coverage_threshold,
            "max_sentences": args.max_sentences,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[find_gold_chunks] Written to: {output_path}")


if __name__ == "__main__":
    main()
