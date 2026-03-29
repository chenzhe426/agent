#!/usr/bin/env python3
"""
export_sft.py – Export evaluation samples to SFT-ready messages format.

Two modes:
  1. --from-evidence  (default for FinanceBench-style data)
     Uses gold_evidence_texts + gold_answer to build messages:
       user:  Question + Evidence
       assistant: gold_answer

  2. --from-supervision  (for datasets with pre-annotated sft_messages)
     Uses SupervisionBlock.sft_messages_with_context / sft_messages_no_context

Usage:
    # From evidence (most common for RAG SFT)
    python evals/scripts/export_sft.py \\
        --dataset evals/data/financebench_v1_subset_3docs_eval.jsonl \\
        --output evals/datasets/kb_eval_sft.jsonl \\
        --from-evidence

    # From pre-annotated supervision
    python evals/scripts/export_sft.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --output evals/datasets/kb_eval_sft.jsonl \\
        --from-supervision

Output format (one JSON per line):
    {
        "id": str,
        "messages": [{"role": "user"|"assistant"|"system", "content": str}, ...],
        "gold_answer": str,       # reference answer
        "task_type": str,
        "difficulty": str,
        "tags": list[str],
        "evidence_texts": list[str],  # evidence chunks (from-evidence mode only)
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.dataset import load_dataset, EvalSample


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful financial analysis assistant. "
    "Answer the user's question based strictly on the provided evidence. "
    "If the evidence does not contain sufficient information to answer the question, "
    "say so rather than speculating."
)

_FINANCE_SYSTEM_PROMPT = (
    "You are a professional financial analyst assistant. "
    "Answer the question based ONLY on the evidence provided below. "
    "Cite specific numbers and facts from the evidence when available. "
    "Do not make up information not present in the evidence."
)


# ---------------------------------------------------------------------------
# Evidence-based SFT builder
# ---------------------------------------------------------------------------

def _build_evidence_prompt(question: str, evidence_texts: list[str]) -> str:
    """Combine question + evidence into a single user prompt."""
    evidence_section = "\n\n---\n\n".join(f"[Evidence {i+1}]\n{text.strip()}" for i, text in enumerate(evidence_texts) if text.strip())
    return (
        f"Question: {question.strip()}\n\n"
        f"Evidence:\n{evidence_section}\n\n"
        f"Please answer the question based on the evidence above."
    )


def _sample_evidence(
    evidence_texts: list[str],
    max_chars: int = 3000,
    max_chunks: int = 5,
) -> list[str]:
    """Select a subset of evidence chunks that fit within max_chars."""
    if not evidence_texts:
        return []

    selected: list[str] = []
    total_chars = 0

    for text in evidence_texts[:max_chunks]:
        text = text.strip()
        if not text:
            continue
        if total_chars + len(text) + 50 > max_chars:
            # If first chunk already too big, truncate it
            if not selected:
                selected.append(text[:max_chars])
            break
        selected.append(text)
        total_chars += len(text) + 50

    return selected


def _evidence_to_messages(
    sample: dict[str, Any],
    system_prompt: str,
    max_evidence_chars: int = 3000,
) -> list[dict[str, str]]:
    """Build SFT messages from question + evidence + gold_answer (from-evidence mode)."""
    question = sample.get("question", "") or sample.get("user_query", "")
    gold_answer = sample.get("gold_answer") or sample.get("answer", {}).get("gold_answer", "")

    # Collect evidence texts from multiple possible fields
    evidence_texts: list[str] = []

    # Field 1: gold_evidence_texts (FinanceBench style)
    if "gold_evidence_texts" in sample:
        evidence_texts = list(sample["gold_evidence_texts"] or [])

    # Field 2: context.gold_context_blocks (structured eval format)
    elif "context" in sample and sample["context"].get("gold_context_blocks"):
        evidence_texts = list(sample["context"].get("gold_context_blocks", []))

    # Field 3: top-level gold_context_blocks
    elif "gold_context_blocks" in sample:
        evidence_texts = list(sample["gold_context_blocks"] or [])

    # Field 4: retrieved_gold_chunks (if already fetched)
    if "retrieved_gold_chunks" in sample:
        for chunk in sample["retrieved_gold_chunks"]:
            text = chunk.get("chunk_text") or chunk.get("search_text") or ""
            if text and text not in evidence_texts:
                evidence_texts.append(text)

    if not evidence_texts:
        return []

    # Sample evidence to fit context window
    evidence_texts = _sample_evidence(evidence_texts, max_chars=max_evidence_chars)

    user_content = _build_evidence_prompt(question, evidence_texts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": gold_answer.strip() if gold_answer else ""},
    ]
    return messages


def export_from_evidence(
    samples: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_evidence_chars: int = 3000,
) -> list[dict[str, Any]]:
    """Export from raw dict samples (FinanceBench JSONL format).

    Works on the raw JSON dicts from FinanceBench-style datasets
    that have gold_evidence_texts + gold_answer directly in each record.
    """
    system_prompt = system_prompt or _FINANCE_SYSTEM_PROMPT
    results = []

    for sample in samples:
        messages = _evidence_to_messages(
            sample,
            system_prompt=system_prompt,
            max_evidence_chars=max_evidence_chars,
        )
        if not messages:
            continue

        gold_answer = (
            sample.get("gold_answer")
            or sample.get("answer", {}).get("gold_answer", "")
            or ""
        )

        results.append({
            "id": sample.get("id") or sample.get("eval_id", ""),
            "messages": messages,
            "gold_answer": gold_answer,
            "task_type": sample.get("task_type", "factoid"),
            "difficulty": sample.get("difficulty", sample.get("metadata", {}).get("difficulty", "easy")),
            "tags": sample.get("tags", sample.get("metadata", {}).get("tags", [])),
            "evidence_texts": [
                msg["content"]
                for msg in messages
                if msg["role"] == "user"
            ],
            "question": sample.get("question", ""),
            "company": sample.get("metadata", {}).get("company", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Supervision-based SFT builder (pre-annotated sft_messages)
# ---------------------------------------------------------------------------

def export_from_supervision(
    samples: list[EvalSample],
    use_context: bool = True,
) -> list[dict[str, Any]]:
    """Export from EvalSample dataclasses using pre-annotated SupervisionBlock.

    Uses sft_messages_with_context or sft_messages_no_context
    from the SupervisionBlock.
    """
    results = []

    for sample in samples:
        supervision = sample.supervision
        if use_context and supervision.sft_messages_with_context:
            messages = supervision.sft_messages_with_context
        elif supervision.sft_messages_no_context:
            messages = supervision.sft_messages_no_context
        else:
            continue  # No SFT messages available for this sample

        results.append({
            "id": sample.id,
            "messages": messages,
            "gold_answer": sample.answer.gold_answer,
            "task_type": sample.task_type,
            "difficulty": sample.difficulty,
            "tags": sample.tags,
        })

    return results


# ---------------------------------------------------------------------------
# Chunk-fetching fallback (when gold_evidence_texts not available)
# ---------------------------------------------------------------------------

def fetch_gold_chunks_from_kb(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fetch actual chunk text from KB using gold_chunk_ids.

    Used as fallback when gold_evidence_texts is not present in the dataset.
    Requires the KB to be populated with indexed documents.
    """
    try:
        from app.db import get_chunks_by_ids
    except Exception as e:
        print(f"[WARN] Cannot import KB modules: {e}", file=sys.stderr)
        return samples

    enriched = []
    for sample in samples:
        gold_chunk_ids = (
            sample.get("gold_chunk_ids")
            or sample.get("retrieval", {}).get("gold_chunk_ids", [])
        )
        if not gold_chunk_ids:
            enriched.append(sample)
            continue

        try:
            chunks = get_chunks_by_ids(gold_chunk_ids)
            if chunks:
                sample["retrieved_gold_chunks"] = [
                    {
                        "chunk_id": c.get("id"),
                        "chunk_text": c.get("chunk_text", ""),
                        "search_text": c.get("search_text", ""),
                        "section_title": c.get("section_title", ""),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                    }
                    for c in chunks
                ]
        except Exception as e:
            print(f"[WARN] Failed to fetch chunks for sample {sample.get('id')}: {e}", file=sys.stderr)

        enriched.append(sample)

    return enriched


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export evaluation dataset to SFT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / Output
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the evaluation JSONL dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output SFT JSONL file",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--from-evidence",
        action="store_true",
        help="Build SFT messages from question + gold_evidence_texts + gold_answer "
             "(FinanceBench / RAG SFT mode)",
    )
    mode_group.add_argument(
        "--from-supervision",
        action="store_true",
        help="Use pre-annotated sft_messages from SupervisionBlock "
             "(requires EvalSample format with supervision field)",
    )

    # Options for --from-evidence
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt (default: finance analyst prompt)",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=3000,
        help="Max total characters for evidence in context (default: 3000)",
    )
    parser.add_argument(
        "--fetch-chunks",
        action="store_true",
        help="Fallback: fetch chunk text from KB using gold_chunk_ids "
             "when gold_evidence_texts is not available",
    )

    # Options for --from-supervision
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Use sft_messages_no_context instead of with_context",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if args.from_supervision:
        # Structured EvalSample format
        samples = load_dataset(dataset_path)
        records = export_from_supervision(samples, use_context=not args.no_context)
        print(f"[export_sft] Exported {len(records)} samples (from supervision)")

    else:
        # Raw dict format (FinanceBench JSONL)
        raw_samples: list[dict[str, Any]] = []
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_samples.append(json.loads(line))

        print(f"[export_sft] Loaded {len(raw_samples)} raw samples")

        # Optionally fetch chunk text from KB
        if args.fetch_chunks:
            print("[export_sft] Fetching gold chunks from KB...")
            raw_samples = fetch_gold_chunks_from_kb(raw_samples)

        records = export_from_evidence(
            raw_samples,
            system_prompt=args.system_prompt,
            max_evidence_chars=args.max_evidence_chars,
        )
        print(f"[export_sft] Exported {len(records)} samples (from evidence)")

    # Write output
    if not records:
        print("[ERROR] No valid SFT records generated. Check dataset format.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[export_sft] Written to: {out_path}")
    print(f"[export_sft] Sample output keys: {list(records[0].keys())}")
    print(f"[export_sft] Sample messages count: {len(records[0]['messages'])}")
    print(f"[export_sft] Evidence chars: {len(records[0].get('evidence_texts', [''])[0])}")


if __name__ == "__main__":
    main()
