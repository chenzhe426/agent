#!/usr/bin/env python3
"""
build_financebench.py – Build EvalSample dataset from FinanceBench JSONL files.

FinanceBench: https://github.com/自信心226/financebench
Paper: "FinanceBench: A Benchmark for Financial Question Answering over Knowledge Bases"

This script converts:
    financebench_document_information.jsonl  →  document metadata (maps doc_name → PDF)
    financebench_open_source.jsonl          →  Q&A pairs with evidence

Into:
    evals/datasets/financebench_{split}.jsonl  (EvalSample format)

Usage:
    # Build full open-source split
    python evals/scripts/build_financebench.py \
        --info data/financebench/data/financebench_document_information.jsonl \
        --qa    data/financebench/data/financebench_open_source.jsonl \
        --pdfs  data/financebench/pdfs/ \
        --output evals/datasets/financebench_open_source.jsonl

    # Build with PDF parsing (for gold chunk mapping)
    python evals/scripts/build_financebench.py \
        --info data/financebench/data/financebench_document_information.jsonl \
        --qa    data/financebench/data/financebench_open_source.jsonl \
        --pdfs  data/financebench/pdfs/ \
        --index-dir data/financebench/indexed/ \
        --output evals/datasets/financebench_open_source.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.dataset import (
    EvalSample,
    QuestionBlock,
    RetrievalBlock,
    ContextBlock,
    AnswerBlock,
    SupervisionBlock,
    EvaluationBlock,
    MetadataBlock,
)


# ---------------------------------------------------------------------------
# FinanceBench JSONL field mappers
# ---------------------------------------------------------------------------

def _normalize_refuse_phrases() -> set[str]:
    """Return the set of refuse detection phrases (used for auto-labeling)."""
    return {
        "没有足够信息", "无法确认", "当前未看到", "不能确认", "没有证据",
        "信息不足", "无法从", "知识库中未找到", "暂未找到", "没有找到相关",
        "not enough information", "cannot confirm", "insufficient information",
        "no relevant", "知识库里没有", "未提供",
    }


def _map_question_type(fb_type: str) -> str:
    """Map FinanceBench question_type to EvalSample task_type."""
    mapping = {
        "metrics-generated": "factoid",
        "domain-relevant": "factoid",
        "novel-generated": "factoid",
    }
    return mapping.get(fb_type, "factoid")


def _determine_difficulty(question: str, answer: str) -> str:
    """Simple heuristic: short answer + numeric = easy; long reasoning = medium."""
    if len(answer) < 50 and any(c.isdigit() for c in answer):
        return "easy"
    if len(question) > 150 or len(answer) > 300:
        return "medium"
    return "easy"


def _is_answer_refuse(answer: str) -> bool:
    """Check if a gold answer looks like a refusal / not-found response."""
    norm = answer.strip().lower()
    refuse_phrases = _normalize_refuse_phrases()
    return any(rp in norm for rp in refuse_phrases)


def _extract_must_include(answer: str) -> list[str]:
    """
    Extract key phrases from a gold answer that should appear in a correct response.
    For FinanceBench this is typically:
      - The numeric answer (e.g. "$1577.00", "8.70")
      - Key terms from the justification
    """
    must_include: list[str] = []

    # Extract dollar amounts
    amounts = re.findall(r'\$[\d,]+\.?\d*', answer)
    must_include.extend(amounts)

    # Extract percentages
    percents = re.findall(r'\d+\.?\d+\s*%', answer)
    must_include.extend(percents)

    # Extract key quoted terms (quoted in answer)
    quoted = re.findall(r'"([^"]{3,50})"', answer)
    must_include.extend(quoted[:3])

    # De-duplicate
    seen = set()
    unique = []
    for phrase in must_include:
        lower = phrase.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(phrase)

    return unique


def _extract_must_not_include() -> list[str]:
    """Answers should not include these phrases for wrong answers."""
    return []  # No automatic must-not rules for FinanceBench


def _build_context_from_evidence(evidence: list[dict]) -> list[str]:
    """
    Build gold_context_blocks from FinanceBench evidence list.
    Each evidence item has: evidence_text, doc_name, evidence_page_num
    """
    contexts = []
    for ev in evidence:
        text = ev.get("evidence_text", "")
        page = ev.get("evidence_page_num")
        doc = ev.get("doc_name", "")
        if text:
            if page:
                ctx = f"[Page {page} | {doc}] {text}"
            else:
                ctx = f"[{doc}] {text}"
            contexts.append(ctx.strip())
    return contexts


def _doc_name_to_pdf_path(doc_name: str, pdfs_dir: Path) -> Path | None:
    """
    Convert a FinanceBench doc_name (e.g. '3M_2015_10K') to a PDF path.
    Tries common naming patterns.
    """
    if not pdfs_dir or not pdfs_dir.exists():
        return None

    # Pattern 1: {doc_name}.pdf (exact)
    p = pdfs_dir / f"{doc_name}.pdf"
    if p.exists():
        return p

    # Pattern 2: case-insensitive search
    for p in pdfs_dir.glob("*.pdf"):
        if p.stem.lower() == doc_name.lower():
            return p

    return None


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def financebench_to_evalsample(
    fb_record: dict[str, Any],
    doc_info: dict[str, dict[str, Any]] | None = None,
    pdfs_dir: Path | None = None,
) -> EvalSample:
    """
    Convert a single FinanceBench open-source JSONL record into an EvalSample.

    Parameters
    ----------
    fb_record : FinanceBench JSONL dict (one line parsed)
    doc_info  : optional dict from document_information JSONL,
                keyed by doc_name, for metadata enrichment.
    pdfs_dir  : optional Path to PDFs dir; used to set gold_doc_ids via
                indexed chunk lookup (requires KB to be pre-indexed).

    For the initial build (no KB index), we create "unlabeled" retrieval
    samples — answer scoring runs, retrieval scoring is skipped.
    After running import-file on the PDFs, gold_chunk_ids can be resolved
    by re-running with --index-dir pointing to the indexed KB.
    """
    fb_id = fb_record.get("financebench_id", "")
    company = fb_record.get("company", "")
    doc_name = fb_record.get("doc_name", "")
    question = fb_record.get("question", "")
    answer = fb_record.get("answer", "")
    justification = fb_record.get("justification") or ""
    evidence_list = fb_record.get("evidence") or []
    fb_type = fb_record.get("question_type", "metrics-generated")
    subset = fb_record.get("dataset_subset_label", "OPEN_SOURCE")
    domain_num = fb_record.get("domain_question_num")

    # Determine tags
    tags = [company, doc_name, subset]
    if fb_type:
        tags.append(fb_type)
    if domain_num:
        tags.append(f"dg{domain_num}")

    # Determine expected_behavior
    if _is_answer_refuse(answer):
        expected_behavior = "refuse"
    else:
        expected_behavior = "answer"

    # Gold context from evidence
    gold_contexts = _build_context_from_evidence(evidence_list)

    # PDF path for this doc
    pdf_path = None
    if pdfs_dir:
        pdf_path = _doc_name_to_pdf_path(doc_name, pdfs_dir)

    # Build metadata
    meta = MetadataBlock(
        created_by="financebench_converter",
        source=f"financebench:{subset}",
        version="1.0",
    )

    # Map to EvalSample sub-blocks
    question_block = QuestionBlock(
        user_query=question,
        conversation_history=[],
    )

    # Retrieval: initially unlabeled (no gold chunk ids until PDF is indexed)
    retrieval_block = RetrievalBlock(
        label_status="unlabeled",
        gold_doc_ids=[],   # Will be set if pdf_path is indexed
        gold_chunk_ids=[],
        hard_negative_chunk_ids=[],
    )

    context_block = ContextBlock(
        gold_context_blocks=gold_contexts,
    )

    must_include = _extract_must_include(answer)
    must_not_include = _extract_must_not_include()

    answer_block = AnswerBlock(
        gold_answer=answer,
        answer_style="verbose",  # FinanceBench answers tend to be verbose
        must_include=must_include,
        must_not_include=must_not_include,
        faithfulness_requirements=[justification] if justification else [],
    )

    supervision_block = SupervisionBlock(
        preferred_output=answer,
        sft_messages_no_context=[],
        sft_messages_with_context=[],
        rejected_outputs=[],
    )

    eval_block = EvaluationBlock(
        expected_behavior=expected_behavior,
        scoring_type="rule+llm",
        error_type=None,
        notes=(
            f"financebench_id={fb_id}; "
            f"doc_name={doc_name}; "
            f"pdf={'found' if pdf_path else 'not_found'}; "
            f"num_evidence_items={len(evidence_list)}; "
            f"justification={justification[:100]!r}"
        ),
    )

    sample = EvalSample(
        id=fb_id,
        dataset=f"financebench_{subset.lower()}",
        task_type=_map_question_type(fb_type),
        difficulty=_determine_difficulty(question, answer),
        tags=[t for t in tags if t],
        question=question_block,
        retrieval=retrieval_block,
        context=context_block,
        answer=answer_block,
        supervision=supervision_block,
        evaluation=eval_block,
        metadata=meta,
    )

    return sample


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def load_doc_info(info_path: Path) -> dict[str, dict[str, Any]]:
    """Load document_information JSONL into a dict keyed by doc_name."""
    info: dict[str, dict[str, Any]] = {}
    if not info_path or not info_path.exists():
        return info

    with info_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_name = rec.get("doc_name", "")
            if doc_name:
                info[doc_name] = rec
    return info


def load_qa_records(qa_path: Path) -> list[dict[str, Any]]:
    """Load Q&A records from open-source JSONL."""
    records: list[dict[str, Any]] = []
    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON line: {e}", file=sys.stderr)
    return records


def resolve_gold_labels(
    samples: list[EvalSample],
    index_dir: Path,
    doc_name_to_doc_id: dict[str, int],
) -> list[EvalSample]:
    """
    Re-map gold_doc_ids and gold_chunk_ids for samples where the PDF
    has been indexed into the KB.

    index_dir should contain a manifest or be a directory that the
    chunk_service can query to resolve doc_name → chunk_ids.

    For now this is a stub — the actual implementation depends on
    how chunks are stored in the KB. Override this function when
    the chunk_id resolution logic is available.
    """
    # TODO: Implement chunk_id resolution from indexed PDFs.
    # For now, leave retrieval as "unlabeled" — scoring will only
    # evaluate the answer quality, not retrieval quality.
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EvalSample dataset from FinanceBench JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--info",
        required=True,
        type=Path,
        help="Path to financebench_document_information.jsonl",
    )
    parser.add_argument(
        "--qa",
        required=True,
        type=Path,
        help="Path to financebench_open_source.jsonl",
    )
    parser.add_argument(
        "--pdfs",
        type=Path,
        default=None,
        help="Path to directory containing PDF files (for doc_name → PDF mapping)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Path to indexed KB directory (enables gold_chunk_ids resolution)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path (EvalSample format)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Filter to a specific dataset_subset_label (e.g. OPEN_SOURCE)",
    )
    parser.add_argument(
        "--skip-empty-answer",
        action="store_true",
        help="Skip records where answer is empty",
    )
    args = parser.parse_args()

    # Load doc info
    print(f"[build_financebench] Loading document info from: {args.info}")
    doc_info = load_doc_info(args.info)
    print(f"[build_financebench] Loaded {len(doc_info)} document records")

    # Load Q&A records
    print(f"[build_financebench] Loading QA records from: {args.qa}")
    qa_records = load_qa_records(args.qa)
    print(f"[build_financebench] Loaded {len(qa_records)} QA records")

    # Filter by subset if specified
    if args.subset:
        qa_records = [r for r in qa_records if r.get("dataset_subset_label") == args.subset]
        print(f"[build_financebench] Filtered to subset '{args.subset}': {len(qa_records)} records")

    # Convert
    samples: list[EvalSample] = []
    skipped = 0

    for i, rec in enumerate(qa_records):
        answer = rec.get("answer", "").strip()
        if args.skip_empty_answer and not answer:
            skipped += 1
            continue

        try:
            sample = financebench_to_evalsample(rec, doc_info=doc_info, pdfs_dir=args.pdfs)
            samples.append(sample)
        except Exception as e:
            print(f"[WARN] Failed to convert record {rec.get('financebench_id', '?')}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"[build_financebench] Processed {i+1}/{len(qa_records)} records...")

    # Optional: resolve gold labels if index-dir provided
    if args.index_dir and args.index_dir.exists():
        print(f"[build_financebench] Resolving gold labels from index: {args.index_dir}")
        # doc_name_to_doc_id mapping would be loaded from the index here
        # samples = resolve_gold_labels(samples, args.index_dir, doc_name_to_doc_id)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    print(f"[build_financebench] Done. Wrote {count} EvalSamples to {args.output}")
    if skipped:
        print(f"[build_financebench] Skipped {skipped} records")


if __name__ == "__main__":
    main()
