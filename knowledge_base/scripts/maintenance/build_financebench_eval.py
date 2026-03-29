#!/usr/bin/env python3
"""
Stage 1: Build FinanceBench eval dataset (no KB dependency).

Reads two JSONL files, joins on doc_name, produces a unified eval dataset
with gold evidence aligned at page level.

Usage:
    python -m evals.scripts.build_financebench_eval

Output:
    evals/data/financebench_v1.jsonl
    (also prints statistics)
"""
from __future__ import annotations

import json
import sys
import unicodedata
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "financebench"


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[\s\u00a0\u3000]+", " ", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text.strip()


def normalize_number_friendly(text: str) -> str:
    """Normalize numbers: $1,577.00 → 1577"""
    text = normalize_text(text)
    text = re.sub(r"[$€£¥]", "", text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    return text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_doc_info(path: Path) -> dict[str, dict]:
    """Load document_information JSONL → dict keyed by doc_name."""
    info: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                dn = rec.get("doc_name", "")
                if dn:
                    info[dn] = rec
            except json.JSONDecodeError:
                continue
    return info


def load_qa_records(path: Path) -> list[dict]:
    """Load open-source QA JSONL records."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON: {e}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Gold evidence extraction
# ---------------------------------------------------------------------------

def extract_gold_evidence(qa_record: dict) -> list[dict]:
    """
    Extract gold evidence list from a QA record's `evidence` field.

    Each evidence item → {
        doc_name, page_num, evidence_text, evidence_text_full_page
    }
    """
    evidence_list = qa_record.get("evidence") or []
    results: list[dict] = []

    for ev in evidence_list:
        doc_name = ev.get("doc_name", qa_record.get("doc_name", ""))
        page_num_raw = ev.get("evidence_page_num")
        try:
            page_num = int(page_num_raw) if page_num_raw is not None else 0
        except (TypeError, ValueError):
            page_num = 0

        results.append({
            "doc_name": doc_name,
            "page_num": page_num,
            "evidence_text": ev.get("evidence_text", ""),
            "evidence_text_full_page": ev.get("evidence_text_full_page", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Must-include / must-not-include extraction (for answer scoring)
# ---------------------------------------------------------------------------

def extract_must_include(answer: str) -> list[str]:
    """Extract key phrases from gold answer for answer scoring."""
    must_include: list[str] = []
    amounts = re.findall(r'\$[\d,]+\.?\d*', answer)
    must_include.extend(amounts)
    percents = re.findall(r'\d+\.?\d+\s*%', answer)
    must_include.extend(percents)
    quoted = re.findall(r'"([^"]{3,50})"', answer)
    must_include.extend(quoted[:3])
    seen = set()
    unique = []
    for phrase in must_include:
        lower = phrase.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(phrase)
    return unique


def is_refuse_answer(answer: str) -> bool:
    REFUSE = {
        "没有足够信息", "无法确认", "当前未看到", "不能确认", "没有证据",
        "信息不足", "无法从", "知识库中未找到", "暂未找到", "没有找到相关",
        "not enough information", "cannot confirm", "insufficient information",
        "no relevant", "知识库里没有", "未提供",
    }
    norm = answer.strip().lower()
    return any(rp in norm for rp in REFUSE)


# ---------------------------------------------------------------------------
# Build one eval sample
# ---------------------------------------------------------------------------

def build_sample(qa_record: dict, doc_info: dict[str, dict]) -> dict[str, Any]:
    """Build a single eval sample from a FinanceBench QA record."""
    fb_id = qa_record.get("financebench_id", "")
    doc_name = qa_record.get("doc_name", "")
    question = qa_record.get("question", "")
    answer = qa_record.get("answer", "")
    justification = qa_record.get("justification") or ""
    question_type = qa_record.get("question_type", "")
    reasoning_type = qa_record.get("question_reasoning", "") or ""

    info = doc_info.get(doc_name, {})

    # Gold evidence
    gold_evidence = extract_gold_evidence(qa_record)
    gold_pages = sorted(set(e["page_num"] for e in gold_evidence if e["page_num"] > 0))
    gold_evidence_texts = [e["evidence_text"] for e in gold_evidence if e["evidence_text"]]

    # Determine expected behavior
    expected_behavior = "refuse" if is_refuse_answer(answer) else "answer"

    # Question type → task type
    task_type_map = {
        "metrics-generated": "factoid",
        "domain-relevant": "factoid",
        "novel-generated": "factoid",
    }
    task_type = task_type_map.get(question_type, "factoid")

    # Difficulty
    if len(answer) < 50 and any(c.isdigit() for c in answer):
        difficulty = "easy"
    elif len(question) > 150 or len(answer) > 300:
        difficulty = "medium"
    else:
        difficulty = "easy"

    # Must-include phrases from answer
    must_include = extract_must_include(answer)
    must_not_include: list[str] = []

    sample = {
        # Core IDs
        "eval_id": fb_id,
        "question": question,

        # Document metadata
        "company": info.get("company", qa_record.get("company", "")),
        "gold_doc_name": doc_name,
        "gold_doc_type": info.get("doc_type", ""),
        "gold_doc_period": info.get("doc_period", ""),

        # Answer gold
        "gold_answer": answer,
        "gold_justification": justification,

        # Evidence
        "gold_evidence": gold_evidence,
        "gold_pages": gold_pages,
        "gold_evidence_texts": gold_evidence_texts,

        # Question type
        "question_type": question_type,
        "reasoning_type": reasoning_type,

        # Scoring hints
        "must_include": must_include,
        "must_not_include": must_not_include,
        "expected_behavior": expected_behavior,
        "task_type": task_type,
        "difficulty": difficulty,

        # Placeholder for future gold_chunk_ids (Stage 3+)
        "gold_chunk_ids": None,
    }

    return sample


# ---------------------------------------------------------------------------
# Subset selection (Stage 2)
# ---------------------------------------------------------------------------

def select_3doc_subset(
    samples: list[dict[str, Any]],
    docs_desired: int = 3,
    questions_per_doc: int = 5,
) -> list[dict[str, Any]]:
    """
    Select a small subset of samples covering exactly `docs_desired` documents.
    Chooses documents with most questions first, then fills quota.
    """
    # Count questions per doc
    doc_counts: Counter = Counter(s.get("gold_doc_name", "") for s in samples)
    if not doc_counts:
        return []

    # Pick top N docs by question count
    top_docs = [doc for doc, _ in doc_counts.most_common(docs_desired)]

    subset: list[dict[str, Any]] = []
    doc_selected: dict[str, int] = {d: 0 for d in top_docs}

    for s in samples:
        doc = s.get("gold_doc_name", "")
        if doc in doc_selected and doc_selected[doc] < questions_per_doc:
            subset.append(s)
            doc_selected[doc] += 1
            if all(v >= questions_per_doc for v in doc_selected.values()):
                break

    return subset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FinanceBench eval dataset")
    parser.add_argument(
        "--info",
        type=Path,
        default=DATA_DIR / "data" / "financebench_document_information.jsonl",
        help="Path to document_information JSONL",
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=DATA_DIR / "data" / "financebench_open_source.jsonl",
        help="Path to open_source JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "data" / "financebench_v1.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--subset-3docs",
        action="store_true",
        help="Also build a 3-doc subset (financebench_v1_subset_3docs.jsonl)",
    )
    args = parser.parse_args()

    # ---- Load data ----
    print("[build_eval] Loading data...")
    doc_info = load_doc_info(args.info)
    qa_records = load_qa_records(args.qa)
    print(f"  doc_info records: {len(doc_info)}")
    print(f"  QA records:       {len(qa_records)}")

    # ---- Build full eval dataset ----
    print("[build_eval] Building eval samples...")
    samples: list[dict] = []
    for qa in qa_records:
        try:
            s = build_sample(qa, doc_info)
            samples.append(s)
        except Exception as e:
            print(f"  [WARN] Failed to build sample {qa.get('financebench_id', '?')}: {e}", file=sys.stderr)

    # ---- Write full dataset ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- Print statistics ----
    _print_stats(samples, label="Full dataset")

    # ---- Stage 2: 3-doc subset ----
    if args.subset_3docs:
        subset = select_3doc_subset(samples, docs_desired=3, questions_per_doc=5)
        subset_path = args.output.parent / "financebench_v1_subset_3docs.jsonl"
        with subset_path.open("w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Also write the 3 docs' metadata
        doc_names_used = sorted(set(s.get("gold_doc_name", "") for s in subset))
        doc_meta_records = []
        for dn in doc_names_used:
            info = doc_info.get(dn, {})
            doc_meta_records.append({
                "doc_name": dn,
                "company": info.get("company", ""),
                "doc_type": info.get("doc_type", ""),
                "doc_period": info.get("doc_period", ""),
                "source_path_or_url": info.get("doc_link", ""),
            })
        docs_meta_path = args.output.parent / "financebench_v1_subset_3docs_docs.jsonl"
        with docs_meta_path.open("w", encoding="utf-8") as f:
            for rec in doc_meta_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n[build_eval] 3-doc subset written to: {subset_path}")
        _print_stats(subset, label="3-doc subset")
        print(f"  Doc metadata: {docs_meta_path}")
        for rec in doc_meta_records:
            print(f"    - {rec['doc_name']} ({rec['company']}, {rec['doc_period']})")


def _print_stats(samples: list[dict], label: str) -> None:
    n = len(samples)
    companies = len(set(s.get("company", "") for s in samples))
    doc_names = len(set(s.get("gold_doc_name", "") for s in samples))
    has_pages = sum(1 for s in samples if s.get("gold_pages"))
    has_evidence = sum(1 for s in samples if s.get("gold_evidence_texts"))

    qt = Counter(s.get("question_type", "") for s in samples)
    rt = Counter(s.get("reasoning_type", "") or "(none)" for s in samples)
    eb = Counter(s.get("expected_behavior", "") for s in samples)

    print(f"\n=== {label} ===")
    print(f"  Total samples:          {n}")
    print(f"  Unique companies:        {companies}")
    print(f"  Unique doc_names:        {doc_names}")
    print(f"  With gold_pages:         {has_pages}")
    print(f"  With gold_evidence_text: {has_evidence}")
    print(f"  Question types:          {dict(qt)}")
    print(f"  Reasoning types:         {dict(rt)}")
    print(f"  Expected behavior:       {dict(eb)}")


if __name__ == "__main__":
    main()
