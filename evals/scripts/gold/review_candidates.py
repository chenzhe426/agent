#!/usr/bin/env python3
"""
review_candidates.py – Review retrieval candidates from an eval run report.

Helps human annotators decide which retrieved chunks/docs should become gold labels.

Usage:
    python evals/scripts/review_candidates.py \\
        --report evals/reports/run_debug.json

    # Single case
    python evals/scripts/review_candidates.py \\
        --report evals/reports/run_debug.json --case kb_0001

    # Only unlabeled samples
    python evals/scripts/review_candidates.py \\
        --report evals/reports/run_debug.json --only-unlabeled

    # Markdown output (for copy-paste into documents)
    python evals/scripts/review_candidates.py \\
        --report evals/reports/run_debug.json --markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt_chunk(ch: dict[str, Any], rank: int) -> list[str]:
    lines = [
        f"  [{rank}] chunk_id={ch.get('chunk_id')}  doc_id={ch.get('document_id')}  score={ch.get('score')}",
        f"      title: {ch.get('title', '-') or '-'}",
        f"      section: {ch.get('section_path', '-') or '-'}",
    ]
    preview = ch.get("content_preview", "")
    if preview:
        # Truncate to 3 lines for readability
        preview_lines = preview.split("\n")[:3]
        preview = "  ".join(preview_lines)
        lines.append(f"      text: {preview[:300]}")
    return lines


def render_case(case: dict[str, Any]) -> list[str]:
    """Render a single case as a list of text lines."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  Case: {case['id']}")
    lines.append(f"  Status: label_status={case['label_status']}  answer={case['answer_label']}")
    lines.append(f"  Question: {case['question']}")
    lines.append(f"  Gold chunk_ids: {case.get('gold_chunk_ids', [])}")
    lines.append(f"  Gold doc_ids:   {case.get('gold_doc_ids', [])}")
    lines.append("-" * 70)
    lines.append(f"  Retrieved chunk_ids: {case.get('retrieved_chunk_ids', [])}")
    lines.append(f"  Retrieved doc_ids:   {case.get('retrieved_doc_ids', [])}")
    lines.append("-" * 70)
    lines.append("  Retrieved Chunks (ranked):")
    chunks = case.get("retrieved_chunks", [])
    if not chunks:
        lines.append("  (none)")
    else:
        for i, ch in enumerate(chunks, start=1):
            lines.extend(_fmt_chunk(ch, i))
    lines.append("-" * 70)
    answer = case.get("final_answer", "")
    lines.append(f"  Final Answer: {(answer or '(empty)')[:200]}")
    lines.append("=" * 70)
    lines.append("")
    return lines


def render_markdown_case(case: dict[str, Any]) -> list[str]:
    """Render a single case as markdown."""
    lines: list[str] = []
    answer = case.get("final_answer", "")
    chunks = case.get("retrieved_chunks", [])

    lines.append(f"## [{case['id']}] {case['question'][:60]}{'...' if len(case['question']) > 60 else ''}")
    lines.append("")
    lines.append(f"**label_status**: `{case['label_status']}`  |  **answer_label**: `{case['answer_label']}`")
    lines.append("")
    lines.append(f"**Gold chunk_ids**: `{case.get('gold_chunk_ids', [])}`")
    lines.append(f"**Gold doc_ids**: `{case.get('gold_doc_ids', [])}`")
    lines.append("")
    lines.append("### Retrieved Chunks")
    lines.append("")

    if not chunks:
        lines.append("_none_")
    else:
        for i, ch in enumerate(chunks, start=1):
            preview = ch.get("content_preview", "") or ""
            lines.append(f"| #{i} | `chunk_id={ch.get('chunk_id')}` | `doc_id={ch.get('document_id')}` | `score={ch.get('score')}` |")
            lines.append(f"| title | {ch.get('title', '-') or '-'} |")
            lines.append(f"| section | {ch.get('section_path', '-') or '-'} |")
            # Truncate preview to 200 chars for readability
            snippet = preview[:200].replace("\n", " ").strip()
            lines.append(f"| text | {snippet} |")
            lines.append("")

    lines.append("### Final Answer")
    lines.append("")
    lines.append(f"> {(answer or '(empty)')[:300].replace(chr(10), ' ')}")
    lines.append("")
    lines.append(f"**Latency**: {case.get('latency_ms', 0):.0f} ms")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Review retrieval candidates for gold annotation")
    parser.add_argument(
        "--report", required=True,
        help="Path to the JSON report from run_eval.py",
    )
    parser.add_argument(
        "--case",
        help="Show only this case ID",
    )
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Show only unlabeled samples",
    )
    parser.add_argument(
        "--only-unanswerable",
        action="store_true",
        help="Show only unanswerable samples",
    )
    parser.add_argument(
        "--only-need-labeling",
        action="store_true",
        help="Show only unlabeled + unanswerable samples (convenience flag)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as markdown (for copy-paste)",
    )
    args = parser.parse_args()

    report = load_report(args.report)
    cases = report.get("cases", [])

    # Filter
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"ERROR: case '{args.case}' not found in report.", file=sys.stderr)
            sys.exit(1)

    if args.only_unlabeled:
        cases = [c for c in cases if c.get("label_status") == "unlabeled"]

    if args.only_unanswerable:
        cases = [c for c in cases if c.get("label_status") == "unanswerable"]

    if args.only_need_labeling:
        cases = [c for c in cases if c.get("label_status") in {"unlabeled", "unanswerable"}]

    if not cases:
        print("No cases match the filter.", file=sys.stderr)
        sys.exit(0)

    # Header
    print(f"# Candidates Review – {report.get('run_id', '?')}  ({len(cases)} cases)\n")

    for case in cases:
        if args.markdown:
            lines = render_markdown_case(case)
        else:
            lines = render_case(case)
        print("\n".join(lines))


if __name__ == "__main__":
    main()
