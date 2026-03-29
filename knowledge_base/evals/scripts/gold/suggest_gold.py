#!/usr/bin/env python3
"""
suggest_gold.py – AI-assisted suggestion of gold retrieval labels.

Usage:
    python evals/scripts/suggest_gold.py \\
        --report evals/reports/run_v2.json

Requirements:
    The report MUST contain the `retrieved_chunks` field with text previews.
    Re-run eval first to generate a fresh report with the updated report.py:
        python evals/scripts/run_eval.py ... --output evals/reports/run_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.report import load_report


# ---------------------------------------------------------------------------
# Data extraction – new format only
# ---------------------------------------------------------------------------

def _extract_case_chunks(case: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract retrieval candidates from a case dict.

    Requires the new report format: case["retrieved_chunks"] with text previews.
    Raises ValueError if the format is missing/invalid.
    """
    chunks = case.get("retrieved_chunks")
    if chunks is None:
        raise ValueError(
            f"case {case.get('id', '?')} is missing 'retrieved_chunks' field. "
            f"This report appears to be in OLD format. "
            f"Re-run eval with the updated report.py to generate a new-format report, "
            f"then re-run suggest_gold on that report."
        )

    if not isinstance(chunks, list):
        raise ValueError(
            f"case {case.get('id', '?')}: 'retrieved_chunks' is not a list (got {type(chunks).__name__})."
        )

    results: list[dict[str, Any]] = []
    for ch in chunks[:8]:
        text = (
            ch.get("content_preview")
            or ch.get("chunk_text")
            or ch.get("search_text")
            or ""
        )[:400]
        results.append({
            "chunk_id": ch.get("chunk_id"),
            "document_id": ch.get("document_id"),
            "score": ch.get("score"),
            "title": ch.get("title", ""),
            "section_path": ch.get("section_path", ""),
            "content_preview": text,
        })
    return results


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def _load_llm():
    from app.services.llm_service import chat_completion
    return chat_completion


def _llm_suggest(case: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Call LLM to analyze retrieved chunks and return a gold suggestion.
    Requires chunks to have text previews (new report format).
    """
    question = case.get("question", "")
    answer_text = (case.get("final_answer") or "")[:300]
    task_type = case.get("task_type", "")

    eval_block = case.get("evaluation")
    if isinstance(eval_block, dict):
        expected_behavior = eval_block.get("expected_behavior", "answer")
    else:
        expected_behavior = "answer"

    # refuse / clarify samples skip retrieval labeling
    if expected_behavior in {"refuse", "clarify"}:
        return {
            "label_status": "skip",
            "gold_chunk_ids": [],
            "gold_doc_ids": [],
            "reason": f"expected_behavior='{expected_behavior}' – retrieval labeling not applicable",
            "error": "",
        }

    # Build chunk display for the prompt
    chunk_lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        preview = ch.get("content_preview", "")
        chunk_lines.append(
            f"[{i}] chunk_id={ch['chunk_id']}  doc_id={ch['document_id']}  "
            f"score={ch.get('score')}\n"
            f"    title: {ch.get('title', '-') or '-'}\n"
            f"    section: {ch.get('section_path', '-') or '-'}\n"
            f"    text: {preview[:300]}"
        )

    chunk_list_text = "\n\n".join(chunk_lines) if chunk_lines else "(no chunks)"

    system_prompt = (
        "You are an expert data annotator for a RAG retrieval evaluation dataset.\n"
        "Given a user question, retrieved chunks with their text, and the system's answer,\n"
        "determine which chunks DIRECTLY contain sufficient evidence to answer.\n\n"
        "IMPORTANT RULES:\n"
        "1. ONLY select chunks that EXPLICITLY contain the answer – not just 'possibly related'.\n"
        "2. Prefer the MINIMAL set (1-2 chunks) that together contain enough evidence.\n"
        "3. If no chunk contains sufficient evidence -> label_status='skip'.\n"
        "4. Use 'labeled_chunk' when specific chunk_ids contain the answer.\n"
        "5. Use 'labeled_doc' when only the doc is relevant, no specific chunk.\n"
        "6. Respond in JSON only, no explanation outside JSON.\n\n"
        "Output format:\n"
        '{\n'
        '  "label_status": "labeled_chunk",\n'
        '  "gold_chunk_ids": [11, 12],    // empty if skip\n'
        '  "gold_doc_ids": [],             // empty if skip\n'
        '  "reason": "why these chunks were chosen"\n'
        '}\n\n'
        "Be strict. When in doubt, prefer skip."
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Expected behavior: {expected_behavior}\n"
        f"Task type: {task_type}\n"
        f"System answer: {answer_text}\n\n"
        f"Retrieved chunks (ranked):\n{chunk_list_text}\n\n"
        "Respond with JSON only."
    )

    try:
        chat_completion = _load_llm()
        raw = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt).strip()

        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.lstrip("json\n").lstrip("json").strip()

        parsed = json.loads(raw)
        label_status = str(parsed.get("label_status", "skip"))
        if label_status not in {"labeled_chunk", "labeled_doc", "skip"}:
            label_status = "skip"

        return {
            "label_status": label_status,
            "gold_chunk_ids": [int(x) for x in parsed.get("gold_chunk_ids", []) if x],
            "gold_doc_ids": [int(x) for x in parsed.get("gold_doc_ids", []) if x],
            "reason": str(parsed.get("reason", "")),
            "error": "",
        }

    except json.JSONDecodeError as e:
        return {
            "label_status": "skip",
            "gold_chunk_ids": [],
            "gold_doc_ids": [],
            "reason": "",
            "error": f"JSON parse error: {e} – raw: {raw[:100]!r}",
        }
    except Exception as e:
        return {
            "label_status": "skip",
            "gold_chunk_ids": [],
            "gold_doc_ids": [],
            "reason": "",
            "error": f"LLM call failed: {e}",
        }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _render_suggestion(case: dict[str, Any], suggestion: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  Case: {case['id']}  |  current status: {case.get('label_status', '?')}")
    lines.append("-" * 70)
    lines.append(f"  Question: {case['question'][:80]}")
    lines.append("-" * 70)

    err = suggestion.get("error", "")
    if err:
        lines.append(f"  ⚠️  Error: {err}")
        lines.append("=" * 70)
        lines.append("")
        return lines

    label = suggestion.get("label_status", "skip")
    lines.append(f"  → Suggested : {label}")
    if label != "skip":
        lines.append(f"  → gold_chunk_ids: {suggestion.get('gold_chunk_ids', [])}")
        lines.append(f"  → gold_doc_ids  : {suggestion.get('gold_doc_ids', [])}")
    reason = suggestion.get("reason", "")
    if reason:
        for ln in reason.split("\n"):
            stripped = ln.strip()
            if stripped:
                lines.append(f"     {stripped}")
    lines.append("=" * 70)
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-assisted gold label suggestions (requires new report format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nNOTE: This script requires the new report format with 'retrieved_chunks' "
            "field (with text previews).\n"
            "If you see a format error, re-run eval first:\n"
            "  python evals/scripts/run_eval.py ... --output evals/reports/run_v2.json\n"
            "Then re-run this script on the new report."
        ),
    )
    parser.add_argument("--report", required=True, help="Path to run_eval JSON report (new format required)")
    parser.add_argument("--case", help="Only process this case ID")
    parser.add_argument("--only-unlabeled", action="store_true", help="Only unlabeled samples")
    parser.add_argument(
        "--only-need-labeling", action="store_true",
        help="Only unlabeled + unanswerable samples",
    )
    parser.add_argument("--output", help="Write suggestions to JSON file (in addition to console)")
    args = parser.parse_args()

    report = load_report(args.report)
    cases = report.get("cases", [])

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"ERROR: case '{args.case}' not found.", file=sys.stderr)
            sys.exit(1)

    if args.only_unlabeled:
        cases = [c for c in cases if c.get("label_status") == "unlabeled"]

    if args.only_need_labeling:
        cases = [c for c in cases if c.get("label_status") in {"unlabeled", "unanswerable"}]

    if not cases:
        print("No cases match the filter.", file=sys.stderr)
        sys.exit(0)

    # Validate new format on first case; fail fast if old format detected
    sample = cases[0]
    if "retrieved_chunks" not in sample:
        print(
            f"\nERROR[suggest_gold]: report '{args.report}' is in OLD format\n"
            f"  (missing 'retrieved_chunks' field in case {sample.get('id')}).\n\n"
            f"  This script requires the new report format with text previews.\n"
            f"  Re-run eval first, then re-run suggest_gold on the new report:\n\n"
            f"    python evals/scripts/run_eval.py \\\n"
            f"        --dataset evals/datasets/kb_eval_seed.jsonl \\\n"
            f"        --config evals/configs/baseline.yaml \\\n"
            f"        --output evals/reports/run_v2.json\n\n"
            f"    python evals/scripts/suggest_gold.py \\\n"
            f"        --report evals/reports/run_v2.json \\\n"
            f"        --only-need-labeling\n",
            file=sys.stderr,
        )
        sys.exit(1)

    all_suggestions: list[dict[str, Any]] = []
    total = len(cases)

    print(f"# Gold Suggestion Report – {report.get('run_id', '?')}")
    print(f"# Processing {total} case(s)\n")

    for i, case in enumerate(cases, start=1):
        tag = case.get("label_status", "?")
        print(f"[{i}/{total}] {case['id']} (status={tag}) ...", end=" ", flush=True)

        try:
            chunks = _extract_case_chunks(case)
            suggestion = _llm_suggest(case, chunks)
        except ValueError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            sys.exit(1)

        print("done" if not suggestion.get("error") else f"ERROR: {suggestion['error'][:60]}")

        all_suggestions.append({
            "case_id": case["id"],
            "question": case["question"],
            "current_label_status": case.get("label_status"),
            **suggestion,
        })
        for line in _render_suggestion(case, suggestion):
            print(line)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "run_id": report.get("run_id", ""),
                "report": args.report,
                "suggestions": all_suggestions,
            }, f, ensure_ascii=False, indent=2)
        print(f"[suggest_gold] Written to: {out_path}")


if __name__ == "__main__":
    main()
