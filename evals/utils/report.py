"""
Report building utilities for evaluation runs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.utils.dataset import EvalSample


# ---------------------------------------------------------------------------
# JSON report loading
# ---------------------------------------------------------------------------

def load_report(path: str | Path) -> dict[str, Any]:
    """Load a run_eval JSON report from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_json_report(
    run_id: str,
    dataset_name: str,
    config: dict[str, Any],
    samples: list[EvalSample],
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a structured JSON report.

    Parameters
    ----------
    run_id        : unique run identifier
    dataset_name  : name of the dataset file
    config        : evaluation config dict
    samples       : original EvalSample list
    case_results  : per-sample result dicts from the scorer
    """
    total = len(case_results)

    # Aggregate retrieval metrics
    retrieval_metrics = _aggregate_retrieval(case_results)
    answer_metrics = _aggregate_answer(case_results)

    # Count labeled vs skipped
    retrieval_labeled_cases = sum(
        1 for cr in case_results if not cr.get("retrieval_skipped")
    )
    retrieval_skipped_cases = sum(
        1 for cr in case_results if cr.get("retrieval_skipped")
    )

    # Build per-case summaries
    cases: list[dict[str, Any]] = []
    for sample, cr in zip(samples, case_results):
        label_status = sample.retrieval.label_status

        # Build rich retrieved chunks list for annotation review
        retrieved_chunks_out: list[dict[str, Any]] = []
        for ch in cr.get("retrieved_chunks", []):
            chunk_text = str(ch.get("chunk_text") or ch.get("search_text") or "")
            retrieved_chunks_out.append({
                "chunk_id": ch.get("chunk_id"),
                "document_id": ch.get("document_id"),
                "score": ch.get("score"),
                "title": ch.get("title", ""),
                "section_title": ch.get("section_title", ""),
                "section_path": ch.get("section_path", ""),
                "content_preview": chunk_text[:400] if chunk_text else "",
            })

        cases.append({
            "id": sample.id,
            "task_type": sample.task_type,
            "difficulty": sample.difficulty,
            "label_status": label_status,
            "question": sample.question.user_query,
            "evaluation": asdict(sample.evaluation),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in cr.get("retrieved_chunks", [])],
            "retrieved_doc_ids": list({
                c.get("document_id") for c in cr.get("retrieved_chunks", [])
                if c.get("document_id") is not None
            }),
            "gold_chunk_ids": sample.retrieval.gold_chunk_ids,
            "gold_doc_ids": sample.retrieval.gold_doc_ids,
            "retrieval": {
                "hit_at_1": cr.get("retrieval_hit_at_1"),
                "hit_at_3": cr.get("retrieval_hit_at_3"),
                "hit_at_5": cr.get("retrieval_hit_at_5"),
                "mrr": cr.get("retrieval_mrr"),
                "recall_at_5": cr.get("retrieval_recall_at_5"),
                "skipped": cr.get("retrieval_skipped", False),
                # New P0 page-level
                "page_strict_hit_at_5": cr.get("retrieval_page_strict_hit_at_5"),
                "page_relaxed_hit_at_5": cr.get("retrieval_page_relaxed_hit_at_5"),
                # New P0 section-level
                "section_hit_at_5": cr.get("retrieval_section_hit_at_5"),
                # New P0 evidence semantic
                "evidence_semantic_hit_at_5": cr.get("evidence_semantic_hit_at_5"),
                "evidence_lexical_hit_at_5": cr.get("evidence_lexical_hit_at_5"),
            },
            "retrieved_chunks": retrieved_chunks_out,
            "final_answer": cr.get("final_answer", ""),
            "answer_label": cr.get("answer_label", ""),
            "must_include_hit_ratio": cr.get("must_include_hit_ratio", 0.0),
            "must_not_include_violations": cr.get("must_not_include_violations", []),
            "latency_ms": round(cr.get("latency_ms", 0.0), 2),
            "error": cr.get("error", ""),
            # P1 failure reason
            "_failure_reason": cr.get("_failure_reason", ""),
            # V4 pipeline fields
            "v4_answer_supported": cr.get("v4_answer_supported"),
            "v4_support_level": cr.get("v4_support_level"),
            "v4_numeric_consistent": cr.get("v4_numeric_consistent"),
            "v4_citation_adequate": cr.get("v4_citation_adequate"),
            "v4_failure_reasons": cr.get("v4_failure_reasons", []),
            "v4_missing_requirements": cr.get("v4_missing_requirements", []),
            "v4_verifier_method": cr.get("v4_verifier_method"),
            "v4_was_refined": cr.get("v4_was_refined"),
            "v4_refinement_round": cr.get("v4_refinement_round"),
            "v4_refine_trigger": cr.get("v4_refine_trigger"),
            "v4_verifier_fallback": cr.get("v4_verifier_fallback"),
            "v4_refine_fallback": cr.get("v4_refine_fallback"),
        })

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "config": config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "retrieval_labeled_cases": retrieval_labeled_cases,
            "retrieval_skipped_cases": retrieval_skipped_cases,
            "retrieval": retrieval_metrics,
            "answer": answer_metrics,
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build a human-readable Markdown report from a JSON report dict.
    """
    summary = report.get("summary", {})
    retr = summary.get("retrieval", {})
    ans = summary.get("answer", {})
    cases = report.get("cases", [])

    lines = [
        "# Evaluation Report",
        "",
        f"**Run ID**: `{report.get('run_id', '')}`",
        f"**Dataset**: {report.get('dataset', '')}",
        f"**Generated**: {report.get('generated_at', '')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total samples | {summary.get('total', 0)} |",
        f"| Retrieval labeled cases | {summary.get('retrieval_labeled_cases', 0)} |",
        f"| Retrieval skipped (unlabeled / unanswerable) | {summary.get('retrieval_skipped_cases', 0)} |",
        "",
        "### Retrieval  _(only over labeled cases)_",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Hit@1 | {retr.get('hit_at_1', 0.0)} |",
        f"| Hit@3 | {retr.get('hit_at_3', 0.0)} |",
        f"| Hit@5 | {retr.get('hit_at_5', 0.0)} |",
        f"| Recall@5 | {retr.get('recall_at_5', 0.0)} |",
        f"| MRR | {retr.get('mrr', 0.0)} |",
        f"| Page strict hit@5 | {retr.get('page_strict_hit_at_5', 'N/A')} |",
        f"| Page relaxed hit@5 | {retr.get('page_relaxed_hit_at_5', 'N/A')} |",
        f"| Section hit@5 | {retr.get('section_hit_at_5', 'N/A')} |",
        f"| Evidence semantic hit@5 | {retr.get('evidence_semantic_hit_at_5', 'N/A')} |",
        f"| Evidence lexical hit@5 | {retr.get('evidence_lexical_hit_at_5', 'N/A')} |",
        "",
        "### Answer",
        "",
        f"| Label | Count |",
        f"|-------|-------|",
        f"| exact | {ans.get('label_counts', {}).get('exact', 0)} |",
        f"| partial | {ans.get('label_counts', {}).get('partial', 0)} |",
        f"| wrong | {ans.get('label_counts', {}).get('wrong', 0)} |",
        f"| refuse_correct | {ans.get('label_counts', {}).get('refuse_correct', 0)} |",
        f"| refuse_wrong | {ans.get('label_counts', {}).get('refuse_wrong', 0)} |",
        f"| clarify_correct | {ans.get('label_counts', {}).get('clarify_correct', 0)} |",
        f"| clarify_wrong | {ans.get('label_counts', {}).get('clarify_wrong', 0)} |",
        "",
        "### V4 Pipeline (answer verifier + self-refine)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Answer supported rate | {ans.get('answer_supported_rate', 'N/A')} |",
        f"| Answer refined rate | {ans.get('answer_refined_rate', 'N/A')} |",
        f"| Numeric verifier pass rate | {ans.get('numeric_verifier_pass_rate', 'N/A')} |",
        f"| Citation adequate rate | {ans.get('citation_adequate_rate', 'N/A')} |",
        "",
        "### V4 Fallback Rates (lower is better)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Verifier fallback rate | {ans.get('verifier_fallback_rate', 'N/A')} |",
        f"| Refine fallback rate | {ans.get('refine_fallback_rate', 'N/A')} |",
        f"| LLM rerank fallback rate | {retr.get('llm_rerank_fallback_rate', 'N/A')} |",
        "",
        "### LLM Reranker",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| LLM rerank hit@5 | {retr.get('llm_rerank_hit5', 'N/A')} |",
        "",
        "---",
        "",
        "## Per-task-type Breakdown",
        "",
    ]

    # Task type distribution
    task_types: dict[str, dict[str, Any]] = {}
    for case in cases:
        tt = case.get("task_type", "unknown")
        if tt not in task_types:
            task_types[tt] = {"total": 0, "wrong": 0, "exact": 0, "partial": 0}
        task_types[tt]["total"] += 1
        label = case.get("answer_label", "")
        if label in task_types[tt]:
            task_types[tt][label] += 1

    lines.append(f"| Task Type | Total | exact | partial | wrong |")
    lines.append(f"|-----------|-------|-------|---------|-------|")
    for tt, counts in sorted(task_types.items()):
        lines.append(
            f"| {tt} | {counts['total']} | "
            f"{counts.get('exact', 0)} | {counts.get('partial', 0)} | "
            f"{counts.get('wrong', 0)} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Skipped cases (unlabeled / unanswerable – no retrieval scoring)
    skipped = [c for c in cases if c.get("retrieval", {}).get("skipped")]
    if skipped:
        lines.append("## Retrieval Skipped Cases  _(no gold labels)_")
        lines.append("")
        for case in skipped:
            lines.append(
                f"- `[{case['id']}]` label_status={case.get('label_status', '?')} | "
                f"{case['question'][:70]}"
            )
        lines.append("")

    # Failure cases – only labeled cases participate in retrieval failure analysis
    failures = [
        c for c in cases
        if c.get("label_status") not in {"unlabeled", "unanswerable"}
        and (
            c.get("answer_label") in {"wrong", "refuse_wrong", "clarify_wrong"}
            or not c.get("retrieval", {}).get("hit_at_5")
        )
    ]

    if failures:
        lines.append("## Failure Cases  _(labeled cases only)_")
        lines.append("")
        for case in failures:
            reason_parts = []
            if case.get("answer_label") in {"wrong", "refuse_wrong", "clarify_wrong"}:
                reason_parts.append(f"answer={case['answer_label']}")
            if not case.get("retrieval", {}).get("hit_at_5"):
                reason_parts.append("retrieval_miss")
            # P1: failure reason
            failure_reason = case.get("_failure_reason", "")
            if failure_reason:
                reason_parts.append(f"reason={failure_reason}")
            retr = case.get("retrieval", {})
            lines.append(f"### [{case['id']}] {case['question'][:60]}")
            lines.append("")
            lines.append(f"- Labels: {', '.join(reason_parts)}")
            lines.append(f"- Retrieved: {case.get('retrieved_chunk_ids', [])}")
            lines.append(f"- Gold: {case.get('gold_chunk_ids', [])}")
            lines.append(f"- Section hit@5: {retr.get('section_hit_at_5', 'N/A')}")
            lines.append(f"- Evidence semantic hit@5: {retr.get('evidence_semantic_hit_at_5', 'N/A')}")
            lines.append(f"- Evidence lexical hit@5: {retr.get('evidence_lexical_hit_at_5', 'N/A')}")
            answer_snippet = (case.get("final_answer") or "")[:120]
            lines.append(f"- Answer: {answer_snippet}")
            lines.append("")
    else:
        lines.append("## Failure Cases  _(none)_")
        lines.append("")

    # Degredation risk: low MRR among labeled cases only
    lines.append("---")
    lines.append("")
    lines.append("## Retrieval Low-Confidence Samples  _(labeled cases)_")
    lines.append("")
    low_retrieval = [
        c for c in cases
        if not c.get("retrieval", {}).get("skipped")
        and c.get("retrieval", {}).get("mrr", 1.0) < 0.5
    ]
    if low_retrieval:
        for case in low_retrieval:
            lines.append(
                f"- `[{case['id']}]` MRR={case['retrieval']['mrr']} | "
                f"{case['question'][:60]}"
            )
    else:
        lines.append("_none_")

    lines.append("")
    lines.append("---")
    lines.append(f"_Report generated at {report.get('generated_at', '')}_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diff report
# ---------------------------------------------------------------------------

def build_diff_report(
    base_report: dict[str, Any],
    new_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two run reports and produce a JSON diff.
    """
    base_cases = {c["id"]: c for c in base_report.get("cases", [])}
    new_cases = {c["id"]: c for c in new_report.get("cases", [])}

    improved: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    label_changes: list[dict[str, Any]] = []

    all_ids = sorted(set(base_cases.keys()) | set(new_cases.keys()))

    for case_id in all_ids:
        base_case = base_cases.get(case_id)
        new_case = new_cases.get(case_id)

        if base_case is None or new_case is None:
            continue

        # Answer label change
        base_label = base_case.get("answer_label", "")
        new_label = new_case.get("answer_label", "")
        if base_label != new_label:
            label_changes.append({
                "id": case_id,
                "question": new_case.get("question", ""),
                "base_label": base_label,
                "new_label": new_label,
            })
            # Check direction
            label_order = ["exact", "partial", "wrong", "refuse_wrong", "refuse_correct",
                           "clarify_correct", "clarify_wrong"]
            try:
                base_idx = label_order.index(base_label)
                new_idx = label_order.index(new_label)
                if new_idx < base_idx:
                    degraded.append({
                        "id": case_id,
                        "question": new_case.get("question", ""),
                        "change": f"{base_label} -> {new_label}",
                        "retrieval_before": base_case.get("retrieval", {}),
                        "retrieval_after": new_case.get("retrieval", {}),
                    })
                else:
                    improved.append({
                        "id": case_id,
                        "question": new_case.get("question", ""),
                        "change": f"{base_label} -> {new_label}",
                    })
            except ValueError:
                pass

        # Retrieval degradation
        base_mrr = base_case.get("retrieval", {}).get("mrr", 0.0)
        new_mrr = new_case.get("retrieval", {}).get("mrr", 0.0)
        if new_mrr < base_mrr - 0.01:
            degraded.append({
                "id": case_id,
                "question": new_case.get("question", ""),
                "change": f"retrieval MRR {base_mrr} -> {new_mrr}",
                "retrieval_before": base_case.get("retrieval", {}),
                "retrieval_after": new_case.get("retrieval", {}),
            })

    base_summary = base_report.get("summary", {})
    new_summary = new_report.get("summary", {})

    return {
        "base_run_id": base_report.get("run_id", ""),
        "new_run_id": new_report.get("run_id", ""),
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_samples": len(all_ids),
            "improved_count": len(improved),
            "degraded_count": len(degraded),
            "label_changed_count": len(label_changes),
            "retrieval": {
                "base_hit_at_1": base_summary.get("retrieval", {}).get("hit_at_1", 0.0),
                "new_hit_at_1": new_summary.get("retrieval", {}).get("hit_at_1", 0.0),
                "base_mrr": base_summary.get("retrieval", {}).get("mrr", 0.0),
                "new_mrr": new_summary.get("retrieval", {}).get("mrr", 0.0),
            },
            "answer": {
                "base_exact": base_summary.get("answer", {}).get("exact", 0),
                "new_exact": new_summary.get("answer", {}).get("exact", 0),
                "base_partial": base_summary.get("answer", {}).get("partial", 0),
                "new_partial": new_summary.get("answer", {}).get("partial", 0),
            },
        },
        "improved": improved,
        "degraded": degraded,
        "label_changes": label_changes,
    }


def build_markdown_diff(diff: dict[str, Any]) -> str:
    """Human-readable markdown diff report."""
    summary = diff.get("summary", {})
    lines = [
        "# Run Comparison Report",
        "",
        f"**Base run**: `{diff.get('base_run_id', '')}`",
        f"**New run**: `{diff.get('new_run_id', '')}`",
        f"**Compared at**: {diff.get('compared_at', '')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"| | Base | New | Delta |",
        f"|-|------|-----|-------|",
        f"| Total samples | {summary.get('total_samples', 0)} | {summary.get('total_samples', 0)} | — |",
        f"| Hit@1 | {summary.get('retrieval', {}).get('base_hit_at_1', 0.0)} | "
        f"{summary.get('retrieval', {}).get('new_hit_at_1', 0.0)} | "
        f"{_delta(summary.get('retrieval', {}).get('new_hit_at_1', 0.0) - summary.get('retrieval', {}).get('base_hit_at_1', 0.0))} |",
        f"| MRR | {summary.get('retrieval', {}).get('base_mrr', 0.0)} | "
        f"{summary.get('retrieval', {}).get('new_mrr', 0.0)} | "
        f"{_delta(summary.get('retrieval', {}).get('new_mrr', 0.0) - summary.get('retrieval', {}).get('base_mrr', 0.0))} |",
        f"| exact | {summary.get('answer', {}).get('base_exact', 0)} | "
        f"{summary.get('answer', {}).get('new_exact', 0)} | "
        f"{_delta(summary.get('answer', {}).get('new_exact', 0) - summary.get('answer', {}).get('base_exact', 0))} |",
        "",
        f"**Improved**: {summary.get('improved_count', 0)}",
        f"**Degraded**: {summary.get('degraded_count', 0)}",
        f"**Label changes**: {summary.get('label_changed_count', 0)}",
        "",
        "---",
        "",
    ]

    degraded = diff.get("degraded", [])
    if degraded:
        lines.append("## Degraded Samples  ⚠️")
        lines.append("")
        for item in degraded:
            lines.append(f"### `[{item['id']}]` {item.get('question', '')[:60]}")
            lines.append(f"- Change: {item.get('change', '')}")
            r_before = item.get("retrieval_before", {})
            r_after = item.get("retrieval_after", {})
            lines.append(
                f"- Retrieval: MRR {r_before.get('mrr', 0.0)} → {r_after.get('mrr', 0.0)}, "
                f"Hit@5 {r_before.get('hit_at_5', False)} → {r_after.get('hit_at_5', False)}"
            )
            lines.append("")
    else:
        lines.append("## Degraded Samples  _(none)_")
        lines.append("")

    improved = diff.get("improved", [])
    if improved:
        lines.append("## Improved Samples")
        lines.append("")
        for item in improved:
            lines.append(f"- `[{item['id']}]` {item.get('question', '')[:60]} — {item.get('change', '')}")
        lines.append("")

    label_changes = diff.get("label_changes", [])
    if label_changes:
        lines.append("---")
        lines.append("")
        lines.append("## Answer Label Changes")
        lines.append("")
        for lc in label_changes:
            lines.append(
                f"- `[{lc['id']}]` {lc.get('question', '')[:50]} — "
                f"**{lc.get('base_label', '')}** → **{lc.get('new_label', '')}**"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_retrieval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate retrieval metrics over cases where retrieval was NOT skipped.
    Skipped cases (label_status=unlabeled/unanswerable) are excluded.
    """
    labeled = [c for c in cases if not c.get("retrieval_skipped")]
    n = len(labeled)
    if n == 0:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "note": "no labeled retrieval cases",
        }

    hit1 = sum(1 for c in labeled if c.get("retrieval_hit_at_1")) / n
    hit3 = sum(1 for c in labeled if c.get("retrieval_hit_at_3")) / n
    hit5 = sum(1 for c in labeled if c.get("retrieval_hit_at_5")) / n
    mrr_sum = sum(c.get("retrieval_mrr", 0.0) for c in labeled) / n
    recall_sum = sum(c.get("retrieval_recall_at_5", 0.0) for c in labeled) / n

    # P0: page-level (strict + relaxed)
    page_hit5 = sum(1 for c in labeled if c.get("retrieval_page_strict_hit_at_5")) / n if n > 0 else 0.0
    page_relaxed_hit5 = sum(1 for c in labeled if c.get("retrieval_page_relaxed_hit_at_5")) / n if n > 0 else 0.0

    # P0: section-level
    section_hit5 = sum(1 for c in labeled if c.get("retrieval_section_hit_at_5")) / n if n > 0 else 0.0

    # P0: evidence semantic
    ev_sem_hit5 = sum(1 for c in labeled if c.get("evidence_semantic_hit_at_5")) / n if n > 0 else 0.0
    ev_lex_hit5 = sum(1 for c in labeled if c.get("evidence_lexical_hit_at_5")) / n if n > 0 else 0.0

    return {
        "hit_at_1": round(hit1, 4),
        "hit_at_3": round(hit3, 4),
        "hit_at_5": round(hit5, 4),
        "recall_at_5": round(recall_sum, 4),
        "mrr": round(mrr_sum, 4),
        "labeled_cases": n,
        # P0
        "page_strict_hit_at_5": round(page_hit5, 4),
        "page_relaxed_hit_at_5": round(page_relaxed_hit5, 4),
        "section_hit_at_5": round(section_hit5, 4),
        "evidence_semantic_hit_at_5": round(ev_sem_hit5, 4),
        "evidence_lexical_hit_at_5": round(ev_lex_hit5, 4),
        # V4: LLM reranker
        "llm_rerank_hit5": round(sum(1 for c in labeled if c.get("llm_rerank_applied") and c.get("section_hit_at_5")) / n, 4) if n > 0 else 0.0,
        "llm_rerank_fallback_rate": round(sum(1 for c in labeled if c.get("llm_rerank_fallback")) / n, 4) if n > 0 else 0.0,
    }


def _aggregate_answer(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [
        "exact", "partial", "wrong",
        "refuse_correct", "refuse_wrong",
        "clarify_correct", "clarify_wrong",
    ]
    counts: dict[str, int] = {l: 0 for l in labels}
    for c in cases:
        label = c.get("answer_label", "")
        if label in counts:
            counts[label] += 1
    # V4 metrics
    n = len(cases)
    answer_supported = sum(1 for c in cases if c.get("v4_answer_supported") is True)
    was_refined = sum(1 for c in cases if c.get("v4_was_refined") is True)
    numeric_consistent = sum(1 for c in cases if c.get("v4_numeric_consistent") is True)
    citation_adequate = sum(1 for c in cases if c.get("v4_citation_adequate") is True)
    verifier_fallback = sum(1 for c in cases if c.get("v4_verifier_fallback"))
    refine_fallback = sum(1 for c in cases if c.get("v4_refine_fallback"))
    return {
        "label_counts": counts,
        # V4 pipeline
        "answer_supported_rate": round(answer_supported / n, 4) if n > 0 else 0.0,
        "answer_refined_rate": round(was_refined / n, 4) if n > 0 else 0.0,
        "numeric_verifier_pass_rate": round(numeric_consistent / n, 4) if n > 0 else 0.0,
        "citation_adequate_rate": round(citation_adequate / n, 4) if n > 0 else 0.0,
        # V4 fallback rates
        "verifier_fallback_rate": round(verifier_fallback / n, 4) if n > 0 else 0.0,
        "refine_fallback_rate": round(refine_fallback / n, 4) if n > 0 else 0.0,
    }


def _delta(v: float) -> str:
    if v > 0:
        return f"+{v:.4f}"
    if v < 0:
        return f"{v:.4f}"
    return "0.0000"
