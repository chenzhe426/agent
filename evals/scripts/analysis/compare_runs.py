#!/usr/bin/env python3
"""
compare_runs.py – Compare two evaluation run reports and produce a diff.

Usage:
    python evals/scripts/compare_runs.py \
        --base evals/reports/run_base.json \
        --new evals/reports/run_exp.json

Outputs:
    evals/reports/compare_<base>_vs_<new>.json   – JSON diff
    evals/reports/compare_<base>_vs_<new>.md     – Markdown diff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.report import build_diff_report, build_markdown_diff, load_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation run reports")
    parser.add_argument(
        "--base",
        required=True,
        help="Path to the base (reference) run JSON report",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="Path to the new (experiment) run JSON report",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path prefix (default: compare_<base>_vs_<new>)",
    )
    args = parser.parse_args()

    base_report = load_report(args.base)
    new_report = load_report(args.new)

    diff = build_diff_report(base_report, new_report)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        base_name = Path(args.base).stem
        new_name = Path(args.new).stem
        out_path = Path(args.base).parent / f"compare_{base_name}_vs_{new_name}"

    json_out = Path(str(out_path) + ".json")
    md_out = Path(str(out_path) + ".md")

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(diff, f, ensure_ascii=False, indent=2)
    print(f"[compare_runs] JSON diff written to: {json_out}")

    md_text = build_markdown_diff(diff)
    with md_out.open("w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[compare_runs] Markdown diff written to: {md_out}")

    # Print a brief console summary
    summary = diff.get("summary", {})
    s = summary
    print("\n=== Comparison Summary ===")
    print(f"  Samples compared  : {s.get('total_samples', 0)}")
    print(f"  Improved         : {s.get('improved_count', 0)}")
    print(f"  Degraded         : {s.get('degraded_count', 0)} ⚠️" if s.get("degraded_count", 0) > 0 else f"  Degraded         : {s.get('degraded_count', 0)}")
    print(f"  Label changes    : {s.get('label_changed_count', 0)}")
    print(f"\n  Retrieval MRR    : {s.get('retrieval', {}).get('base_mrr', 0.0):.4f} → {s.get('retrieval', {}).get('new_mrr', 0.0):.4f}")
    print(f"  Answer exact     : {s.get('answer', {}).get('base_exact', 0)} → {s.get('answer', {}).get('new_exact', 0)}")

    if s.get("degraded_count", 0) > 0:
        print("\n  ⚠️  Degraded sample IDs:")
        for item in diff.get("degraded", []):
            print(f"    - [{item['id']}] {item.get('change', '')}: {item.get('question', '')[:60]}")


if __name__ == "__main__":
    main()
