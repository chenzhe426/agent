#!/usr/bin/env python3
"""
Stage 3: Prepare document list from eval subset.

Reads a eval subset JSONL (from Stage 2), extracts unique doc_names,
maps to local PDF paths, and outputs a document manifest.

Usage:
    python -m evals.scripts.prepare_eval_documents \
        --dataset evals/data/financebench_v1_subset_3docs

Output:
    evals/data/financebench_required_documents_3docs.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FINANCEBENCH_DATA = ROOT / "data" / "financebench"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare required documents for eval")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the eval subset JSONL (e.g. financebench_v1_subset_3docs.jsonl)",
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=FINANCEBENCH_DATA / "data" / "financebench_document_information.jsonl",
        help="Path to document_information JSONL",
    )
    parser.add_argument(
        "--pdfs",
        type=Path,
        default=FINANCEBENCH_DATA / "pdfs",
        help="Path to PDFs directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: evals/data/financebench_required_<basename>.jsonl)",
    )
    args = parser.parse_args()

    # Load doc_info for metadata
    doc_info: dict[str, dict] = {}
    with args.info.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                dn = rec.get("doc_name", "")
                if dn:
                    doc_info[dn] = rec
            except json.JSONDecodeError:
                continue

    # Load eval subset
    samples = []
    with args.dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Extract unique doc_names
    doc_names_used = sorted(set(s.get("gold_doc_name", "") for s in samples))

    # Build document manifest
    records: list[dict] = []
    missing_pdfs: list[str] = []

    for doc_name in doc_names_used:
        info = doc_info.get(doc_name, {})
        pdf_path = args.pdfs / f"{doc_name}.pdf"
        source = str(pdf_path) if pdf_path.exists() else info.get("doc_link", "")

        if not pdf_path.exists():
            missing_pdfs.append(doc_name)

        records.append({
            "doc_name": doc_name,
            "company": info.get("company", ""),
            "doc_type": info.get("doc_type", ""),
            "doc_period": info.get("doc_period", ""),
            "gics_sector": info.get("gics_sector", ""),
            "source_path_or_url": source,
            "pdf_available": pdf_path.exists(),
        })

    # Determine output path
    output_path = args.output
    if output_path is None:
        stem = args.dataset.stem.replace("subset_", "")
        output_path = args.dataset.parent / f"financebench_required_{stem}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[prepare_docs] Output: {output_path}")
    print(f"  Total documents: {len(records)}")
    print(f"  PDFs available: {sum(1 for r in records if r['pdf_available'])}")
    print(f"  Missing PDFs:   {len(missing_pdfs)}")
    if missing_pdfs:
        print(f"  Missing: {missing_pdfs}")
    for rec in records:
        status = "✓" if rec["pdf_available"] else "?"
        print(f"  [{status}] {rec['doc_name']} ({rec['company']}, {rec['doc_period']})")


if __name__ == "__main__":
    main()
