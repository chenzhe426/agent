#!/usr/bin/env python3
"""
Stage 4: Build knowledge base for eval documents.

Reads a required-documents manifest (from Stage 3), imports and indexes
each PDF into the KB.

Usage:
    python -m evals.scripts.build_eval_knowledge_base \
        --dataset financebench_v1_subset_3docs

Output:
    Prints import statistics (doc count, chunk counts per doc, failures).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build KB for eval documents")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset basename (e.g. financebench_v1_subset_3docs) to find required-docs manifest",
    )
    parser.add_argument(
        "--docs-file",
        type=Path,
        default=None,
        help="Override path to required-docs manifest",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for parse+index (default: 4). "
             "Note: pdfplumber's C extension is protected by a global lock, "
             "so PDF parsing is serialized but chunking/embedding per document runs in parallel.",
    )
    args = parser.parse_args()

    # Find the required-docs file
    evals_data = ROOT / "evals" / "data"
    if args.docs_file:
        required_docs_path = args.docs_file
    else:
        required_docs_path = evals_data / f"financebench_required_{args.dataset}.jsonl"
        if not required_docs_path.exists():
            # Try alternate naming
            required_docs_path = evals_data / f"financebench_required_{args.dataset}_docs.jsonl"

    if not required_docs_path.exists():
        print(f"[build_kb] ERROR: required-docs file not found: {required_docs_path}", file=sys.stderr)
        sys.exit(1)

    # Load required docs
    required_docs: list[dict] = []
    with required_docs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                required_docs.append(json.loads(line))

    print(f"[build_kb] Loading {len(required_docs)} documents from: {required_docs_path}")

    # Init DB
    from app.db.bootstrap import init_db
    from app.db import get_all_documents, insert_document, get_document_by_id
    from app.services.chunk_service import index_document
    from app.ingestion.pipeline import parse_document
    import app.services.common as common

    init_db()

    # Check what's already indexed
    existing_docs = get_all_documents()
    existing_titles = {d["title"]: d["id"] for d in existing_docs}
    print(f"[build_kb] Currently indexed docs: {len(existing_titles)}")

    stats = {
        "total": len(required_docs),
        "already_indexed": 0,
        "imported": 0,
        "failed": 0,
        "skipped": 0,
        "chunk_counts": {},
        "failures": [],
        "lock": __import__("threading").Lock(),
    }

    def process_one(rec: dict) -> None:
        """Full parse→insert→index pipeline for one document."""
        from app.db import get_chunks_by_document_id

        doc_name = rec["doc_name"]
        pdf_path = Path(rec["source_path_or_url"])

        if not pdf_path.exists():
            with stats["lock"]:
                stats["skipped"] += 1
                print(f"  [SKIP] {doc_name}: PDF not found at {pdf_path}")
            return

        if doc_name in existing_titles:
            doc_id = existing_titles[doc_name]
            chunks = get_chunks_by_document_id(doc_id)
            with stats["lock"]:
                stats["already_indexed"] += 1
                stats["chunk_counts"][doc_name] = len(chunks)
                print(f"  [EXIST] {doc_name}: already indexed, {len(chunks)} chunks")
            return

        print(f"  [IMPORT] {doc_name}...")
        try:
            t0 = time.time()
            # 1. Parse PDF (protected by pdfplumber's global RLock)
            parsed = parse_document(str(pdf_path))
            print(f"    Parsed: {len(parsed.blocks)} blocks in {time.time()-t0:.1f}s")

            # 2. Insert document into DB
            payload = {
                "title": doc_name,
                "file_path": str(pdf_path),
                "file_type": "pdf",
                "source_type": "folder_import",
                "content": "",
                "blocks_json": json.dumps(parsed.blocks),
                "metadata_json": {
                    "parser_used": parsed.metadata.get("parser_used"),
                    "quality_score": parsed.metadata.get("quality_score"),
                    "page_count": parsed.metadata.get("page_count"),
                    "table_count": parsed.metadata.get("table_count"),
                    "heading_count": parsed.metadata.get("heading_count"),
                },
            }
            doc_id = insert_document(**payload)
            print(f"    Inserted: doc_id={doc_id}")

            # 3. Index (chunking + embedding + Qdrant upsert)
            result = index_document(doc_id)
            elapsed = time.time() - t0
            chunk_count = result.get("chunk_count", 0)
            with stats["lock"]:
                stats["imported"] += 1
                stats["chunk_counts"][doc_name] = chunk_count
                print(f"    Indexed: {chunk_count} chunks in {elapsed:.1f}s")

        except Exception as e:
            with stats["lock"]:
                stats["failed"] += 1
                stats["failures"].append((doc_name, str(e)))
                print(f"  [FAIL] {doc_name}: {e}", file=sys.stderr)

    # Parallel processing: each worker handles full parse→insert→index for one doc
    # pdfplumber's global RLock serializes PDF parsing, but chunking/embedding
    # per worker overlaps with other workers' parsing, giving ~2-3x speedup.
    workers = min(args.workers, len(required_docs))
    print(f"[build_kb] Starting {workers} parallel workers...")

    with __import__("concurrent.futures").ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, rec): rec for rec in required_docs}
        for future in __import__("concurrent.futures").as_completed(futures):
            try:
                future.result()
            except Exception as e:
                rec = futures[future]
                with stats["lock"]:
                    stats["failed"] += 1
                    stats["failures"].append((rec["doc_name"], str(e)))

    # Print summary
    print(f"\n=== KB Build Summary ===")
    print(f"  Total:      {stats['total']}")
    print(f"  Already:    {stats['already_indexed']}")
    print(f"  Imported:  {stats['imported']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"\n  Chunk counts:")
    for doc, cnt in stats["chunk_counts"].items():
        print(f"    {doc}: {cnt} chunks")
    if stats["failures"]:
        print(f"\n  Failures:")
        for doc, err in stats["failures"]:
            print(f"    {doc}: {err}")


if __name__ == "__main__":
    main()
