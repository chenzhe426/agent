#!/usr/bin/env python3
"""
run_financebench_eval.py – Run eval on FinanceBench-format dataset.

Reads FinanceBench JSONL (financebench_v1_subset_3docs.jsonl), runs the RAG
pipeline via EvalAdapter for each sample, and writes a JSONL run file that
score_eval.py can consume.

Usage:
    python -m evals.scripts.run_financebench_eval \
        --dataset evals/data/financebench_v1_subset_3docs.jsonl \
        --output evals/runs/run_$(date +%Y%m%d_%H%M%S).jsonl

Cache:
    Results are cached per question in --cache-dir (default: evals/cache/).
    Use --no-cache to skip cache and re-run all samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.adapters import EvalAdapter

# Default concurrency for eval runs
DEFAULT_WORKERS = int(__import__("os").getenv("EVAL_WORKERS", "1"))

# ShareGPT system prompt for financial QA
_SHAREGPT_SYSTEM_PROMPT = (
    "You are a professional financial analyst assistant. "
    "Answer the question based ONLY on the evidence provided below. "
    "Cite specific numbers and facts from the evidence when available. "
    "Do not make up information not present in the evidence."
)


def _result_to_sharegpt(result: dict) -> dict:
    """Convert a run result dict to ShareGPT format."""
    # Format evidence chunks
    evidence_blocks = []
    for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
        text = chunk.get("chunk_text", "")
        page = chunk.get("page_start") or chunk.get("page_end") or "N/A"
        title = chunk.get("title", "")
        section = chunk.get("section_title", "")
        evidence_blocks.append(
            f"[Evidence {i}] ({title}, p{page}, {section}):\n{text}"
        )

    evidence_text = "\n\n---\n\n".join(evidence_blocks)
    user_content = (
        f"Question: {result.get('query', '')}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Please answer the question based on the evidence above."
    )

    return {
        "id": result.get("id", ""),
        "messages": [
            {"role": "system", "content": _SHAREGPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": result.get("final_answer", "")},
        ],
    }


def _cache_key(question: str, top_k: int, doc_filter: int | None = None) -> str:
    """Cache key includes question hash + top_k + doc_filter."""
    data = f"{top_k}:{doc_filter}:{question}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _get_cache_path(cache_dir: Path, question: str, top_k: int, doc_filter: int | None = None) -> Path:
    h = _cache_key(question, top_k, doc_filter)
    return cache_dir / f"{h}.json"


def _load_cache(cache_dir: Path) -> dict[str, dict]:
    """Load all cached results into a dict keyed by cache key (question hash + top_k + doc_filter)."""
    if not cache_dir.exists():
        return {}
    cache = {}
    for f in cache_dir.glob("*.json"):
        try:
            cached = json.loads(f.read_text(encoding="utf-8"))
            cache[f.stem] = cached
        except Exception:
            pass
    return cache


def _save_to_cache(cache_dir: Path, question: str, top_k: int, doc_filter: int | None, result: dict) -> None:
    """Save a single result to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _get_cache_path(cache_dir, question, top_k, doc_filter)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def load_financebench_dataset(path: Path) -> list[dict]:
    """Load FinanceBench-format JSONL."""
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


# Cache for doc_name -> internal doc_id mapping
_doc_id_cache: dict[str, int] = {}


def _ensure_doc_mapping() -> None:
    """Build doc_name -> internal doc_id cache from KB."""
    if _doc_id_cache:
        return
    from app.db.bootstrap import init_db
    from app.db import get_all_documents
    init_db()
    docs = get_all_documents()
    for d in docs:
        _doc_id_cache[d["title"]] = d["id"]


def doc_name_to_id(title: str) -> int | None:
    """Map document title to internal document_id."""
    _ensure_doc_mapping()
    return _doc_id_cache.get(title)


# Company keywords for doc_filter detection (same as EvalAdapter)
_COMPANY_KEYWORDS = {
    "AMD": "AMD_2022_10K",
    "BOEING": "BOEING_2022_10K",
    "AMERICAN EXPRESS": "AMERICANEXPRESS_2022_10K",
    "AMEX": "AMERICANEXPRESS_2022_10K",
}


def _detect_company_doc_id(query: str) -> int | None:
    """Detect company name from query and return corresponding doc_id."""
    _ensure_doc_mapping()
    query_upper = query.upper()
    for keyword, doc_title in _COMPANY_KEYWORDS.items():
        if keyword in query_upper:
            doc_id = _doc_id_cache.get(doc_title)
            if doc_id is not None:
                return doc_id
    return None


def run_one_sample(sample: dict, adapter: EvalAdapter, cache: dict, use_cache: bool = True, cache_dir: Path | None = None, top_k: int = 5) -> dict:
    """Run RAG pipeline for one FinanceBench sample and merge gold info."""
    eval_id = sample.get("id", "") or sample.get("eval_id", "")
    question = sample.get("question", "")
    print(f"[DEBUG] ====== Processing sample: {eval_id} ======")
    print(f"[DEBUG] Question: {question[:100]}...")

    # Detect company for doc_filter (same logic as adapter)
    doc_filter = _detect_company_doc_id(question)
    print(f"[DEBUG] Detected doc_filter: {doc_filter}")

    # Build base result structure (gold info always needed)
    result = {
        "id": eval_id,
        "query": question,
        "final_answer": "",
        "retrieved_chunks": [],
        "latency_ms": 0.0,
        "stage_timings": {},
        "total_pipeline_ms": 0.0,
        "error": "",
        # Gold info needed by score_eval
        "gold_doc_name": sample.get("gold_doc_name", ""),
        "gold_pages": sample.get("gold_pages", []),
        "gold_answer": sample.get("gold_answer", ""),
        "gold_chunk_ids": sample.get("gold_chunk_ids") or [],
        "gold_chunk_status": sample.get("gold_chunk_status", "unresolved"),
        "gold_chunk_match_method": sample.get("gold_chunk_match_method", ""),
        # Map gold_doc_name -> internal doc_id for document-level scoring
        "retrieval": {
            "gold_doc_ids": [],
            "gold_chunk_ids": sample.get("gold_chunk_ids") or [],
            "label_status": "labeled_doc" if sample.get("gold_doc_name") else "unlabeled",
        },
        "answer": {
            "gold_answer": sample.get("gold_answer", ""),
            "must_include": sample.get("must_include", []),
        },
        "evaluation": {
            "expected_behavior": sample.get("expected_behavior", "answer"),
        },
    }

    # Convert gold_doc_name to gold_doc_ids
    gold_doc_name = sample.get("gold_doc_name", "")
    if gold_doc_name:
        doc_id = doc_name_to_id(gold_doc_name)
        if doc_id is not None:
            result["retrieval"]["gold_doc_ids"] = [doc_id]

    # Check cache first (include doc_filter in cache key)
    qhash = _cache_key(question, top_k, doc_filter)
    if use_cache and qhash in cache:
        print(f"[DEBUG CACHE HIT] {eval_id} -> using cached result")
        cached = cache[qhash]
        # Merge gold info into cached result
        cached["id"] = eval_id
        # Rebuild retrieval/answer/evaluation since they were stripped when caching
        cached["retrieval"] = result.get("retrieval", {"gold_doc_ids": [], "gold_chunk_ids": [], "label_status": "unlabeled"})
        cached["answer"] = result.get("answer", {"gold_answer": "", "must_include": []})
        cached["evaluation"] = result.get("evaluation", {"expected_behavior": "answer"})
        return cached

    try:
        print(f"[DEBUG] Calling adapter.answer for: {eval_id}")
        answer_result = adapter.answer(question, conversation_history=None)
        print(f"[DEBUG] adapter.answer returned for: {eval_id}")
        result["retrieved_chunks"] = answer_result.get("retrieved_chunks", [])
        result["final_answer"] = answer_result.get("final_answer", "")
        result["latency_ms"] = answer_result.get("latency_ms", 0.0)
        result["stage_timings"] = answer_result.get("stage_timings", {})
        result["total_pipeline_ms"] = answer_result.get("total_pipeline_ms", 0.0)
        print(f"[DEBUG] Got {len(result['retrieved_chunks'])} chunks, answer length={len(result['final_answer'])}")

        # Save to cache (strip gold info since it's per-sample)
        if cache_dir:
            cacheable = {k: v for k, v in result.items() if k not in ("retrieval", "answer", "evaluation")}
            _save_to_cache(cache_dir, question, top_k, doc_filter, cacheable)

        print(f"[DEBUG] ====== Done sample: {eval_id} ======")
    except Exception as e:
        import traceback
        print(f"[DEBUG] ERROR for {eval_id}: {e}")
        traceback.print_exc()
        result["error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval on FinanceBench dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to FinanceBench JSONL dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for run JSONL",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        help="Number of concurrent workers (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "evals" / "cache"),
        type=str,
        help="Directory for result cache (default: evals/cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache, re-run all samples from scratch",
    )
    parser.add_argument(
        "--llm-timeout",
        default=180,
        type=int,
        help="Timeout for each LLM call in seconds (default: 180)",
    )
    args = parser.parse_args()

    # Load dataset
    samples = load_financebench_dataset(args.dataset)
    print(f"[run_financebench_eval] Loaded {len(samples)} samples")
    print(f"[run_financebench_eval] Using {args.workers} concurrent workers")
    print(f"[DEBUG] First sample: {samples[0] if samples else 'NONE'}")

    # Load cache
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    if use_cache:
        cache = _load_cache(cache_dir)
        n_cached = len([c for c in cache.values() if c.get("final_answer")])
        print(f"[run_financebench_eval] Cache: {n_cached} cached results in {cache_dir}")
    else:
        cache = {}
        print(f"[run_financebench_eval] Cache disabled")

    # Run concurrently
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results: list[dict] = [None] * len(samples)
    errors: list[str] = []

    def worker(i: int, sample: dict) -> tuple[int, dict]:
        print(f"[DEBUG WORKER {i}] Starting...")
        adapter = EvalAdapter(mode="internal", top_k=args.top_k, llm_timeout=args.llm_timeout)
        result = run_one_sample(sample, adapter, cache, use_cache=use_cache, cache_dir=cache_dir, top_k=args.top_k)
        print(f"[DEBUG WORKER {i}] Finished")
        return i, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, i, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            try:
                i, result = future.result()
                results[i] = result
                print(f"[DEBUG MAIN] Completed future for index {i}")
            except Exception as e:
                idx = futures[future]
                errors.append(f"[{idx}] {type(e).__name__}: {e}")
                results[idx] = {"id": samples[idx].get("eval_id", ""), "error": str(e)}

    # Print status in original order
    for i, result in enumerate(results):
        if result is None:
            continue
        status = "✓" if not result.get("error") else "✗"
        has_chunks = len(result.get("retrieved_chunks", [])) > 0
        has_answer = bool(result.get("final_answer"))
        print(
            f"  [{i+1}/{len(samples)}] {status} [{result.get('id', '?')}] "
            f"chunks={has_chunks} answer={has_answer}"
        )

    if errors:
        print(f"\n  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")

    # Write JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[run_financebench_eval] Written {len(results)} results to: {args.output}")

    # Write ShareGPT format alongside
    sharegpt_output = args.output.with_suffix(".sharegpt.jsonl")
    with sharegpt_output.open("w", encoding="utf-8") as f:
        for r in results:
            if r.get("final_answer"):
                sg = _result_to_sharegpt(r)
                f.write(json.dumps(sg, ensure_ascii=False) + "\n")

    print(f"[run_financebench_eval] Written ShareGPT format to: {sharegpt_output}")

    # Summary
    n_total = len(results)
    n_ok = sum(1 for r in results if not r.get("error"))
    n_with_chunks = sum(1 for r in results if r.get("retrieved_chunks"))
    n_with_answer = sum(1 for r in results if r.get("final_answer"))
    print(f"\n=== Summary ===")
    print(f"  Total:       {n_total}")
    print(f"  Success:      {n_ok}")
    print(f"  With chunks:  {n_with_chunks}")
    print(f"  With answer: {n_with_answer}")


if __name__ == "__main__":
    main()
