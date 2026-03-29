"""
Scoring logic for retrieval and answer evaluation.

All scoring is rule-based (no LLM dependency).
Future extension points are clearly marked.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from evals.utils.dataset import EvalSample


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

class RetrievalScorer:
    """
    Compute retrieval-level metrics.

    Metrics computed:
      - hit_at_1 / hit_at_3 / hit_at_5
      - recall_at_5
      - mrr (Mean Reciprocal Rank)

    Matching logic:
      1. Try to match chunk_id first.
      2. If use_doc_level_fallback=True and chunk_id is not available,
         fall back to document_id matching.
    """

    def __init__(self, use_doc_level_fallback: bool = True) -> None:
        self.use_doc_level_fallback = use_doc_level_fallback

    def _build_gold_sets(self, sample: EvalSample) -> tuple[set[int], set[int]]:
        gold_chunks = set(int(x) for x in sample.retrieval.gold_chunk_ids if x is not None)
        gold_docs = set(int(x) for x in sample.retrieval.gold_doc_ids if x is not None)
        return gold_chunks, gold_docs

    def _retrieve_id(self, chunk: dict[str, Any]) -> Optional[tuple[str, int]]:
        """
        Return (level, id) where level is 'chunk' or 'doc'.
        Prefers chunk_id if available.
        """
        cid = chunk.get("chunk_id")
        if cid is not None:
            try:
                return ("chunk", int(cid))
            except (TypeError, ValueError):
                pass
        did = chunk.get("document_id")
        if did is not None and self.use_doc_level_fallback:
            try:
                return ("doc", int(did))
            except (TypeError, ValueError):
                pass
        return None

    def score(self, sample: EvalSample, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute all retrieval metrics for a single sample.

        Parameters
        ----------
        sample    : EvalSample with gold labels
        retrieved : list of chunk dicts as returned by EvalAdapter.retrieve()

        Returns
        -------
        dict with keys:
          - hit_at_1, hit_at_3, hit_at_5 : bool
          - recall_at_5                  : float (0.0–1.0)
          - mrr                          : float (0.0–1.0)
          - hit_chunk_ids                : list[int]  – IDs that hit gold
          - miss_chunk_ids               : list[int]  – retrieved IDs not in gold
        """
        gold_chunks, gold_docs = self._build_gold_sets(sample)
        all_gold = gold_chunks or gold_docs
        if not all_gold:
            return self._empty_result()

        hit_chunk_ids: list[int] = []
        miss_chunk_ids: list[int] = []

        reciprocal_rank = 0.0
        found_rr = False

        for rank, chunk in enumerate(retrieved[:5], start=1):
            hit = False
            level_id = self._retrieve_id(chunk)
            if level_id is None:
                continue

            level, rid = level_id
            if level == "chunk" and rid in gold_chunks:
                hit = True
            elif level == "doc" and rid in gold_docs:
                hit = True

            if hit:
                hit_chunk_ids.append(rid)
                if not found_rr:
                    reciprocal_rank = 1.0 / rank
                    found_rr = True
            else:
                miss_chunk_ids.append(rid)

        top5 = retrieved[:5]
        hits_at_1 = int(len(top5) >= 1 and self._is_hit(top5[0], gold_chunks, gold_docs))
        hits_at_3 = int(any(self._is_hit(c, gold_chunks, gold_docs) for c in top5[:3]))
        hits_at_5 = int(any(self._is_hit(c, gold_chunks, gold_docs) for c in top5))

        recall = len(hit_chunk_ids) / len(all_gold) if all_gold else 0.0

        return {
            "hit_at_1": bool(hits_at_1),
            "hit_at_3": bool(hits_at_3),
            "hit_at_5": bool(hits_at_5),
            "recall_at_5": round(recall, 4),
            "mrr": round(reciprocal_rank, 4),
            "hit_chunk_ids": hit_chunk_ids,
            "miss_chunk_ids": miss_chunk_ids,
        }

    def _is_hit(self, chunk: dict[str, Any], gold_chunks: set[int], gold_docs: set[int]) -> bool:
        level_id = self._retrieve_id(chunk)
        if level_id is None:
            return False
        level, rid = level_id
        if level == "chunk" and rid in gold_chunks:
            return True
        if level == "doc" and rid in gold_docs:
            return True
        return False

    def _empty_result(self) -> dict[str, Any]:
        return {
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": False,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "hit_chunk_ids": [],
            "miss_chunk_ids": [],
        }


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

_REFUSE_PHRASES = {
    "没有足够信息",
    "无法确认",
    "当前未看到",
    "不能确认",
    "没有证据",
    "信息不足",
    "无法从",
    "知识库中未找到",
    "暂未找到",
    "没有找到相关",
    "not enough information",
    "cannot confirm",
    "insufficient information",
    "no relevant",
    "知识库里没有",
    "未提供",
}

_CLARIFY_PHRASES = {
    "你想具体看",
    "你指的是哪个",
    "能具体说明",
    "能否补充",
    "请提供更多",
    "请说明",
    "具体是哪部分",
    "具体指",
}


class AnswerScorer:
    """
    Rule-based answer scorer.

    Supports three expected_behavior modes:
      - answer   : standard question answering
      - refuse   : system should decline / express uncertainty
      - clarify  : system should ask a clarifying question

    Per-mode rules are described in the score() method.
    """

    def __init__(self, normalize_text: bool = True) -> None:
        self.normalize_text = normalize_text

    def score(self, sample: EvalSample, result: dict[str, Any]) -> dict[str, Any]:
        """
        Compute answer-level metrics for a single sample.

        Parameters
        ----------
        sample : EvalSample
        result : dict with keys {final_answer, retrieved_chunks, ...}

        Returns
        -------
        dict with keys:
          - answer_label   : str  (exact | partial | wrong | refuse_correct |
                                  refuse_wrong | clarify_correct | clarify_wrong)
          - must_include_hit_ratio  : float
          - must_not_include_violations : list[str]
          - notes          : str
        """
        expected = sample.evaluation.expected_behavior
        answer_text = result.get("final_answer", "")
        norm_answer = self._norm(answer_text)

        if expected == "answer":
            return self._score_answer(sample, answer_text, norm_answer)
        elif expected == "refuse":
            return self._score_refuse(sample, answer_text, norm_answer)
        elif expected == "clarify":
            return self._score_clarify(sample, answer_text, norm_answer)
        else:
            return {"answer_label": "wrong", "must_include_hit_ratio": 0.0,
                    "must_not_include_violations": [], "notes": f"unknown expected_behavior: {expected}"}

    # ------------------------------------------------------------------
    # Answer mode
    # ------------------------------------------------------------------

    def _score_answer(self, sample: EvalSample, raw: str, norm: str) -> dict[str, Any]:
        gold = sample.answer.gold_answer
        gold_norm = self._norm(gold)

        # Exact match (after normalization)
        if gold_norm and self._fuzzy_equal(norm, gold_norm):
            label = "exact"
        else:
            # Partial: most must_include present, no must_not_include violations
            inc_ratio, inc_violations = self._check_must(raw, sample.answer.must_include)
            not_inc_violations = self._check_must_not(raw, sample.answer.must_not_include)

            if inc_ratio >= 0.5 and not not_inc_violations:
                label = "partial"
            else:
                label = "wrong"

        inc_ratio, inc_violations = self._check_must(raw, sample.answer.must_include)
        not_inc_violations = self._check_must_not(raw, sample.answer.must_not_include)

        return {
            "answer_label": label,
            "must_include_hit_ratio": round(inc_ratio, 4),
            "must_not_include_violations": not_inc_violations,
            "notes": "",
        }

    # ------------------------------------------------------------------
    # Refuse mode
    # ------------------------------------------------------------------

    def _score_refuse(self, sample: EvalSample, raw: str, norm: str) -> dict[str, Any]:
        """
        refuse_correct  – answer expresses uncertainty / not-found.
        refuse_wrong    – answer tries to provide a definitive answer.
        """
        is_refuse = self._contains_refuse(norm)

        if is_refuse:
            label = "refuse_correct"
        else:
            label = "refuse_wrong"

        inc_ratio, inc_violations = self._check_must(raw, sample.answer.must_include)
        not_inc_violations = self._check_must_not(raw, sample.answer.must_not_include)

        return {
            "answer_label": label,
            "must_include_hit_ratio": round(inc_ratio, 4),
            "must_not_include_violations": not_inc_violations,
            "notes": "",
        }

    # ------------------------------------------------------------------
    # Clarify mode
    # ------------------------------------------------------------------

    def _score_clarify(self, sample: EvalSample, raw: str, norm: str) -> dict[str, Any]:
        """
        clarify_correct – answer contains a clarifying question / request for detail.
        clarify_wrong   – answer does not ask for clarification.
        """
        is_clarify = self._contains_clarify(norm)

        if is_clarify:
            label = "clarify_correct"
        else:
            label = "clarify_wrong"

        inc_ratio, inc_violations = self._check_must(raw, sample.answer.must_include)
        not_inc_violations = self._check_must_not(raw, sample.answer.must_not_include)

        return {
            "answer_label": label,
            "must_include_hit_ratio": round(inc_ratio, 4),
            "must_not_include_violations": not_inc_violations,
            "notes": "",
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _norm(self, text: str) -> str:
        if not self.normalize_text:
            return text.strip().lower()
        if not text:
            return ""
        # collapse whitespace, remove punctuation that adds noise
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()
        return text

    def _fuzzy_equal(self, a: str, b: str, threshold: float = 0.85) -> bool:
        """Simple token-overlap fuzzy equality."""
        if not a or not b:
            return False
        if a == b:
            return True
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return (overlap / union) >= threshold

    def _check_must(self, text: str, phrases: list[str]) -> tuple[float, list[str]]:
        """Return (hit_ratio, list of missed phrases)."""
        if not phrases:
            return 1.0, []
        norm_text = self._norm(text)
        hits = [p for p in phrases if self._norm(p) in norm_text]
        return len(hits) / len(phrases), [p for p in phrases if p not in hits]

    def _check_must_not(self, text: str, phrases: list[str]) -> list[str]:
        """Return list of phrases that ARE found (i.e. violations)."""
        if not phrases:
            return []
        norm_text = self._norm(text)
        return [p for p in phrases if self._norm(p) in norm_text]

    def _contains_refuse(self, norm_text: str) -> bool:
        return any(rp in norm_text for rp in _REFUSE_PHRASES)

    def _contains_clarify(self, norm_text: str) -> bool:
        return any(cp in norm_text for cp in _CLARIFY_PHRASES)
