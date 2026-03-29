"""
Dataset schema and loading utilities for the evaluation framework.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema definitions (plain dataclasses, no external dependencies)
# ---------------------------------------------------------------------------


@dataclass
class QuestionBlock:
    user_query: str = ""
    conversation_history: list[str] = field(default_factory=list)


@dataclass
class RetrievalBlock:
    # Label status for retrieval scoring:
    #   unlabeled        – no gold labels yet, skip retrieval metrics (answer metrics still run)
    #   labeled_doc      – gold_doc_ids provided, doc-level retrieval metrics active
    #   labeled_chunk    – gold_chunk_ids provided (gold_doc_ids may also exist), chunk-level active
    #   unanswerable     – unanswerable/refusal eval sample, skip retrieval metrics
    label_status: str = "unlabeled"
    gold_doc_ids: list[int] = field(default_factory=list)
    gold_chunk_ids: list[int] = field(default_factory=list)
    hard_negative_chunk_ids: list[int] = field(default_factory=list)
    # Section-level gold label (section title or path containing the gold evidence)
    gold_evidence_section: str = ""


@dataclass
class ContextBlock:
    gold_context_blocks: list[str] = field(default_factory=list)


@dataclass
class AnswerBlock:
    gold_answer: str = ""
    answer_style: str = "concise"  # concise | verbose | structured
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    faithfulness_requirements: list[str] = field(default_factory=list)


@dataclass
class SupervisionBlock:
    preferred_output: str = ""
    sft_messages_no_context: list[dict[str, str]] = field(default_factory=list)
    sft_messages_with_context: list[dict[str, str]] = field(default_factory=list)
    rejected_outputs: list[str] = field(default_factory=list)


@dataclass
class EvaluationBlock:
    expected_behavior: str = "answer"  # answer | refuse | clarify
    scoring_type: str = "rule+llm"  # rule | rule+llm
    error_type: Optional[str] = None
    notes: str = ""


@dataclass
class MetadataBlock:
    created_by: str = "human"
    source: str = "manual_seed"
    version: str = "1.0"


@dataclass
class EvalSample:
    """
    Single evaluation sample covering retrieval + answer + future SFT needs.
    """
    id: str = ""
    dataset: str = "kb_eval_seed"
    task_type: str = "factoid"  # factoid | procedural | yesno | refuse | clarify
    difficulty: str = "easy"   # easy | medium | hard
    tags: list[str] = field(default_factory=list)
    question: QuestionBlock = field(default_factory=QuestionBlock)
    retrieval: RetrievalBlock = field(default_factory=RetrievalBlock)
    context: ContextBlock = field(default_factory=ContextBlock)
    answer: AnswerBlock = field(default_factory=AnswerBlock)
    supervision: SupervisionBlock = field(default_factory=SupervisionBlock)
    evaluation: EvaluationBlock = field(default_factory=EvaluationBlock)
    metadata: MetadataBlock = field(default_factory=MetadataBlock)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalSample:
        # Handle question field: can be a plain string or a QuestionBlock dict
        question_raw = d.get("question", {})
        if isinstance(question_raw, str):
            question_block = QuestionBlock(user_query=question_raw)
        else:
            question_block = QuestionBlock(**question_raw)

        return cls(
            id=str(d.get("id", "")),
            dataset=str(d.get("dataset", "kb_eval_seed")),
            task_type=str(d.get("task_type", "factoid")),
            difficulty=str(d.get("difficulty", "easy")),
            tags=list(d.get("tags", [])),
            question=question_block,
            retrieval=RetrievalBlock(**d.get("retrieval", {})),
            context=ContextBlock(**d.get("context", {})),
            answer=AnswerBlock(**d.get("answer", {})),
            supervision=SupervisionBlock(**d.get("supervision", {})),
            evaluation=EvaluationBlock(**d.get("evaluation", {})),
            metadata=MetadataBlock(**d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {"id", "question", "retrieval", "answer", "evaluation", "metadata"}


def validate_sample(raw: dict[str, Any]) -> list[str]:
    """
    Validate a raw dict against the schema.
    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            errors.append(f"missing required top-level key: '{key}'")

    # question.user_query must be non-empty
    q = raw.get("question", {})
    if not str(q.get("user_query", "")).strip():
        errors.append("question.user_query must be non-empty")

    # Gold targets required only when label_status is labeled_doc or labeled_chunk
    retr = raw.get("retrieval", {})
    label_status = str(retr.get("label_status", "unlabeled"))
    if label_status in {"labeled_doc", "labeled_chunk"}:
        gold_chunks = retr.get("gold_chunk_ids", [])
        gold_docs = retr.get("gold_doc_ids", [])
        if not gold_chunks and not gold_docs:
            errors.append(
                "retrieval.label_status is 'labeled_doc/labeled_chunk' but both "
                "gold_chunk_ids and gold_doc_ids are empty"
            )

    # label_status must be a known value
    if label_status not in {"unlabeled", "labeled_doc", "labeled_chunk", "unanswerable"}:
        errors.append(
            f"retrieval.label_status must be one of unlabeled/labeled_doc/labeled_chunk/unanswerable, "
            f"got: {label_status!r}"
        )

    # evaluation.expected_behavior must be one of the known values
    eb = raw.get("evaluation", {}).get("expected_behavior", "")
    if eb not in {"answer", "refuse", "clarify"}:
        errors.append(f"evaluation.expected_behavior must be one of answer/refuse/clarify, got: {eb!r}")

    return errors


def validate_dataset(samples: list[EvalSample]) -> list[str]:
    """Validate an entire dataset. Returns list of error messages."""
    errors: list[str] = []
    ids_seen: set[str] = set()
    for i, sample in enumerate(samples):
        if sample.id in ids_seen:
            errors.append(f"[{i}] duplicate sample id: {sample.id!r}")
        ids_seen.add(sample.id)
        sample_errors = validate_sample(sample.to_dict())
        for err in sample_errors:
            errors.append(f"[{sample.id}] {err}")
    return errors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[EvalSample]:
    """
    Load a JSONL dataset from disk.
    Each line must be a valid JSON object.
    Raises ValueError if any line fails to parse or validate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[EvalSample] = []
    errors: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: JSON parse error: {e}")
                continue

            # Auto-assign id if missing
            if "id" not in raw or not raw["id"]:
                raw["id"] = f"auto_{uuid.uuid4().hex[:8]}"

            sample = EvalSample.from_dict(raw)
            samples.append(sample)

    if errors:
        raise ValueError(f"Errors loading dataset:\n" + "\n".join(errors))

    validation_errors = validate_dataset(samples)
    if validation_errors:
        raise ValueError(f"Dataset validation errors:\n" + "\n".join(validation_errors))

    return samples


# ---------------------------------------------------------------------------
# JSONL writing utility
# ---------------------------------------------------------------------------

def samples_to_jsonl(samples: list[EvalSample], path: str | Path) -> None:
    """Write a list of EvalSamples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
