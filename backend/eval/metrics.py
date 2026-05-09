"""
Metric definitions for the RBAC-RAG evaluation harness.

Three families:

1. **Retrieval metrics** (label-level)
     - Precision@k, Recall@k, MRR, Hit Rate

2. **RBAC isolation metrics** (security)
     - Authorization Precision (AP)  : fraction of retrieved chunks whose
       access_label is in the asker's allowed set. Should be 1.00 always.
     - Authorization Leak Rate (ALR) : fraction of cross-role-denial cases
       where any unauthorized chunk reached the LLM. Should be 0.00.
     - Refusal Rate (RR)             : fraction of cross-role-denial cases
       answered with an explicit refusal.

3. **Generation metrics** (answer quality)
     - Keyword Coverage              : fraction of expected_keywords present
       (case-insensitive).
     - Forbidden Term Containment    : 1.0 if no forbidden_keywords appear,
       else 0.0.
     - Answer Similarity             : cosine similarity between the
       generated answer and the gold answer using the same embedding model
       used by retrieval.

Faithfulness and Answer Relevance (RAGAS-style) are computed by an LLM
judge in ``llm_judge.py`` and merged in by ``run_evaluation.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# We deliberately do NOT import vector_service at module load time to keep
# this file importable in environments without chromadb.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    r"do not have enough information",
    r"don't have enough information",
    r"cannot find",
    r"no information",
    r"not authori[sz]ed",
    r"unable to answer",
    r"\[llm unavailable\]",  # our graceful fallback also counts as a refusal
]


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(re.search(p, a) for p in _REFUSAL_PATTERNS)


def _normalize_label_set(labels: Sequence[str]) -> set:
    return {l.strip() for l in labels if l}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_labels: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of retrieved chunks whose access_label is in ``expected``."""
    if not retrieved_labels:
        return 0.0
    expected_set = _normalize_label_set(expected)
    hits = sum(1 for l in retrieved_labels if l in expected_set)
    return hits / len(retrieved_labels)


def recall_at_k(retrieved_labels: Sequence[str], expected: Sequence[str]) -> float:
    """
    Fraction of *expected* labels that appear at least once in retrieval.

    Defined at the *label* level rather than the chunk level because the gold
    dataset specifies which folders the answer should come from, not which
    chunks.
    """
    expected_set = _normalize_label_set(expected)
    if not expected_set:
        return 1.0  # vacuously true
    retrieved_set = _normalize_label_set(retrieved_labels)
    hits = expected_set & retrieved_set
    return len(hits) / len(expected_set)


def mean_reciprocal_rank(retrieved_labels: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = _normalize_label_set(expected)
    for rank, l in enumerate(retrieved_labels, start=1):
        if l in expected_set:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved_labels: Sequence[str], expected: Sequence[str]) -> float:
    return 1.0 if any(l in _normalize_label_set(expected) for l in retrieved_labels) else 0.0


# ---------------------------------------------------------------------------
# RBAC isolation metrics
# ---------------------------------------------------------------------------
def authorization_precision(
    retrieved_labels: Sequence[str], allowed_for_role: Sequence[str]
) -> float:
    """
    Fraction of retrieved chunks whose access_label is in the asker's
    permitted set. **Must be 1.0** for the RBAC layer to be considered correct.

    ``allowed_for_role == ["*"]`` means ADMIN/CLevel - everything is allowed.
    """
    if not retrieved_labels:
        return 1.0  # nothing returned -> nothing leaked
    if "*" in allowed_for_role:
        return 1.0
    allowed = _normalize_label_set(allowed_for_role)
    authorized = sum(1 for l in retrieved_labels if l in allowed)
    return authorized / len(retrieved_labels)


def is_leak(retrieved_labels: Sequence[str], forbidden_labels: Sequence[str]) -> bool:
    """True if any retrieved chunk has a forbidden access_label."""
    forbidden = _normalize_label_set(forbidden_labels)
    return any(l in forbidden for l in retrieved_labels)


def forbidden_term_in_answer(answer: str, forbidden_keywords: Sequence[str]) -> List[str]:
    """Return the list of forbidden keywords that *did* appear in the answer."""
    a = (answer or "").lower()
    return [kw for kw in forbidden_keywords if kw and kw.lower() in a]


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------
def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    if not expected_keywords:
        return 1.0
    a = (answer or "").lower()
    hit = sum(1 for kw in expected_keywords if kw and kw.lower() in a)
    return hit / len(expected_keywords)


def forbidden_term_containment(answer: str, forbidden_keywords: Sequence[str]) -> float:
    """1.0 if no forbidden keyword appears, else 0.0. (Containment in the
    confidentiality sense - we 'contained' the secret.)"""
    return 0.0 if forbidden_term_in_answer(answer, forbidden_keywords) else 1.0


def cosine_similarity(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    case_id: str
    category: str
    asker_role: str
    question: str
    answer: str
    retrieved_labels: List[str]
    retrieved_sources: List[str]

    # Retrieval
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0

    # RBAC
    authorization_precision: float = 1.0
    leaked: bool = False
    forbidden_terms_seen: List[str] = field(default_factory=list)

    # Generation
    keyword_coverage: float = 0.0
    forbidden_term_containment: float = 1.0
    answer_similarity: float = 0.0
    refusal: bool = False
    refusal_correct: Optional[bool] = None  # None for non-denial cases

    # LLM judge (filled in optionally)
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None

    error: Optional[str] = None


def mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def aggregate(results: List[CaseResult]) -> dict:
    """Aggregate per-category and overall metrics."""
    if not results:
        return {}
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    def summarise(rs: List[CaseResult]) -> dict:
        denials = [r for r in rs if r.category == "cross_role_denial"]
        return {
            "n": len(rs),
            "precision@k":              round(mean([r.precision_at_k for r in rs]), 4),
            "recall@k":                 round(mean([r.recall_at_k for r in rs]), 4),
            "mrr":                      round(mean([r.mrr for r in rs]), 4),
            "hit_rate":                 round(mean([r.hit_rate for r in rs]), 4),
            "authorization_precision":  round(mean([r.authorization_precision for r in rs]), 4),
            "leak_rate":                round(mean([1.0 if r.leaked else 0.0 for r in rs]), 4),
            "refusal_rate":             round(mean([1.0 if r.refusal else 0.0 for r in rs]), 4),
            "refusal_rate_on_denials":  round(
                mean([1.0 if r.refusal else 0.0 for r in denials]), 4
            ) if denials else None,
            "forbidden_term_containment": round(
                mean([r.forbidden_term_containment for r in rs]), 4
            ),
            "keyword_coverage":         round(mean([r.keyword_coverage for r in rs]), 4),
            "answer_similarity":        round(mean([r.answer_similarity for r in rs]), 4),
            "faithfulness":             round(mean([r.faithfulness for r in rs if r.faithfulness is not None]), 4) if any(r.faithfulness is not None for r in rs) else None,
            "answer_relevance":         round(mean([r.answer_relevance for r in rs if r.answer_relevance is not None]), 4) if any(r.answer_relevance is not None for r in rs) else None,
        }

    return {
        "overall": summarise(results),
        "by_category": {cat: summarise(rs) for cat, rs in by_cat.items()},
    }
