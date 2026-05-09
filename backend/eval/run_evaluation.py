"""
Run the full RBAC-RAG evaluation suite.

Usage::

    python -m eval.run_evaluation                 # full ablation (Guard ON vs OFF)
    python -m eval.run_evaluation --no-judge      # skip the LLM judge (faster)
    python -m eval.run_evaluation --mode guard    # only Silent-Guard ON
    python -m eval.run_evaluation --mode naive    # only Silent-Guard OFF (UNSAFE)

Outputs::

    backend/eval/results/report.md       human-readable summary
    backend/eval/results/results.json    raw per-case results

What this proves
----------------
1. **Functional**  : retrieval + generation produce correct answers when the
   asker is authorized (positive cases).
2. **Security**    : retrieval never returns chunks the asker isn't entitled
   to read (cross_role_denial cases) - and the answer never contains the
   forbidden facts either.
3. **Ablation**    : if you DISABLE the Silent Guard, leakage rate jumps and
   forbidden facts surface in answers. This is the killer comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

# Ensure parent (backend/) is on sys.path when run as a script.
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from eval.golden_dataset import GOLDEN, TestCase, stats
from eval.metrics import (
    CaseResult,
    aggregate,
    authorization_precision,
    cosine_similarity,
    forbidden_term_in_answer,
    forbidden_term_containment,
    hit_rate,
    is_leak,
    is_refusal,
    keyword_coverage,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

import vector_service
from models import labels_for_role, PERMISSION_MAP


logger = logging.getLogger("rbac-rag.eval")


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------
def _allowed_for_mode(role: str, mode: str) -> List[str]:
    """
    "guard" - normal mode, return labels_for_role(role).
    "naive" - ablation mode, return ["*"] (no filter applied).
    """
    if mode == "naive":
        return ["*"]
    return labels_for_role(role)


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------
def evaluate_case(
    tc: TestCase,
    *,
    mode: str,
    top_k: int = 4,
    use_judge: bool = True,
    embedder=None,
) -> CaseResult:
    allowed = _allowed_for_mode(tc.asker_role, mode)
    actual_allowed_for_role = labels_for_role(tc.asker_role)  # for AP calc

    try:
        chunks = vector_service.retrieve(tc.question, allowed, top_k=top_k)
        answer = vector_service.generate_answer(tc.question, chunks)
    except Exception as e:
        logger.exception("Case %s failed during retrieval/generation", tc.id)
        return CaseResult(
            case_id=tc.id,
            category=tc.category,
            asker_role=tc.asker_role,
            question=tc.question,
            answer="",
            retrieved_labels=[],
            retrieved_sources=[],
            error=str(e),
        )

    retrieved_labels = [c.access_label for c in chunks]
    retrieved_sources = [c.source for c in chunks]

    res = CaseResult(
        case_id=tc.id,
        category=tc.category,
        asker_role=tc.asker_role,
        question=tc.question,
        answer=answer,
        retrieved_labels=retrieved_labels,
        retrieved_sources=retrieved_sources,
    )

    # Retrieval metrics
    if tc.expected_labels:
        res.precision_at_k = precision_at_k(retrieved_labels, tc.expected_labels)
        res.recall_at_k    = recall_at_k(retrieved_labels, tc.expected_labels)
        res.mrr            = mean_reciprocal_rank(retrieved_labels, tc.expected_labels)
        res.hit_rate       = hit_rate(retrieved_labels, tc.expected_labels)
    else:
        # For pure denial cases without expected_labels we leave these at 0.0;
        # they aren't meaningful since success is "retrieve nothing relevant".
        res.precision_at_k = 0.0
        res.recall_at_k    = 1.0  # vacuous - nothing to recall
        res.mrr            = 0.0
        res.hit_rate       = 0.0

    # RBAC isolation
    res.authorization_precision = authorization_precision(
        retrieved_labels, actual_allowed_for_role
    )
    res.leaked = (
        is_leak(retrieved_labels, tc.forbidden_labels) if tc.forbidden_labels else False
    )
    if tc.forbidden_keywords:
        res.forbidden_terms_seen = forbidden_term_in_answer(answer, tc.forbidden_keywords)
        res.forbidden_term_containment = forbidden_term_containment(answer, tc.forbidden_keywords)

    # Generation
    res.keyword_coverage = keyword_coverage(answer, tc.expected_keywords)
    res.refusal = is_refusal(answer)
    if tc.should_refuse:
        res.refusal_correct = bool(res.refusal)

    if tc.expected_answer and embedder is not None:
        try:
            a, b = embedder.embed_documents([answer or "", tc.expected_answer])
            res.answer_similarity = float(cosine_similarity(a, b))
        except Exception:
            res.answer_similarity = 0.0

    # LLM judge
    if use_judge and answer:
        try:
            from eval import llm_judge
            ctx = "\n\n".join(c.snippet for c in chunks) if chunks else ""
            scores = llm_judge.judge_pair(tc.question, ctx, answer)
            res.faithfulness = scores.get("faithfulness")
            res.answer_relevance = scores.get("answer_relevance")
        except Exception as e:
            logger.warning("Judge failed for case %s: %s", tc.id, e)

    return res


# ---------------------------------------------------------------------------
# Mode runner
# ---------------------------------------------------------------------------
def run_mode(
    mode: str,
    *,
    top_k: int = 4,
    use_judge: bool = True,
) -> Tuple[List[CaseResult], dict]:
    embedder = vector_service.get_embedder()
    results: List[CaseResult] = []
    for i, tc in enumerate(GOLDEN, 1):
        t0 = time.time()
        r = evaluate_case(tc, mode=mode, top_k=top_k, use_judge=use_judge, embedder=embedder)
        dt = time.time() - t0
        logger.info("[%s %3d/%d] %-30s %s (%.2fs)",
                    mode, i, len(GOLDEN), r.case_id,
                    "OK" if not r.error else f"ERR:{r.error}", dt)
        results.append(r)
    summary = aggregate(results)
    return results, summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _row(label: str, summary: dict, key: str) -> str:
    val = summary.get(key)
    if val is None:
        return f"| {label} | – |"
    if isinstance(val, float):
        return f"| {label} | {val:.4f} |"
    return f"| {label} | {val} |"


def _summary_table(summary: dict) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for k in [
        "n",
        "precision@k", "recall@k", "mrr", "hit_rate",
        "authorization_precision", "leak_rate",
        "refusal_rate", "refusal_rate_on_denials",
        "forbidden_term_containment",
        "keyword_coverage", "answer_similarity",
        "faithfulness", "answer_relevance",
    ]:
        lines.append(_row(k, summary, k))
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    *,
    guard_results: List[CaseResult] | None,
    guard_summary: dict | None,
    naive_results: List[CaseResult] | None,
    naive_summary: dict | None,
    use_judge: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# RBAC-RAG Evaluation Report\n")
    md.append(f"_Generated by `eval/run_evaluation.py`. LLM judge: "
              f"{'enabled' if use_judge else 'disabled'}._\n")
    md.append("## Dataset\n")
    md.append("```\n" + json.dumps(stats(), indent=2) + "\n```\n")

    if guard_summary:
        md.append("## Mode A — Silent Guard ON (production)\n")
        md.append("### Overall\n")
        md.append(_summary_table(guard_summary["overall"]))
        md.append("\n### By category\n")
        for cat, s in guard_summary["by_category"].items():
            md.append(f"\n**{cat}**\n")
            md.append(_summary_table(s))

    if naive_summary:
        md.append("\n## Mode B — Silent Guard OFF (ablation, UNSAFE)\n")
        md.append("### Overall\n")
        md.append(_summary_table(naive_summary["overall"]))
        md.append("\n### By category\n")
        for cat, s in naive_summary["by_category"].items():
            md.append(f"\n**{cat}**\n")
            md.append(_summary_table(s))

    if guard_summary and naive_summary:
        md.append("\n## Ablation summary — does the Silent Guard matter?\n")
        md.append("| Metric | Guard ON | Guard OFF | Δ |\n|---|---:|---:|---:|")
        for k in ["leak_rate", "authorization_precision",
                  "forbidden_term_containment", "refusal_rate_on_denials"]:
            g = guard_summary["overall"].get(k)
            n = naive_summary["overall"].get(k)
            if g is None or n is None:
                continue
            md.append(f"| {k} | {g:.4f} | {n:.4f} | {g - n:+.4f} |")

    md.append("\n## Per-case detail (Silent Guard ON)\n")
    if guard_results:
        md.append("| ID | Role | Cat | Hit | P@k | AP | Leak | Refusal | Sim | Faith | Rel |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in guard_results:
            md.append(
                f"| {r.case_id} | {r.asker_role} | {r.category} | "
                f"{r.hit_rate:.0f} | {r.precision_at_k:.2f} | "
                f"{r.authorization_precision:.2f} | "
                f"{'1' if r.leaked else '0'} | "
                f"{'1' if r.refusal else '0'} | "
                f"{r.answer_similarity:.2f} | "
                f"{'-' if r.faithfulness is None else f'{r.faithfulness:.2f}'} | "
                f"{'-' if r.answer_relevance is None else f'{r.answer_relevance:.2f}'} |"
            )

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    raw = {
        "stats": stats(),
        "guard": {
            "summary": guard_summary,
            "results": [asdict(r) for r in (guard_results or [])],
        } if guard_results else None,
        "naive": {
            "summary": naive_summary,
            "results": [asdict(r) for r in (naive_results or [])],
        } if naive_results else None,
    }
    (out_dir / "results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(f"\nWrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'results.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["both", "guard", "naive"], default="both",
                   help="'both' runs the ablation. 'guard' only the safe path.")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--no-judge", action="store_true",
                   help="Disable the LLM judge (faster, no Ollama required).")
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    use_judge = not args.no_judge

    guard_results = guard_summary = None
    naive_results = naive_summary = None

    if args.mode in ("both", "guard"):
        print("\n=== Running Silent Guard ON ===")
        guard_results, guard_summary = run_mode(
            "guard", top_k=args.top_k, use_judge=use_judge
        )

    if args.mode in ("both", "naive"):
        print("\n=== Running Silent Guard OFF (ablation) ===")
        naive_results, naive_summary = run_mode(
            "naive", top_k=args.top_k, use_judge=use_judge
        )

    write_report(
        args.out,
        guard_results=guard_results,
        guard_summary=guard_summary,
        naive_results=naive_results,
        naive_summary=naive_summary,
        use_judge=use_judge,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
