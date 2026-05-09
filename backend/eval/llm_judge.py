"""
RAGAS-style LLM judge using a local Ollama model.

Two scores per (question, context, answer) triple:

* **Faithfulness**     : are all factual claims in the answer supported by
  the retrieved context? (precision-oriented; hallucination detector)
* **Answer Relevance** : does the answer actually address the question?
  (recall-oriented; off-topic detector)

The judge returns floats in [0, 1] when the LLM is reachable; otherwise
``None`` (so callers can omit the column from their reports cleanly).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """\
You are an evaluator. Rate FAITHFULNESS on a 0.0 to 1.0 scale.

Faithfulness = the proportion of factual claims in the ANSWER that are
*directly supported* by the CONTEXT.

* 1.0 means every claim in the answer is supported by the context.
* 0.0 means none of the claims are supported.
* If the answer is a refusal ("I do not have enough information"), and the
  context indeed lacks the information needed, the score is 1.0.

Respond with a single JSON object: {{"score": 0.0-1.0, "rationale": "..."}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

JSON:
"""

_RELEVANCE_PROMPT = """\
You are an evaluator. Rate ANSWER RELEVANCE on a 0.0 to 1.0 scale.

Relevance = how directly the ANSWER addresses the QUESTION.

* 1.0 means the answer is fully on-topic and addresses what was asked.
* 0.0 means the answer is off-topic or completely unrelated.
* A well-justified refusal because the role lacks access still scores 0.5
  (the system did the right thing but did not produce the requested content).

Respond with a single JSON object: {{"score": 0.0-1.0, "rationale": "..."}}

QUESTION:
{question}

ANSWER:
{answer}

JSON:
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def judge_pair(
    question: str,
    context: str,
    answer: str,
    *,
    timeout: float = 30.0,
) -> dict:
    """
    Return ``{"faithfulness": float|None, "answer_relevance": float|None}``.

    Returns ``None`` for any score the judge could not produce (LLM down,
    parse failure, etc.) - the harness handles that gracefully.
    """
    return {
        "faithfulness": _score(
            _FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
            timeout=timeout,
        ),
        "answer_relevance": _score(
            _RELEVANCE_PROMPT.format(question=question, answer=answer),
            timeout=timeout,
        ),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _score(prompt: str, *, timeout: float) -> Optional[float]:
    text = _call_ollama(prompt, timeout=timeout)
    if text is None:
        return None
    return _parse_score(text)


def _call_ollama(prompt: str, *, timeout: float) -> Optional[str]:
    """Plain stdlib HTTP call to Ollama's /api/generate endpoint."""
    import urllib.request, urllib.error
    body = json.dumps({
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
            return payload.get("response")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("Judge LLM unavailable: %s", e)
        return None
    except Exception as e:  # pragma: no cover
        logger.warning("Judge LLM failed: %s", e)
        return None


_JSON_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)
_FLOAT_RE = re.compile(r"\b(0?\.\d+|1(?:\.0+)?|0(?:\.0+)?)\b")


def _parse_score(text: str) -> Optional[float]:
    # Try JSON first.
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = float(obj.get("score"))
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # Fall back to first plausible float in [0, 1].
    for match in _FLOAT_RE.finditer(text):
        try:
            v = float(match.group(0))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None
