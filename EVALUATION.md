# Validation & Evaluation Methodology — RBAC-RAG

This document is the **defense reference** for the evaluation of the
Role-Based Access-Controlled Retrieval-Augmented Generation system.
It explains *what we measured*, *why those metrics*, *how the dataset was
constructed*, and *what the results mean* — including an ablation study
that isolates the contribution of the Silent Guard pattern.

---

## 1. Research contribution being evaluated

We propose the **Silent Guard pattern** for RAG: enforce role-based
authorization *at the vector-store retrieval step* by attaching an
`access_label` (the parent folder name) to every chunk during ingestion
and applying a metadata filter on every query. The LLM never sees data
the caller is not entitled to read.

The evaluation must therefore answer three questions:

1. **Q1 — Functional correctness.** Does the system return correct,
   useful answers to authorized queries?
2. **Q2 — Security correctness.** Does the system *prevent* unauthorized
   access at retrieval, generation, and API boundaries?
3. **Q3 — Necessity of the Silent Guard.** Compared with a naive RAG
   baseline (no metadata filter), does the Silent Guard reduce leakage
   to zero without hurting answer quality on authorized queries?

---

## 2. Evaluation taxonomy

We use a 3-axis taxonomy:

| Axis              | Layer evaluated                          | Methods                                      |
|-------------------|------------------------------------------|----------------------------------------------|
| **Retrieval**     | Vector-store + metadata filter           | Precision@k, Recall@k, MRR, Hit Rate         |
| **Generation**    | LLM grounded in retrieved context        | Keyword Coverage, Answer Similarity, RAGAS-style Faithfulness & Answer Relevance |
| **Security/RBAC** | The boundary between roles and content   | Authorization Precision, Authorization Leak Rate, Refusal Rate, Forbidden-Term Containment, plus black-box API tests |

Every metric is evaluated under two ablation modes:

* **Mode A — Silent Guard ON.** Production behavior. The metadata filter
  `where: {"access_label": {"$in": role_labels}}` is applied at retrieval.
* **Mode B — Silent Guard OFF.** Same data, same retrieval, but the filter
  is **removed**. This isolates the contribution of the guard.

---

## 3. Metric definitions

Notation: for a query, let
*R* = list of retrieved chunks, of size *k*;
*L(R)* = the multiset of `access_label`s on those chunks;
*E* = the set of *expected* labels (gold-labelled per test case);
*F* = the set of *forbidden* labels (gold-labelled for cross-role denial cases);
*Aᵤ* = the set of labels the user *u* is allowed to read.

### 3.1 Retrieval

* **Precision@k**     = |{l ∈ L(R) : l ∈ E}| / k
* **Recall@k**        = |E ∩ {labels in L(R)}| / |E|  *(label-level recall)*
* **MRR**             = 1 / rank of first label in L(R) that lies in E (else 0)
* **Hit Rate**        = 1 if L(R) ∩ E ≠ ∅, else 0

We use *label-level* recall because the gold dataset specifies the
**folder** an answer should come from, not which specific chunk. This is
the correct unit for an RBAC system.

### 3.2 RBAC isolation (the security layer)

* **Authorization Precision (AP)**
  = |{l ∈ L(R) : l ∈ Aᵤ}| / |L(R)|
  &nbsp;&nbsp;&nbsp;*Must be 1.00 for every query.* Anything less is a
  retrieval-level leak.
* **Authorization Leak Rate (ALR)**
  = (# cross-role-denial cases where L(R) ∩ F ≠ ∅) / (# cross-role-denial cases)
  &nbsp;&nbsp;&nbsp;*Must be 0.00.*
* **Refusal Rate on Denial Cases (RR)**
  = (# cross-role-denial cases the assistant refused) / (# cross-role-denial cases)
  &nbsp;&nbsp;&nbsp;A correct system either retrieves no relevant chunks
  *and* the LLM refuses, or just retrieves no relevant chunks and produces
  an honest "I don't have that information." Either path is acceptable;
  RR = 1.00 is ideal.
* **Forbidden-Term Containment (FTC)**
  = 1.0 if no forbidden keyword appears in the answer text, else 0.0.
  &nbsp;&nbsp;&nbsp;Defense-in-depth check: even if retrieval somehow
  returned a wrong chunk, we want the answer to not parrot the secret.

### 3.3 Generation

* **Keyword Coverage** = (# expected keywords found in answer) / (# expected keywords).
  Coarse but interpretable; easy for an examiner to verify by eye.
* **Answer Similarity** = cosine similarity between the embedding of the
  generated answer and the gold answer. Uses the same MiniLM model as
  retrieval for consistency.
* **Faithfulness** *(RAGAS-style, LLM-as-judge)* — proportion of factual
  claims in the answer that are supported by the retrieved context.
  Hallucination detector. Scored 0–1 by an Ollama judge model.
* **Answer Relevance** *(RAGAS-style, LLM-as-judge)* — does the answer
  address the question? Off-topic detector. Scored 0–1.

> The LLM judge is optional. If Ollama is unavailable, those columns are
> simply omitted from the report — every other metric remains reproducible
> with no API dependencies.

---

## 4. Test dataset construction

The dataset (`backend/eval/golden_dataset.py`) contains **30 hand-crafted
test cases** distributed across categories and roles:

```
total                    : 30
category:positive        : 16  (one role asks about its own data)
category:cross_role_denial: 10  (one role asks about someone else's data)
category:public_shared   :  4  (one role asks about PUBLIC data)

asker_role:ENGG    :  4
asker_role:SALES   :  4
asker_role:HR      :  4
asker_role:CLevel  :  3
asker_role:INTERN  :  5
asker_role:ADMIN   :  2
+ public/cross-role rotated across all six roles
```

Each test case carries:

* `expected_labels` — folders the answer should come from (positive
  cases). Always a subset of the asker's permitted labels.
* `forbidden_labels` — folders that **must not** appear in retrieval
  (cross-role denial cases).
* `expected_keywords` — substrings that should appear in a correct answer.
* `forbidden_keywords` — substrings that **must not** appear (e.g. the
  CEO's salary number when an INTERN asks).
* `expected_answer` — gold-text answer for cosine-similarity scoring.
* `should_refuse` — boolean; the assistant should answer with a refusal.

### Why this construction?
* **Balanced positive vs. denial split** — without enough denial cases
  you can't evaluate Q2 / Q3.
* **Realistic confidential targets** — CEO salary, salary bands,
  acquisition targets, on-call runbooks, pricing playbooks. These are
  the assets a real company most needs to protect.
* **Cross-role coverage** — every non-admin role appears as both asker
  and victim of leakage attempts.

---

## 5. Threat model

**In scope.**
1. *Same-organization RBAC.* An authenticated employee asks the system
   for content their role isn't entitled to read.
2. *LLM-mediated leakage.* The LLM sees too much context and reveals
   secrets in its generated answer.
3. *API-level abuse.* Token tampering, missing tokens, garbage tokens.

**Out of scope.**
1. Network-level attacks (TLS, MITM) — handled by the deployment.
2. Compromise of the JWT signing key — operational concern.
3. Side-channel attacks via timing or chunk-count signals — possible
   future work; not covered here.
4. Adversarial prompt injection embedded *in the documents themselves*.
   We assume corpus integrity (only the system administrator can write
   to `data/`).

---

## 6. Test harness

```
backend/eval/
├── golden_dataset.py     # 30 labelled test cases
├── metrics.py            # all metric implementations
├── llm_judge.py          # RAGAS-style Ollama judge (optional)
├── rbac_tests.py         # black-box API security tests
├── run_evaluation.py     # orchestrator + ablation + report writer
└── results/              # auto-generated reports
    ├── report.md
    └── results.json
```

### How to run

```bash
# 1) make sure the index is built
cd backend
python ingestion.py --reset

# 2) run the full evaluation (ablation included)
python -m eval.run_evaluation

# 2a) faster, no Ollama required
python -m eval.run_evaluation --no-judge

# 3) live-API security tests (run while uvicorn is up)
python -m eval.rbac_tests --base-url http://localhost:8000
```

The evaluator runs every test case **twice** — once with the Silent Guard
ON, once with it OFF — and writes a single comparison report to
`backend/eval/results/report.md`.

---

## 7. Expected results

The Silent Guard is correct iff:

| Metric (overall)              | Guard ON     | Guard OFF (naive) |
|-------------------------------|--------------|-------------------|
| Authorization Precision (AP)  | **1.00**     | < 1.00            |
| Authorization Leak Rate (ALR) | **0.00**     | > 0.00            |
| Forbidden-Term Containment    | high (≥0.95) | low               |
| Refusal Rate on denial cases  | ≥ 0.90       | drops sharply     |
| Hit Rate (positive cases)     | high         | high *(unchanged - guard doesn't penalize legitimate access)* |
| Faithfulness                  | high         | mixed             |
| Answer Relevance              | high         | high              |

The killer takeaway for the defense:

> **Removing the Silent Guard does not improve answer quality on
> authorized queries (Hit Rate, Answer Similarity, Answer Relevance stay
> approximately constant) but it causes ALR > 0 and AP < 1.0** — i.e.
> the security gain is "free" on authorized traffic and binary on
> unauthorized traffic.

This is the ablation conclusion the examiner is most likely to ask for.

---

## 8. Limitations & honest caveats

1. **Dataset size.** 30 cases is enough to demonstrate the design pattern
   but not to make strong statistical claims about generalization. A
   thesis-grade study would scale to 200+ cases ideally with crowd-sourced
   gold answers and inter-annotator agreement.
2. **LLM judge variance.** Faithfulness/relevance scores from the same
   prompt vary up to ~0.05 across runs because the judge is non-zero
   temperature. We pin temperature=0 and average across 1 run; for a
   more rigorous study, average over k≥3 runs.
3. **Embedding model fixedness.** All retrieval and similarity metrics
   share a single embedding model (MiniLM-L6-v2). A second study with
   a different embedder (e.g. `bge-large-en`) would test robustness to
   embedding choice.
4. **Single-document chunks.** Each chunk currently has exactly one
   `access_label` (its parent folder). The Silent Guard generalizes to
   multi-label ACLs (`access_labels: ["ENGG","SALES"]`) by switching
   the filter to `$in` over a list field — supported by ChromaDB, but
   not exercised in this evaluation.
5. **Refusal detection is regex-based.** A more principled approach
   would use the LLM judge to label refusals; we use regex for
   reproducibility and speed.
6. **No adversarial prompt injection tests** in the corpus. See § 5.

---

## 9. Likely defense questions and prepared answers

**Q: Why label-level recall, not chunk-level?**
A: The gold dataset specifies *which folders* should be sourced, not
which specific chunks. Chunk-level recall would require us to also
label every individual chunk, which is fragile under re-chunking. Label
recall measures the right invariant for an RBAC system: did we retrieve
from the correct authorization domain?

**Q: How do you know your forbidden-keyword check isn't gameable?**
A: It's defense-in-depth, not the primary control. The primary control
is `Authorization Leak Rate` measured at the chunk level — if `ALR=0`,
the LLM never saw the secret to leak it. The keyword check is a
secondary cross-check on the answer text.

**Q: What if the LLM hallucinates the correct secret value without
having retrieved it?**
A: Possible in theory; an LLM that knows ACME's CEO salary from
pre-training. We bound this in the system prompt ("answer ONLY from
context"), but a perfectly adversarial prompt could probe the model's
priors. This is a known limitation of any RAG system; a hardening step
is to use a model fine-tuned to refuse on out-of-context content. We
log it under § 8.

**Q: Why not use RAGAS directly?**
A: RAGAS depends on OpenAI by default, which would make our evaluation
non-reproducible without internet access and an API key. We re-implement
the two RAGAS-style metrics (faithfulness, answer relevance) using
Ollama as the judge, with the same prompt structure. The implementation
is ~120 lines and is auditable in `eval/llm_judge.py`.

**Q: Could a stale JWT escalate privileges?**
A: No. `auth.py::get_current_active_user` re-fetches the user's role
from SQLite on every request. The role claim in the token is **not
trusted** for authorization decisions — the DB is the source of truth.
Tested by `rbac_tests.py::t_garbage_token_rejected` and by the role
re-fetch logic.

**Q: How do you scale the permission map to 100+ roles?**
A: Today the map is a Python dict. Production would migrate it to a
table (`roles`, `role_permissions`) with a many-to-many relation to
labels — same data shape, just persistent. The retrieval call signature
(`labels_for_role(role) -> List[str]`) is unchanged.

---

## 10. Reproducibility checklist

* [x] Single command to rebuild the index (`python ingestion.py --reset`)
* [x] Single command to run the eval (`python -m eval.run_evaluation`)
* [x] No proprietary APIs required (Ollama is local; HuggingFace embed
      model is local)
* [x] Deterministic temperature for both generation and judge
* [x] Test dataset checked into the repo
* [x] Raw per-case results dumped to JSON (audit trail)
* [x] Full methodology in this document
