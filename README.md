# RBAC RAG — Role-Based Access-Controlled Retrieval-Augmented Generation

A production-shaped RAG application where every retrieval is filtered by the
caller's role, so users (and the LLM) never see chunks they aren't authorized
to read.

```
┌────────────────┐   JWT (Bearer)    ┌──────────────────────────────┐
│   Angular 17   │ ─────────────────▶│      FastAPI (Python)        │
│  Login + Chat  │                   │                              │
│  HTTP intercep.│◀──── JSON ────────│  /api/auth/login   /api/query│
└────────────────┘                   │           │                  │
                                     │           ▼                  │
                                     │  ┌──────────────────────┐    │
                                     │  │   Silent Guard       │    │
                                     │  │ where: access_label  │    │
                                     │  │   ∈ user's labels    │    │
                                     │  └──────────┬───────────┘    │
                                     │             ▼                │
                                     │  ┌──────────────────────┐    │
                                     │  │   ChromaDB           │    │
                                     │  │   + MiniLM embeds    │    │
                                     │  └──────────┬───────────┘    │
                                     │             ▼                │
                                     │  ┌──────────────────────┐    │
                                     │  │   Ollama LLM         │    │
                                     │  └──────────────────────┘    │
                                     └──────────────────────────────┘
```

## What's in the box

| Path                               | Purpose                                                |
|------------------------------------|--------------------------------------------------------|
| `backend/main.py`                  | FastAPI app; `/api/auth/login`, `/api/query`, `/api/folders`, `/api/auth/me` |
| `backend/auth.py`                  | JWT issuance/validation, bcrypt, `get_current_active_user` dependency |
| `backend/vector_service.py`        | RAG core: ingestion, retrieval (Silent Guard), generation |
| `backend/models.py`                | Pydantic + SQLAlchemy models, `PERMISSION_MAP`         |
| `backend/database.py`              | SQLite engine + session helpers                         |
| `backend/ingestion.py`             | CLI to (re)build the vector store from `./data/`       |
| `backend/seed_users.py`            | Seeds 6 demo users (one per role)                      |
| `backend/requirements.txt`         | Python deps                                            |
| `backend/.env.example`             | Configuration template                                 |
| `frontend/`                        | Angular 17 standalone app (Login, Chat, AuthService, interceptor, guard) |
| `data/<ROLE>/*.txt`                | One folder per role; folder name becomes `access_label` |

## Roles & Permission Map

```
ADMIN   →  *  (all rooms)
CLevel  →  *  (all rooms)
ENGG    →  ENGG   + PUBLIC
SALES   →  SALES  + PUBLIC
HR      →  HR     + PUBLIC
INTERN  →  INTERN + PUBLIC
```

The map lives in `backend/models.py::PERMISSION_MAP`. Update it there and the
filter and the UI both follow.

## The `./data/` folder convention

Every file's *parent folder name* is injected into its vector metadata as
`access_label`. So:

```
data/
├── PUBLIC/         → access_label = "PUBLIC"
├── ENGG/           → access_label = "ENGG"
├── SALES/          → access_label = "SALES"
├── CLevel/         → access_label = "CLevel"
├── HR/             → access_label = "HR"
└── INTERN/         → access_label = "INTERN"
```

To add a new document:

1. Drop a `.txt` or `.md` file into the appropriate role folder.
2. Re-run `python ingestion.py --reset` (the `--reset` flag wipes the
   collection so renames/deletes are picked up cleanly).

> Folder names not in `ALLOWED_LABELS` (in `vector_service.py`) are skipped.
> Add new role names there *and* in `PERMISSION_MAP` to expand the system.

## Setup — Backend

Prereqs: **Python 3.10+** and (optionally) **Ollama** for the LLM.

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 1) seed demo users into SQLite
python seed_users.py

# 2) ingest sample docs into ChromaDB (the parent folder = access_label)
python ingestion.py --reset

# 3) (optional) install Ollama and pull a small model
#    https://ollama.com  -- then in another terminal:
#    ollama pull llama3.2
#    ollama serve         # default http://localhost:11434

# 4) run the API
cp .env.example .env       # edit JWT_SECRET first!
uvicorn main:app --reload --port 8000
```

> **No Ollama installed?** The API still works — `/api/query` returns the
> retrieved (authorized) chunks with a `[LLM unavailable]` notice instead of a
> generated answer, so you can verify the RBAC filter end-to-end without the LLM.

OpenAPI docs: http://localhost:8000/docs

### Demo credentials

| Username | Password   | Role   | Sees                  |
|----------|-----------|--------|-----------------------|
| admin    | admin123  | ADMIN  | everything            |
| ceo      | ceo123    | CLevel | everything            |
| eng      | eng123    | ENGG   | ENGG + PUBLIC         |
| sales    | sales123  | SALES  | SALES + PUBLIC        |
| hr       | hr123     | HR     | HR + PUBLIC           |
| intern   | intern123 | INTERN | INTERN + PUBLIC       |

## Setup — Frontend

Prereqs: **Node 18+** and **Angular CLI 17**.

```bash
cd frontend
npm install
npm start          # serves http://localhost:4200
```

Sign in with any of the demo accounts above. The sidebar shows your role and
the "Data Rooms" you can read from. Try asking the same question while logged
in as different users — e.g. *"What are the salary bands?"* — and watch the
answer change based on what the role is allowed to see.

## How the Silent Guard works

`vector_service.retrieve()` translates the caller's allowed labels into a
ChromaDB `where` filter:

```python
def _build_filter(allowed_labels):
    if "*" in allowed_labels:
        return None  # admins / CLevel see everything
    return {"access_label": {"$in": allowed_labels}}
```

The filter is applied **at the vector-store level**, before any chunks are
returned. The LLM receives only authorized text, so it cannot hallucinate
about content it never saw.

## End-to-end smoke test

```bash
# 1) login as the intern
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=intern&password=intern123" | python -c "import json,sys;print(json.load(sys.stdin)['access_token'])")

# 2) ask about CEO compensation - should NOT leak CLevel content
curl -s -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the CEO base salary?"}'
```

You should see an honest "I do not have enough information…" answer because
the intern's allowed labels are `["INTERN", "PUBLIC"]` and `CLevel` is filtered
out of the retrieval.

## Validation & RAG Evaluation

A full evaluation harness lives in `backend/eval/`. The methodology — written
for an academic defense — is in [`EVALUATION.md`](./EVALUATION.md).

```bash
# from backend/, with the index already built:
python -m eval.run_evaluation               # full ablation (Guard ON vs OFF) + LLM judge
python -m eval.run_evaluation --no-judge    # faster, no Ollama needed
python -m eval.rbac_tests --base-url http://localhost:8000   # live API security tests
```

What it measures:

- **Retrieval**: Precision@k, Recall@k, MRR, Hit Rate (label-level).
- **RBAC isolation**: Authorization Precision, Authorization Leak Rate,
  Refusal Rate on denial cases, Forbidden-Term Containment.
- **Generation**: Keyword Coverage, cosine Answer Similarity to gold,
  RAGAS-style Faithfulness & Answer Relevance via Ollama-as-judge.
- **Ablation**: every metric is run twice — Silent Guard ON vs OFF —
  to isolate the security contribution of the guard.

Reports are written to `backend/eval/results/report.md` and `results.json`.

## Production checklist

- [ ] Replace `JWT_SECRET` with a 32+ byte random value, kept in a secret store.
- [ ] Move from SQLite to Postgres for multi-instance deployments.
- [ ] Front the API with HTTPS (e.g. behind nginx/Cloudfront/an ALB).
- [ ] Tighten `CORS_ORIGINS` to only your real frontend domain.
- [ ] Add audit logging on `/api/query` (we log username + role + question).
- [ ] Rotate `OLLAMA_MODEL` to whatever your hardware can run; `llama3.2` is a
      good default on modest CPUs/GPUs.
- [ ] Consider per-document ACLs (`access_labels: ["ENGG","SALES"]` on a
      single chunk) by switching the filter to `$in` over a list field.















ssh -L 11434:localhost:11434 ubuntu@3.86.247.241

