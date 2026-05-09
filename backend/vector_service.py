"""
vector_service.py - The secure RAG core.

Responsibilities
----------------
1. Hold a single ChromaDB persistent collection for the whole app.
2. Use a local sentence-transformers embedding model (no API key required).
3. Apply the **Silent Guard** metadata filter on every retrieval - the LLM
   never sees chunks the caller is not entitled to read.
4. Call Ollama (configurable model) to generate the final answer.

The ``access_label`` metadata field is set to the *parent folder name* during
ingestion and matched against the user's allowed labels at query time.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import SourceChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHROMA_DIR = BACKEND_DIR / "chroma_store"

DATA_DIR = Path(os.getenv("RBAC_DATA_DIR", str(DEFAULT_DATA_DIR)))
CHROMA_DIR = Path(os.getenv("RBAC_CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))
COLLECTION_NAME = os.getenv("RBAC_COLLECTION", "rbac_rag")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Folders here will be ingested. Anything else under data/ is ignored.
ALLOWED_LABELS = {"PUBLIC", "ENGG", "SALES", "CLevel", "HR", "INTERN"}


# ---------------------------------------------------------------------------
# Singletons (lazy)
# ---------------------------------------------------------------------------
_embedder: Optional[HuggingFaceEmbeddings] = None
_chroma_client: Optional[chromadb.api.ClientAPI] = None
_collection = None
_llm: Optional[Ollama] = None


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        logger.info("Connecting to Ollama at %s, model=%s", OLLAMA_BASE_URL, OLLAMA_MODEL)
        _llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    return _llm


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def _iter_role_files(data_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (access_label, file_path) for every readable file under data_dir."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in ALLOWED_LABELS:
            logger.warning("Skipping unknown role folder: %s", label)
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                yield label, path


def ingest(data_dir: Optional[Path] = None, *, reset: bool = False) -> dict:
    """
    Walk ``data_dir`` and load every file into Chroma.

    Each chunk gets metadata::

        {
          "access_label": <parent folder name>,   # e.g. "ENGG"
          "source":       <relative path>,        # e.g. "ENGG/oncall_runbook.txt"
          "filename":     <basename>,
        }

    ``reset=True`` wipes the collection first - useful for re-runs.
    """
    data_dir = data_dir or DATA_DIR
    collection = get_collection()
    embedder = get_embedder()

    if reset:
        try:
            ids = collection.get(include=[])["ids"]
            if ids:
                collection.delete(ids=ids)
                logger.info("Reset: deleted %d existing chunks.", len(ids))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not reset collection: %s", exc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_texts: List[str] = []
    all_metas: List[dict] = []
    all_ids: List[str] = []
    files_seen = 0

    for label, path in _iter_role_files(data_dir):
        files_seen += 1
        try:
            docs = TextLoader(str(path), encoding="utf-8").load()
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            continue

        rel = path.relative_to(data_dir).as_posix()
        for doc in docs:
            for i, chunk in enumerate(splitter.split_text(doc.page_content)):
                all_texts.append(chunk)
                all_metas.append({
                    "access_label": label,
                    "source": rel,
                    "filename": path.name,
                })
                all_ids.append(f"{rel}::chunk-{i}-{len(all_texts):06d}")

    if not all_texts:
        return {"files": 0, "chunks": 0, "data_dir": str(data_dir)}

    logger.info("Embedding %d chunks from %d files...", len(all_texts), files_seen)
    embeddings = embedder.embed_documents(all_texts)

    # Chroma can take big batches but stay well under any limit.
    BATCH = 256
    for i in range(0, len(all_texts), BATCH):
        collection.add(
            ids=all_ids[i:i + BATCH],
            documents=all_texts[i:i + BATCH],
            metadatas=all_metas[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
        )

    return {
        "files": files_seen,
        "chunks": len(all_texts),
        "data_dir": str(data_dir),
        "labels": sorted({m["access_label"] for m in all_metas}),
    }


# ---------------------------------------------------------------------------
# Retrieval - the Silent Guard
# ---------------------------------------------------------------------------
def _build_filter(allowed_labels: List[str]) -> Optional[dict]:
    """
    Translate a role's allowed labels into a Chroma ``where`` filter.

    ``["*"]`` means no filter (admin / CLevel see everything).
    """
    if "*" in allowed_labels:
        return None
    # Chroma supports the $in operator for metadata filtering.
    return {"access_label": {"$in": allowed_labels}}


def retrieve(question: str, allowed_labels: List[str], top_k: int = 4) -> List[SourceChunk]:
    """
    Embed the question, run a similarity search, and apply the metadata filter
    so chunks the caller is not entitled to read are *never returned*.
    """
    collection = get_collection()
    embedder = get_embedder()

    where = _build_filter(allowed_labels)
    q_emb = embedder.embed_query(question)

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    chunks: List[SourceChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        snippet = doc if len(doc) <= 600 else doc[:600] + "..."
        chunks.append(SourceChunk(
            access_label=meta.get("access_label", "UNKNOWN"),
            source=meta.get("source", "unknown"),
            snippet=snippet,
            score=float(1.0 - dist) if dist is not None else None,  # cosine sim
        ))
    return chunks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an internal assistant for TEST Corporation. "
    "Answer the user's question using ONLY the context provided. "
    "If the context does not contain enough information, say 'I do not have "
    "enough information in the documents accessible to your role.' "
    "Never reveal information beyond what the context provides. "
    "Cite sources by their access_label and filename when relevant."
)


def _format_context(chunks: List[SourceChunk]) -> str:
    if not chunks:
        return "(no context available)"
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] access_label={c.access_label} source={c.source}\n{c.snippet}"
        )
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[SourceChunk]) -> str:
    """Compose a prompt from the authorized chunks and ask the LLM."""
    if not chunks:
        return (
            "I do not have enough information in the documents accessible "
            "to your role to answer that."
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    try:
        llm = get_llm()
        return llm.invoke(prompt).strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        # Graceful fallback so the API stays usable even when Ollama is down.
        return (
            "[LLM unavailable] Based on the retrieved context, here is what I found:\n\n"
            + _format_context(chunks)
        )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def answer_query(
    question: str,
    allowed_labels: List[str],
    top_k: int = 4,
) -> Tuple[str, List[SourceChunk]]:
    chunks = retrieve(question, allowed_labels, top_k=top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks
