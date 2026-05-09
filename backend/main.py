"""
main.py - FastAPI entrypoint for the RBAC RAG service.

Endpoints
---------
POST /api/auth/login       OAuth2 form login -> JWT
GET  /api/auth/me          Current user + accessible folders
POST /api/query            Authenticated, role-filtered RAG query
GET  /api/folders          Folders the current role can access
GET  /api/health           Liveness probe

Run locally::

    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    to_public,
)
from database import get_db, init_db
from models import (
    PERMISSION_MAP,
    QueryRequest,
    QueryResponse,
    Token,
    UserORM,
    UserPublic,
    labels_for_role,
)
from vector_service import answer_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rbac-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("RBAC-RAG API starting up. Database initialised.")
    yield
    logger.info("RBAC-RAG API shutting down.")


app = FastAPI(
    title="RBAC RAG API",
    description="Role-Based Access-Controlled Retrieval Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - Angular dev server runs on :4200 by default.
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:4200,http://127.0.0.1:4200",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/api/auth/login", response_model=Token, tags=["auth"])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(username=user.username, role=user.role)
    return Token(access_token=token, user=to_public(user))


@app.get("/api/auth/me", response_model=UserPublic, tags=["auth"])
def me(current: UserORM = Depends(get_current_active_user)) -> UserPublic:
    return to_public(current)


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
@app.get("/api/folders", tags=["rbac"])
def list_folders(current: UserORM = Depends(get_current_active_user)) -> dict:
    """Return the data rooms the current user is authorized to read."""
    labels = labels_for_role(current.role)
    if "*" in labels:
        labels = sorted(set(PERMISSION_MAP[current.role]) | {
            "PUBLIC", "ENGG", "SALES", "CLevel", "HR", "INTERN",
        })
        labels = [l for l in labels if l != "*"]
    return {
        "role": current.role,
        "accessible_folders": labels,
    }


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------
@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
def query(
    body: QueryRequest,
    current: UserORM = Depends(get_current_active_user),
) -> QueryResponse:
    """
    Run a RAG query.

    The Silent Guard pattern lives here: we resolve the caller's allowed labels
    and pass them straight into the vector search. The LLM never sees chunks
    the caller is not entitled to read.
    """
    allowed = labels_for_role(current.role)
    logger.info(
        "RAG query user=%s role=%s allowed=%s q=%r",
        current.username, current.role, allowed, body.question[:120],
    )
    answer, chunks = answer_query(body.question, allowed, top_k=body.top_k)

    accessed: List[str] = sorted({c.access_label for c in chunks})
    return QueryResponse(answer=answer, sources=chunks, accessed_labels=accessed)
