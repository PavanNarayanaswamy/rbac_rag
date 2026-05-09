"""
Pydantic models and SQLAlchemy ORM models.

Roles:
    - ADMIN  : super-admin, can access every folder.
    - CLevel : executives, can access every folder.
    - ENGG   : engineering, can access ENGG + PUBLIC.
    - SALES  : sales, can access SALES + PUBLIC.
    - HR     : human resources, can access HR + PUBLIC.
    - INTERN : interns, can access INTERN + PUBLIC.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ---------------------------------------------------------------------------
# Roles & permission map
# ---------------------------------------------------------------------------
class Role(str, Enum):
    ADMIN = "ADMIN"
    CLEVEL = "CLevel"
    ENGG = "ENGG"
    SALES = "SALES"
    HR = "HR"
    INTERN = "INTERN"


# A role -> list of access_labels it is allowed to read from the vector store.
# CLevel and ADMIN can see everything (the wildcard "*" is interpreted by
# vector_service to skip the metadata filter entirely).
PERMISSION_MAP: dict[str, List[str]] = {
    Role.ADMIN.value:  ["*"],
    Role.CLEVEL.value: ["*"],
    Role.ENGG.value:   ["ENGG", "PUBLIC"],
    Role.SALES.value:  ["SALES", "PUBLIC"],
    Role.HR.value:     ["HR", "PUBLIC"],
    Role.INTERN.value: ["INTERN", "PUBLIC"],
}


def labels_for_role(role: str) -> List[str]:
    """Return the access_labels a given role is allowed to retrieve."""
    return PERMISSION_MAP.get(role, ["PUBLIC"])


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------
class UserORM(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False)
    full_name = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class UserPublic(BaseModel):
    username: str
    role: str
    full_name: Optional[str] = None
    accessible_folders: List[str] = Field(default_factory=list)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class LoginRequest(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceChunk(BaseModel):
    access_label: str
    source: str
    snippet: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    accessed_labels: List[str]
