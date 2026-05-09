"""
auth.py - JWT-based authentication, password hashing, and the
``get_current_active_user`` dependency that drives RBAC.

Security notes
--------------
* Passwords are hashed with bcrypt via passlib.
* Tokens are signed with HS256. The signing secret comes from the
  ``JWT_SECRET`` environment variable (a default is provided ONLY for local dev).
* The token carries ``sub`` (username) and ``role``. The role is re-checked
  against the database on every request to defeat token-only privilege escalation
  (a stale token whose user has been demoted will reflect the *current* role).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import get_db
from models import PERMISSION_MAP, UserORM, UserPublic, labels_for_role

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod-please-this-is-only-for-dev")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(
    *,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# ---------------------------------------------------------------------------
# DB lookups
# ---------------------------------------------------------------------------
def authenticate_user(db: Session, username: str, password: str) -> Optional[UserORM]:
    user = db.query(UserORM).filter(UserORM.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def to_public(user: UserORM) -> UserPublic:
    return UserPublic(
        username=user.username,
        role=user.role,
        full_name=user.full_name,
        accessible_folders=labels_for_role(user.role),
    )


# ---------------------------------------------------------------------------
# FastAPI dependency: get_current_active_user
# ---------------------------------------------------------------------------
_credentials_exc = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_active_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserORM:
    """
    Validate the JWT, look the user up in SQLite, and return the ORM object.

    We deliberately re-fetch the user on every request so that role changes
    take effect immediately - we don't trust the role claim from the token alone.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            raise _credentials_exc
    except JWTError:
        raise _credentials_exc

    user = db.query(UserORM).filter(UserORM.username == username).first()
    if user is None:
        raise _credentials_exc

    if user.role not in PERMISSION_MAP:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role}' is not recognized.",
        )
    return user
