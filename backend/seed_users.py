"""
Seed a starter set of users so the app is usable out-of-the-box.

Default credentials (CHANGE BEFORE DEPLOYING):

    username  role     password
    --------- -------- ------------
    admin     ADMIN    admin123
    ceo       CLevel   ceo123
    eng       ENGG     eng123
    sales     SALES    sales123
    hr        HR       hr123
    intern    INTERN   intern123

Run::

    python seed_users.py
"""
from __future__ import annotations

import logging

from auth import hash_password
from database import SessionLocal, init_db
from models import Role, UserORM

SEED = [
    ("admin",  Role.ADMIN.value,  "admin123",  "Admin User"),
    ("ceo",    Role.CLEVEL.value, "ceo123",    "Carla CEO"),
    ("eng",    Role.ENGG.value,   "eng123",    "Eli Engineer"),
    ("sales",  Role.SALES.value,  "sales123",  "Sasha Sales"),
    ("hr",     Role.HR.value,     "hr123",     "Hank HR"),
    ("intern", Role.INTERN.value, "intern123", "Iris Intern"),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()
    db = SessionLocal()
    try:
        created, skipped = 0, 0
        for username, role, password, full_name in SEED:
            existing = db.query(UserORM).filter(UserORM.username == username).first()
            if existing:
                skipped += 1
                continue
            db.add(UserORM(
                username=username,
                hashed_password=hash_password(password),
                role=role,
                full_name=full_name,
            ))
            created += 1
        db.commit()
        print(f"Seed complete: created={created}, skipped={skipped}")
        print("\nLogin credentials:")
        for username, role, password, _ in SEED:
            print(f"  {username:8s}  /  {password:10s}  (role: {role})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
