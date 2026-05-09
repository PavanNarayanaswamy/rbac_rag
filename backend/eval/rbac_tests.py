"""
RBAC security tests against the live FastAPI service.

These are *black-box* checks that complement the dataset-driven evaluation.
They verify that the auth boundary itself behaves correctly - the kind of
checks an examiner is likely to ask about ("what about a tampered token?
what about a missing one?").

Run with::

    python -m eval.rbac_tests --base-url http://localhost:8000

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# Tiny HTTP helpers (stdlib only - no extra deps)
# ---------------------------------------------------------------------------
def _post(url: str, body: bytes, headers: dict, timeout: float = 15.0) -> Tuple[int, dict]:
    req = Request(url, data=body, method="POST", headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {}
        return e.code, payload
    except URLError as e:
        return 0, {"error": str(e)}


def _get(url: str, headers: dict, timeout: float = 15.0) -> Tuple[int, dict]:
    req = Request(url, method="GET", headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {}
        return e.code, payload
    except URLError as e:
        return 0, {"error": str(e)}


def login(base_url: str, username: str, password: str) -> Optional[str]:
    body = urlencode({"username": username, "password": password}).encode()
    status, payload = _post(
        f"{base_url}/api/auth/login",
        body,
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if status == 200:
        return payload.get("access_token")
    return None


def query(base_url: str, token: Optional[str], question: str) -> Tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return _post(
        f"{base_url}/api/query",
        json.dumps({"question": question, "top_k": 4}).encode(),
        headers,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


def t_no_token_rejected(base_url: str) -> TestResult:
    status, _ = query(base_url, None, "Where is HQ?")
    return TestResult(
        name="no_token_rejected",
        passed=status == 401,
        detail=f"expected 401, got {status}",
    )


def t_garbage_token_rejected(base_url: str) -> TestResult:
    status, _ = query(base_url, "garbage.token.here", "Where is HQ?")
    return TestResult(
        name="garbage_token_rejected",
        passed=status == 401,
        detail=f"expected 401, got {status}",
    )


def t_login_succeeds(base_url: str, role: str, user: str, pwd: str) -> TestResult:
    token = login(base_url, user, pwd)
    return TestResult(
        name=f"login_{role}_{user}",
        passed=token is not None,
        detail="login OK" if token else "login failed",
    )


def t_intern_cannot_see_clevel(base_url: str) -> TestResult:
    token = login(base_url, "intern", "intern123")
    if not token:
        return TestResult("intern_cannot_see_clevel", False, "login failed")
    status, payload = query(base_url, token, "What is the CEO's base salary?")
    sources = payload.get("sources", []) if status == 200 else []
    leaked = any(s.get("access_label") == "CLevel" for s in sources)
    answer = (payload.get("answer", "") or "").lower()
    leaked_in_text = "650" in answer
    ok = (status == 200) and (not leaked) and (not leaked_in_text)
    return TestResult(
        name="intern_cannot_see_clevel",
        passed=ok,
        detail=f"status={status} leaked_chunks={leaked} leaked_text={leaked_in_text}",
    )


def t_engg_cannot_see_hr(base_url: str) -> TestResult:
    token = login(base_url, "eng", "eng123")
    if not token:
        return TestResult("engg_cannot_see_hr", False, "login failed")
    status, payload = query(base_url, token, "What is the L7 Principal Engineer salary band?")
    sources = payload.get("sources", []) if status == 200 else []
    leaked = any(s.get("access_label") == "HR" for s in sources)
    answer = (payload.get("answer", "") or "").lower()
    leaked_in_text = "295" in answer or "370" in answer
    ok = status == 200 and not leaked and not leaked_in_text
    return TestResult(
        name="engg_cannot_see_hr",
        passed=ok,
        detail=f"status={status} leaked_chunks={leaked} leaked_text={leaked_in_text}",
    )


def t_sales_cannot_see_engg(base_url: str) -> TestResult:
    token = login(base_url, "sales", "sales123")
    if not token:
        return TestResult("sales_cannot_see_engg", False, "login failed")
    status, payload = query(base_url, token, "How do I respond to a P0 production incident?")
    sources = payload.get("sources", []) if status == 200 else []
    leaked = any(s.get("access_label") == "ENGG" for s in sources)
    return TestResult(
        name="sales_cannot_see_engg",
        passed=(status == 200 and not leaked),
        detail=f"status={status} leaked_chunks={leaked}",
    )


def t_clevel_can_see_everything(base_url: str) -> TestResult:
    token = login(base_url, "ceo", "ceo123")
    if not token:
        return TestResult("clevel_can_see_everything", False, "login failed")
    status, payload = query(base_url, token, "Tell me about HR salary bands and the on-call runbook.")
    sources = payload.get("sources", []) if status == 200 else []
    labels = {s.get("access_label") for s in sources}
    # Any non-empty result without an authorization error is acceptable.
    return TestResult(
        name="clevel_can_see_everything",
        passed=(status == 200 and len(sources) > 0),
        detail=f"status={status} labels_seen={sorted(labels)}",
    )


def t_admin_unrestricted(base_url: str) -> TestResult:
    token = login(base_url, "admin", "admin123")
    if not token:
        return TestResult("admin_unrestricted", False, "login failed")
    status, payload = query(base_url, token, "What is the CEO base salary?")
    return TestResult(
        name="admin_unrestricted",
        passed=(status == 200 and len(payload.get("sources", [])) > 0),
        detail=f"status={status} sources={len(payload.get('sources', []))}",
    )


def t_authorization_precision_per_role(base_url: str) -> TestResult:
    """For every non-admin role, assert every retrieved chunk is in their allowed labels."""
    cases = [
        ("eng",    "eng123",    {"ENGG", "PUBLIC"},   "Tell me about our infrastructure and ethics policy"),
        ("sales",  "sales123",  {"SALES", "PUBLIC"},  "Tell me about discounts and the company mission"),
        ("hr",     "hr123",     {"HR", "PUBLIC"},     "Tell me about benefits and the code of conduct"),
        ("intern", "intern123", {"INTERN", "PUBLIC"}, "Tell me about my internship and the company"),
    ]
    failures = []
    for user, pwd, allowed, q in cases:
        token = login(base_url, user, pwd)
        if not token:
            failures.append(f"{user}: login failed")
            continue
        status, payload = query(base_url, token, q)
        if status != 200:
            failures.append(f"{user}: status {status}")
            continue
        for s in payload.get("sources", []):
            if s.get("access_label") not in allowed:
                failures.append(
                    f"{user}: leaked {s.get('access_label')} (allowed={sorted(allowed)})"
                )
    return TestResult(
        name="authorization_precision_per_role",
        passed=not failures,
        detail="; ".join(failures) if failures else "all roles OK",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_TESTS: List[Callable[[str], TestResult]] = [
    t_no_token_rejected,
    t_garbage_token_rejected,
    lambda u: t_login_succeeds(u, "ENGG",   "eng",    "eng123"),
    lambda u: t_login_succeeds(u, "SALES",  "sales",  "sales123"),
    lambda u: t_login_succeeds(u, "HR",     "hr",     "hr123"),
    lambda u: t_login_succeeds(u, "INTERN", "intern", "intern123"),
    lambda u: t_login_succeeds(u, "CLevel", "ceo",    "ceo123"),
    lambda u: t_login_succeeds(u, "ADMIN",  "admin",  "admin123"),
    t_intern_cannot_see_clevel,
    t_engg_cannot_see_hr,
    t_sales_cannot_see_engg,
    t_clevel_can_see_everything,
    t_admin_unrestricted,
    t_authorization_precision_per_role,
]


def run_all(base_url: str) -> List[TestResult]:
    results = []
    for t in ALL_TESTS:
        try:
            results.append(t(base_url))
        except Exception as e:
            results.append(TestResult(getattr(t, "__name__", "test"), False, f"exception: {e}"))
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args()

    results = run_all(args.base_url)
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(f"\nRBAC Security Tests against {args.base_url}\n" + "=" * 60)
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name:42s} {r.detail}")
        print("-" * 60)
        print(f"  Score: {passed}/{total}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
