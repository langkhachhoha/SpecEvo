#!/usr/bin/env python3
"""Smoke-test OpenRouter API key loaded from .env.

Checks auth via GET /auth/key, then a tiny chat completion (~fraction of a cent).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

BASE_URL = "https://openrouter.ai/api/v1"
TEST_MODEL = "openrouter/openai/gpt-4o-mini"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    root = _repo_root()
    load_dotenv(root / ".env")
    load_dotenv(root / ".env")


def _resolve_api_key() -> str:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key.startswith("sk-or-"):
        os.environ.setdefault("OPENROUTER_API_KEY", openai_key)
        return openai_key
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def _request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def check_auth_key(api_key: str) -> dict:
    return _request("GET", "/auth/key", api_key)


def check_chat_completion(api_key: str) -> dict:
    return _request(
        "POST",
        "/chat/completions",
        api_key,
        {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )


def main() -> int:
    _load_env()
    api_key = _resolve_api_key()
    if not api_key:
        print(
            "Khong tim thay API key. Dat OPENAI_API_KEY hoac OPENROUTER_API_KEY trong .env.",
            file=sys.stderr,
        )
        return 1

    print(f"Key: {_mask_key(api_key)}")
    print(f"Model test: {TEST_MODEL}")
    print()

    # 1) Auth / key metadata
    print("[1/2] GET /auth/key ...")
    try:
        auth = check_auth_key(api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"FAIL — HTTP {exc.code}")
        if body:
            print(body)
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL — khong ket noi duoc: {exc.reason}")
        return 1

    data = auth.get("data") or {}
    label = data.get("label") or "(no label)"
    limit = data.get("limit")
    usage = data.get("usage")
    print(f"OK — label={label!r}, usage={usage}, limit={limit}")

    # 2) Minimal completion
    print("[2/2] POST /chat/completions (max_tokens=8) ...")
    try:
        chat = check_chat_completion(api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"FAIL — HTTP {exc.code}")
        if body:
            print(body)
        if exc.code == 403 and "limit" in body.lower():
            print()
            print("Key hop le nhung da vuot credit/limit. Tang limit tren OpenRouter hoac doi reset.")
            return 2
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL — khong ket noi duoc: {exc.reason}")
        return 1

    reply = chat["choices"][0]["message"]["content"].strip()
    model = chat.get("model", "?")
    usage_info = chat.get("usage") or {}
    print(f"OK — model={model!r}, reply={reply!r}")
    print(f"     tokens: prompt={usage_info.get('prompt_tokens')}, "
          f"completion={usage_info.get('completion_tokens')}")

    print()
    print("OpenRouter API key hoat dong binh thuong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
