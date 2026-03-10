from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any


_GEMINI_COUNT_TOKENS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:countTokens"
)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return ascii_only or "item"


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    upper = without_marks.upper()
    upper = upper.replace("İ", "I").replace("ı", "I")
    upper = re.sub(r"[ \t]+", " ", upper)
    upper = re.sub(r"\s+\n", "\n", upper)
    upper = re.sub(r"\n{3,}", "\n\n", upper)
    return upper.strip()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_tokens_gemini(text: str, api_key: str) -> int:
    """Count tokens using Google Gemini countTokens endpoint.

    Returns 0 on failure so cost tracking degrades gracefully.
    """
    if not text or not api_key:
        return 0
    url = f"{_GEMINI_COUNT_TOKENS_URL}?key={urllib.parse.quote(api_key, safe='')}"
    payload = json.dumps({"contents": [{"parts": [{"text": text}]}]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return int(result.get("totalTokens", 0))
    except Exception:
        return 0
