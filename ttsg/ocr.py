from __future__ import annotations

import base64
import json
import re as _re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .models import OCRDocument, OCRPage
from .utils import normalize_text, sha256_bytes, sha256_file


MISTRAL_OCR_ENDPOINT = "https://api.mistral.ai/v1/ocr"
MISTRAL_DEFAULT_MODEL = "mistral-ocr-2512"
MISTRAL_PRICE_PER_1K_PAGES = 2.0

GEMINI_GENERATE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_COUNT_TOKENS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
GEMINI_DEFAULT_MODEL = "gemini-3-flash-preview"
GEMINI_PRICE_INPUT_PER_1M = 0.50
GEMINI_PRICE_OUTPUT_PER_1M = 3.0
GEMINI_OCR_PROMPT = (
    "Convert this scanned document page to markdown. "
    "Preserve the exact text, layout, tables and formatting. "
    "Output only the markdown, no explanations."
)

REPLICATE_PREDICTIONS_ENDPOINT = "https://api.replicate.com/v1/models/google/gemini-3-flash/predictions"
REPLICATE_FILES_ENDPOINT = "https://api.replicate.com/v1/files"
REPLICATE_POLL_INTERVAL = 2.0
REPLICATE_POLL_TIMEOUT = 300.0


class OCRProvider(ABC):
    name: str

    @abstractmethod
    def process(self, doc_id: str, image_paths: list[Path], pdf_sha256: str) -> OCRDocument:
        raise NotImplementedError


class MistralOCRProvider(OCRProvider):
    name = "mistral"

    def __init__(self, api_key: str, model: str = MISTRAL_DEFAULT_MODEL) -> None:
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is required for the Mistral OCR provider.")
        self.api_key = api_key
        self.model = model

    def process(self, doc_id: str, image_paths: list[Path], pdf_sha256: str) -> OCRDocument:
        pages: list[OCRPage] = []
        started = datetime.now(timezone.utc).isoformat()
        for page_number, image_path in enumerate(image_paths, start=1):
            markdown, runtime_ms, usage, cost = self._ocr_single_image(image_path)
            pages.append(
                OCRPage(
                    provider=self.name,
                    doc_id=doc_id,
                    page=page_number,
                    source_image=str(image_path),
                    raw_markdown=markdown,
                    runtime_ms=runtime_ms,
                    usage=usage,
                    cost_usd=cost,
                    sha256=sha256_bytes(markdown.encode("utf-8")),
                    notes=[],
                )
            )
        return OCRDocument(
            provider=self.name,
            doc_id=doc_id,
            pages=pages,
            created_at=started,
            input_sha256=pdf_sha256,
            notes=[],
        )

    def process_tiled(
        self, doc_id: str, page_images: list[Path], tile_groups: list[list[Path]], pdf_sha256: str,
    ) -> OCRDocument:
        """OCR each page via its tiles, merge per page, return one OCRDocument."""
        pages: list[OCRPage] = []
        started = datetime.now(timezone.utc).isoformat()
        for page_number, (page_image, tiles) in enumerate(
            zip(page_images, tile_groups), start=1,
        ):
            tile_markdowns: list[str] = []
            total_runtime = 0
            total_cost = 0.0
            merged_usage: dict = {}
            for tile_path in tiles:
                md, rt, usage, cost = self._ocr_single_image(tile_path)
                tile_markdowns.append(md)
                total_runtime += rt
                total_cost += cost
                for key, val in usage.items():
                    if isinstance(val, (int, float)):
                        merged_usage[key] = merged_usage.get(key, 0) + val
            merged_md = merge_tile_markdowns(tile_markdowns)
            pages.append(
                OCRPage(
                    provider=self.name,
                    doc_id=doc_id,
                    page=page_number,
                    source_image=str(page_image),
                    raw_markdown=merged_md,
                    runtime_ms=total_runtime,
                    usage=merged_usage,
                    cost_usd=total_cost,
                    sha256=sha256_bytes(merged_md.encode("utf-8")),
                    notes=[f"tiles={len(tiles)}"],
                )
            )
        return OCRDocument(
            provider=self.name,
            doc_id=doc_id,
            pages=pages,
            created_at=started,
            input_sha256=pdf_sha256,
            notes=[f"tile_mode={len(tile_groups[0]) if tile_groups else 0}"],
        )

    def _ocr_single_image(self, image_path: Path) -> tuple[str, int, dict, float]:
        """OCR a single image.  Returns (markdown, runtime_ms, usage, cost_usd)."""
        with image_path.open("rb") as handle:
            image_bytes = handle.read()
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "document": {
                "type": "image_url",
                "image_url": data_uri,
            },
            "include_image_base64": False,
        }
        request = urllib.request.Request(
            MISTRAL_OCR_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        page_start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mistral OCR error for {image_path.name}: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Mistral OCR network error for {image_path.name}: {exc}") from exc
        runtime_ms = int((time.perf_counter() - page_start) * 1000)

        returned_pages = result.get("pages") or []
        if not returned_pages:
            raise RuntimeError(f"Mistral OCR returned no pages for {image_path.name}.")
        markdown = returned_pages[0].get("markdown", "")
        usage = result.get("usage_info", {})
        return markdown, runtime_ms, usage, MISTRAL_PRICE_PER_1K_PAGES / 1000.0


# ---------------------------------------------------------------------------
# Tile markdown merge – de-duplicate overlapping lines between consecutive
# tiles.  We normalise both sides and locate the longest overlapping run.
# ---------------------------------------------------------------------------

def merge_tile_markdowns(tile_mds: list[str]) -> str:
    """Merge markdown outputs from overlapping tiles into one document."""
    if not tile_mds:
        return ""
    if len(tile_mds) == 1:
        return tile_mds[0]

    merged = tile_mds[0]
    for next_md in tile_mds[1:]:
        merged = _merge_two(merged, next_md)
    return merged


def _merge_two(upper: str, lower: str) -> str:
    """Merge two consecutive tiles, removing duplicated overlap lines.

    Uses similarity-based matching because OCR of overlapping image
    regions rarely produces byte-identical text.
    """
    upper_lines = upper.splitlines()
    lower_lines = lower.splitlines()
    if not upper_lines or not lower_lines:
        return (upper + "\n" + lower).strip()

    norm_upper = [normalize_text(l) for l in upper_lines]
    norm_lower = [normalize_text(l) for l in lower_lines]

    # 1) Exact multi-line overlap
    best_overlap = 0
    max_check = min(len(norm_upper), len(norm_lower), 40)
    for n in range(1, max_check + 1):
        if norm_upper[-n:] == norm_lower[:n]:
            best_overlap = n
    if best_overlap >= 2:
        return "\n".join(upper_lines + lower_lines[best_overlap:])

    # 2) Fuzzy block overlap — score how well upper's tail matches lower's head
    search_depth = min(len(norm_upper), len(norm_lower), 40)
    best_score = 0.0
    best_cut = 0
    for candidate in range(2, search_depth + 1):
        tail = [l for l in norm_upper[-candidate:] if l.strip()]
        head = [l for l in norm_lower[:candidate] if l.strip()]
        if len(tail) < 2 or len(head) < 2:
            continue
        ratio = SequenceMatcher(None, "\n".join(tail), "\n".join(head)).ratio()
        if ratio > best_score:
            best_score = ratio
            best_cut = candidate
    if best_score >= 0.70 and best_cut >= 2:
        return "\n".join(upper_lines + lower_lines[best_cut:])

    # 3) Single-line anchor — find a substantial upper line in lower's head
    for ui in range(len(norm_upper) - 1, max(len(norm_upper) - 20, -1), -1):
        anchor = norm_upper[ui]
        if len(anchor.strip()) < 10:
            continue
        for li in range(min(20, len(norm_lower))):
            cand = norm_lower[li]
            if len(cand.strip()) < 10:
                continue
            if SequenceMatcher(None, anchor, cand).ratio() >= 0.80:
                return "\n".join(upper_lines + lower_lines[li + 1:])

    # 4) No overlap detected
    return "\n".join(upper_lines + lower_lines)


class GeminiOCRProvider(OCRProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = GEMINI_DEFAULT_MODEL, replicate_token: str | None = None, prefer_replicate: bool = False) -> None:
        if not api_key and not replicate_token:
            raise RuntimeError("GEMINI_API_KEY or REPLICATE_API_TOKEN is required for the Gemini OCR provider.")
        self.api_key = api_key
        self.model = model
        self.replicate_token = replicate_token
        self.prefer_replicate = prefer_replicate

    def process(self, doc_id: str, image_paths: list[Path], pdf_sha256: str) -> OCRDocument:
        pages: list[OCRPage] = []
        started = datetime.now(timezone.utc).isoformat()
        for page_number, image_path in enumerate(image_paths, start=1):
            with image_path.open("rb") as handle:
                image_bytes = handle.read()
            b64_data = base64.b64encode(image_bytes).decode("ascii")

            page_start = time.perf_counter()
            markdown, usage, notes = self._call_page(b64_data, image_path.name)
            runtime_ms = int((time.perf_counter() - page_start) * 1000)

            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            cost_usd = (
                input_tokens * GEMINI_PRICE_INPUT_PER_1M / 1_000_000
                + output_tokens * GEMINI_PRICE_OUTPUT_PER_1M / 1_000_000
            )
            pages.append(
                OCRPage(
                    provider=self.name,
                    doc_id=doc_id,
                    page=page_number,
                    source_image=str(image_path),
                    raw_markdown=markdown,
                    runtime_ms=runtime_ms,
                    usage=usage,
                    cost_usd=cost_usd,
                    sha256=sha256_bytes(markdown.encode("utf-8")),
                    notes=notes,
                )
            )
        return OCRDocument(
            provider=self.name,
            doc_id=doc_id,
            pages=pages,
            created_at=started,
            input_sha256=pdf_sha256,
            notes=[],
        )

    def _call_page(self, b64_data: str, label: str) -> tuple[str, dict, list[str]]:
        if self.prefer_replicate and self.replicate_token:
            return self._call_replicate(b64_data, label)
        if self.api_key:
            try:
                return self._call_google(b64_data, label)
            except RuntimeError:
                if not self.replicate_token:
                    raise
                print(f"[gemini] Google API failed for {label}, falling back to Replicate")
        return self._call_replicate(b64_data, label)

    def _call_google(self, b64_data: str, label: str) -> tuple[str, dict, list[str]]:
        endpoint = GEMINI_GENERATE_ENDPOINT.format(model=self.model)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": GEMINI_OCR_PROMPT},
                        {"inline_data": {"mime_type": "image/png", "data": b64_data}},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            f"{endpoint}?key={self.api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        result = _gemini_request_with_retry(request, label)
        candidates = result.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates for {label}.")
        parts = candidates[0].get("content", {}).get("parts", [])
        markdown = parts[0].get("text", "") if parts else ""
        usage = result.get("usageMetadata", {})
        return markdown, usage, []

    def _call_replicate(self, b64_data: str, label: str) -> tuple[str, dict, list[str]]:
        if not self.replicate_token:
            raise RuntimeError("REPLICATE_API_TOKEN is required for the Replicate fallback.")
        image_url = _replicate_upload(self.replicate_token, base64.b64decode(b64_data), label)
        payload = {
            "input": {
                "prompt": GEMINI_OCR_PROMPT,
                "images": [image_url],
            },
        }
        request = urllib.request.Request(
            REPLICATE_PREDICTIONS_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.replicate_token}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Replicate error for {label}: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Replicate network error for {label}: {exc}") from exc

        if result.get("status") not in ("succeeded",):
            result = _replicate_poll(result, self.replicate_token, label)

        output = result.get("output", "")
        if isinstance(output, list):
            output = "".join(str(chunk) for chunk in output)
        output_tokens = _count_tokens(self.api_key, self.model, output) if self.api_key else max(1, len(output) // 4)
        prompt_tokens = _count_tokens(self.api_key, self.model, GEMINI_OCR_PROMPT, b64_image=b64_data) if self.api_key else 300
        usage = {
            "backend": "replicate",
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
        }
        return output, usage, ["replicate_fallback"]


GEMINI_MAX_RETRIES = 5


def _count_tokens(api_key: str, model: str, text: str, b64_image: str | None = None) -> int:
    endpoint = GEMINI_COUNT_TOKENS_ENDPOINT.format(model=model)
    parts: list[dict] = [{"text": text}]
    if b64_image:
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64_image}})
    payload = {"contents": [{"parts": parts}]}
    request = urllib.request.Request(
        f"{endpoint}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("totalTokens", 0)
    except Exception:
        return max(1, len(text) // 4)


def _gemini_request_with_retry(request: urllib.request.Request, label: str) -> dict:
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < GEMINI_MAX_RETRIES - 1:
                retry_seconds = _parse_retry_delay(body)
                print(f"[gemini] 429 rate limit for {label}, retrying in {retry_seconds}s (attempt {attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(retry_seconds)
                request = urllib.request.Request(
                    request.full_url,
                    data=request.data,
                    headers=dict(request.headers),
                    method=request.get_method(),
                )
                continue
            raise RuntimeError(f"Gemini OCR error for {label}: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini OCR network error for {label}: {exc}") from exc
    raise RuntimeError(f"Gemini OCR failed for {label} after {GEMINI_MAX_RETRIES} retries")


def _parse_retry_delay(body: str) -> float:
    try:
        import re as _re
        match = _re.search(r'"retryDelay"\s*:\s*"(\d+)s"', body)
        if match:
            return float(match.group(1)) + 1.0
    except Exception:
        pass
    return 30.0


def _replicate_upload(token: str, image_bytes: bytes, label: str) -> str:
    import uuid
    boundary = uuid.uuid4().hex
    filename = label.replace("/", "_")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="content"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        REPLICATE_FILES_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate file upload error for {label}: {exc.code} {err_body}") from exc
    url = result.get("urls", {}).get("get")
    if not url:
        raise RuntimeError(f"Replicate file upload for {label} returned no URL: {result}")
    return url


def _replicate_poll(prediction: dict, token: str, label: str) -> dict:
    poll_url = prediction.get("urls", {}).get("get")
    if not poll_url:
        raise RuntimeError(f"Replicate prediction for {label} has no poll URL.")
    deadline = time.perf_counter() + REPLICATE_POLL_TIMEOUT
    while time.perf_counter() < deadline:
        time.sleep(REPLICATE_POLL_INTERVAL)
        request = urllib.request.Request(
            poll_url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        status = result.get("status")
        if status == "succeeded":
            return result
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate prediction for {label} {status}: {result.get('error', '')}")
    raise RuntimeError(f"Replicate prediction for {label} timed out after {REPLICATE_POLL_TIMEOUT}s")


def build_provider(
    name: str,
    *,
    mistral_api_key: str | None,
    gemini_api_key: str | None = None,
    replicate_api_token: str | None = None,
    prefer_replicate: bool = False,
) -> OCRProvider:
    if name == "mistral":
        return MistralOCRProvider(api_key=mistral_api_key or "")
    if name == "gemini":
        return GeminiOCRProvider(
            api_key=gemini_api_key or "",
            replicate_token=replicate_api_token,
            prefer_replicate=prefer_replicate,
        )
    raise ValueError(f"Unsupported provider: {name}")


def pdf_digest(path: Path) -> str:
    return sha256_file(path)
