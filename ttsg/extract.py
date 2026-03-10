from __future__ import annotations

import difflib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import ConsensusDocument, OCRDocument, SegmentMatch
from .utils import count_tokens_gemini, normalize_text, normalize_whitespace

REPLICATE_CLEAN_ENDPOINT = "https://api.replicate.com/v1/models/google/gemini-3-flash/predictions"
REPLICATE_POLL_INTERVAL = 2.0
REPLICATE_POLL_TIMEOUT = 300.0
_CLEAN_MAX_RETRIES = 3

# Gemini 3.0 Flash pricing (via Replicate)
_GEMINI_FLASH_INPUT_PRICE_PER_1M = 0.50   # USD per 1M input tokens
_GEMINI_FLASH_OUTPUT_PRICE_PER_1M = 3.00   # USD per 1M output tokens

_CLEAN_SYSTEM_PROMPT = (
    "Sen bir belge ayıklama asistanısın. Sana bir Ticaret Sicili Gazetesi "
    "sayfasından OCR ile çıkarılmış metin verilecek. Bu metin birden fazla "
    "şirketin ilanını içerebilir.\n\n"
    "GÖREV: Metinden YALNIZCA \"{company}\" şirketine ait bölümü ayıkla.\n\n"
    "KURALLAR:\n"
    "1. Metni olduğu gibi koru — hiçbir kelimeyi değiştirme, ekleme veya düzeltme yapma.\n"
    "2. Başka şirketlere ait ilanları (farklı ticaret unvanı, farklı MERSİS no, "
    "farklı adres) tamamen çıkar.\n"
    "3. Gazete başlık/sayfa numarası satırları hedef şirkete ait değilse onları da çıkar.\n"
    "4. Sadece ayıkladığın metni döndür, açıklama yapma."
)


TARGET_COMPANY = normalize_text("Parla Enerji Yatırımları Anonim Şirketi")


@dataclass(frozen=True)
class PageLine:
    page: int
    line_index: int
    raw: str
    normalized: str


def extract_target_segment(ocr_document: OCRDocument) -> SegmentMatch:
    lines = _flatten_lines(ocr_document)
    if not lines:
        return SegmentMatch(
            start_index=-1,
            end_index=-1,
            text="",
            status="manual_review",
            reason="empty_ocr",
            source_pages=[],
            provider=ocr_document.provider,
            doc_id=ocr_document.doc_id,
        )

    start_positions = [index for index, line in enumerate(lines) if TARGET_COMPANY in line.normalized]
    if not start_positions:
        # Heading not found — return full OCR text so LLM cleaning can filter it
        full_text = "\n".join(line.raw for line in lines).strip()
        all_pages = sorted({line.page for line in lines})
        return SegmentMatch(
            start_index=0,
            end_index=len(lines),
            text=full_text,
            status="accepted",
            reason="target_heading_not_found_full_text",
            source_pages=all_pages,
            provider=ocr_document.provider,
            doc_id=ocr_document.doc_id,
        )

    start_index = start_positions[0]
    end_index = len(lines)
    reason = "until_document_end"
    for index in range(start_index + 1, len(lines)):
        if is_company_heading(lines[index].normalized):
            end_index = index
            reason = "next_company_heading"
            break

    chosen = lines[start_index:end_index]
    text = "\n".join(line.raw for line in chosen).strip()
    source_pages = sorted({line.page for line in chosen})
    status = "accepted"
    if len(start_positions) > 1:
        status = "manual_review"
        reason = "multiple_target_headings"
    if not text:
        status = "manual_review"
        reason = "empty_segment"
    return SegmentMatch(
        start_index=start_index,
        end_index=end_index,
        text=text,
        status=status,
        reason=reason,
        source_pages=source_pages,
        provider=ocr_document.provider,
        doc_id=ocr_document.doc_id,
    )


def build_consensus(doc_id: str, segments: list[SegmentMatch], review_dir: Path | None = None) -> ConsensusDocument:
    available = [segment for segment in segments if segment.text.strip()]
    if not available:
        return ConsensusDocument(
            doc_id=doc_id,
            normalized_text="",
            status="manual_review",
            providers=[],
            source_pages=[],
            notes=["no_extractions_available"],
        )

    canonical = {
        segment.provider: normalize_whitespace(normalize_text(segment.text)) for segment in available
    }
    unique_texts = {value for value in canonical.values() if value}
    notes: list[str] = []
    if len(available) == 1:
        notes.append("single_provider_only")
        return ConsensusDocument(
            doc_id=doc_id,
            normalized_text=available[0].text.strip(),
            status="accepted",
            providers=[available[0].provider],
            source_pages=available[0].source_pages,
            notes=notes,
        )
    if len(unique_texts) == 1 and all(item.status == "accepted" for item in available):
        exemplar = available[0]
        return ConsensusDocument(
            doc_id=doc_id,
            normalized_text=exemplar.text.strip(),
            status="accepted",
            providers=[item.provider for item in available],
            source_pages=sorted({page for item in available for page in item.source_pages}),
            notes=notes,
        )

    notes.append("provider_mismatch")
    if review_dir is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / f"{doc_id}.md"
        review_path.write_text(_render_manual_review(doc_id, available), encoding="utf-8")
        notes.append(f"manual_review_file={review_path}")

    # Prefer Mistral as primary when available (better structural output)
    primary = available[0]
    for seg in available:
        if seg.provider == "mistral":
            primary = seg
            break
    return ConsensusDocument(
        doc_id=doc_id,
        normalized_text=primary.text.strip(),
        status="accepted_primary",
        providers=[item.provider for item in available],
        source_pages=sorted({page for item in available for page in item.source_pages}),
        notes=notes,
    )


def clean_with_llm(
    text: str,
    company_name: str,
    replicate_token: str,
    *,
    gemini_api_key: str | None = None,
) -> tuple[str, dict]:
    """Call Gemini Flash via Replicate to strip content not belonging to *company_name*.

    Returns (cleaned_text, metrics_dict).
    """
    empty_metrics: dict = {"input_tokens": 0, "output_tokens": 0, "predict_time_s": 0.0, "cost_usd": 0.0}
    if not text.strip():
        return text, empty_metrics
    prompt = _CLEAN_SYSTEM_PROMPT.format(company=company_name)
    full_prompt = f"{prompt}\n\n---\n\n{text}"
    input_tokens = count_tokens_gemini(full_prompt, gemini_api_key) if gemini_api_key else 0
    payload = {"input": {"prompt": full_prompt}}
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(_CLEAN_MAX_RETRIES):
        request = urllib.request.Request(
            REPLICATE_CLEAN_ENDPOINT,
            data=data,
            headers={
                "Authorization": f"Bearer {replicate_token}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _CLEAN_MAX_RETRIES - 1:
                print(f"[clean] 429 rate limit, retrying in 30s (attempt {attempt + 1})")
                time.sleep(30)
                continue
            print(f"[clean] Replicate error {exc.code}, keeping original text")
            return text, empty_metrics
        except urllib.error.URLError as exc:
            print(f"[clean] Network error: {exc}, keeping original text")
            return text, empty_metrics
        # Poll if prediction not yet complete
        if result.get("status") not in ("succeeded",):
            result = _replicate_poll_clean(result, replicate_token)
            if result is None:
                return text, empty_metrics
        # Count output tokens and compute cost
        predict_time = float(result.get("metrics", {}).get("predict_time", 0.0))
        output = result.get("output", "")
        if isinstance(output, list):
            output = "".join(str(chunk) for chunk in output)
        output_tokens = count_tokens_gemini(output, gemini_api_key) if gemini_api_key else 0
        cost_usd = (
            input_tokens * _GEMINI_FLASH_INPUT_PRICE_PER_1M / 1_000_000
            + output_tokens * _GEMINI_FLASH_OUTPUT_PRICE_PER_1M / 1_000_000
        )
        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "predict_time_s": round(predict_time, 2),
            "cost_usd": round(cost_usd, 6),
        }
        cleaned = output.strip() if output.strip() else text
        return cleaned, metrics
    return text, empty_metrics


def _replicate_poll_clean(prediction: dict, token: str) -> dict | None:
    """Poll a Replicate prediction until it succeeds, fails, or times out."""
    poll_url = prediction.get("urls", {}).get("get")
    if not poll_url:
        print("[clean] Replicate prediction has no poll URL, keeping original text")
        return None
    deadline = time.perf_counter() + REPLICATE_POLL_TIMEOUT
    while time.perf_counter() < deadline:
        time.sleep(REPLICATE_POLL_INTERVAL)
        request = urllib.request.Request(
            poll_url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[clean] Poll error: {exc}, keeping original text")
            return None
        status = result.get("status")
        if status == "succeeded":
            return result
        if status in ("failed", "canceled"):
            print(f"[clean] Replicate prediction {status}: {result.get('error', '')}")
            return None
    print(f"[clean] Replicate prediction timed out after {REPLICATE_POLL_TIMEOUT}s")
    return None





def is_company_heading(normalized_line: str) -> bool:
    if not normalized_line:
        return False
    if any(token in normalized_line for token in ("TURKIYE TICARET SICILI GAZETESI", "TICARET SICIL MUDURLUGU", "MERSIS")):
        return False
    if ":" in normalized_line or normalized_line.startswith(("SAYFA", "NO ", "MADDE ")):
        return False
    suffixes = ("ANONIM SIRKETI", "LIMITED SIRKETI", "LIMITED SIRKET", "KOOPERATIFI")
    return normalized_line.endswith(suffixes)


def _flatten_lines(ocr_document: OCRDocument) -> list[PageLine]:
    flattened: list[PageLine] = []
    for page in ocr_document.pages:
        for line_index, raw_line in enumerate(page.raw_markdown.splitlines()):
            raw = raw_line.rstrip()
            normalized = normalize_text(raw)
            if normalized:
                flattened.append(PageLine(page=page.page, line_index=line_index, raw=raw, normalized=normalized))
    return flattened


def _render_manual_review(doc_id: str, segments: list[SegmentMatch]) -> str:
    parts = [f"# Manual Review: {doc_id}", ""]
    for current in segments:
        parts.append(f"## Provider: {current.provider}")
        parts.append(f"- Status: {current.status}")
        parts.append(f"- Reason: {current.reason}")
        parts.append(f"- Pages: {', '.join(str(page) for page in current.source_pages)}")
        parts.append("")
        parts.append("```text")
        parts.append(current.text.strip())
        parts.append("```")
        parts.append("")
    if len(segments) >= 2:
        parts.append("## Unified Diff")
        parts.append("```diff")
        diff = difflib.unified_diff(
            segments[0].text.splitlines(),
            segments[1].text.splitlines(),
            fromfile=segments[0].provider,
            tofile=segments[1].provider,
            lineterm="",
        )
        parts.extend(diff)
        parts.append("```")
    return "\n".join(parts).strip() + "\n"
