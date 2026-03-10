from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cache import CacheLayout
from .documents import discover_documents
from .extract import TARGET_COMPANY
from .models import (
    ArticleVersion,
    BoardMemberRecord,
    ConsolidationResult,
    CurrentCompanyInfo,
    DocumentMeta,
    SourceRef,
)
from .utils import count_tokens_gemini, normalize_text, slugify, write_json


REPLICATE_PREDICTIONS_ENDPOINT = "https://api.replicate.com/v1/models/google/gemini-3-flash/predictions"
_LLM_POLL_INTERVAL = 2.0
_LLM_POLL_TIMEOUT = 300.0
_LLM_MAX_RETRIES = 3

# Gemini 3.0 Flash pricing (via Replicate)
_GEMINI_FLASH_INPUT_PRICE_PER_1M = 0.50   # USD per 1M input tokens
_GEMINI_FLASH_OUTPUT_PRICE_PER_1M = 3.00   # USD per 1M output tokens

_LLM_EXTRACT_PROMPT = '''Sen bir Ticaret Sicili Gazetesi analiz asistanısın.
Sana OCR ile çıkarılmış gazete metni verilecek. Bu metin birden fazla şirketin ilanını içerebilir.

GÖREV: Metinden YALNIZCA "Parla Enerji Yatırımları Anonim Şirketi" şirketine ait bilgileri çıkar.

Aşağıdaki JSON formatında yanıt ver (sadece JSON, başka bir şey yazma):
{{
  "company_info": {{
    "ticaret_unvani": "str veya null",
    "adres": "str veya null",
    "sermaye": "str veya null (örn: 54.885.000,00 TL)",
    "denetci": "str veya null (tam firma adı)",
    "kurulus_tarihi": "str veya null (gg.aa.yyyy)"
  }},
  "board_events": [
    {{
      "action": "appoint veya remove",
      "name": "ad soyad veya firma adı",
      "role": "Yönetim Kurulu Başkanı / Başkan Yardımcısı / Üyesi veya null",
      "representative": "tüzel kişi adına hareket edecek gerçek kişinin adı soyadı veya null",
      "role_end_date": "str veya null (gg.aa.yyyy)"
    }}
  ],
  "articles": [
    {{
      "article_no": "numara (str)",
      "title": "madde başlığı",
      "body": "madde metninin tamamı — olduğu gibi koru"
    }}
  ]
}}

KURALLAR:
1. Sadece Parla Enerji Yatırımları Anonim Şirketi'ne ait bilgileri çıkar.
2. Başka şirketlere ait bilgileri kesinlikle dahil etme.
3. Metinde yoksa null döndür veya boş liste kullan.
4. Madde metinlerini (body) olduğu gibi koru - düzeltme yapma.
5. İç Yönerge maddelerini dahil etme - sadece Esas Sözleşme maddelerini çıkar.
6. Sadece JSON döndür, açıklama yapma.
7. Sermaye değerini her zaman TL cinsinden yaz (Türk Lirası → TL olarak dönüştür).
8. Yönetim kurulu üyesi olarak atanan kişi/tüzel kişileri board_events listesine ekle. TEMSİLCİ ve YETKİLİ atamalarını board_events'e ekleme.
9. Görevden alınan üyeleri action=remove olarak ekle.
10. role alanına YK Başkanı ise "Yönetim Kurulu Başkanı", Başkan Yardımcısı ise "Yönetim Kurulu Başkan Yardımcısı", sadece üye ise "Yönetim Kurulu Üyesi" yaz. Belirtilmemişse null.
11. "İlk X Yıl için ... Temsile Yetkili olarak seçilmiştir" ifadesiyle yapılan atamalar temsil yetkisidir, YK üyeliği değildir — board_events'e ekleme.
12. Denetçi tablosunda "Yeni Denetçi" satırındaki firma adını denetci alanına yaz. "Görevi Sona Eren Denetçi" satırını dikkate alma.
13. YK üyesi tüzel kişiyse (şirketse), "Tüzel kişi adına; ... hareket edecektir" ifadesindeki gerçek kişinin adı soyadını representative alanına yaz.
'''


ARTICLE_HEADING_RE = re.compile(
    r"(?mi)^(?:#+\s+)?(?:madde\s+)?(?P<num>\d{1,2})[.\-:]\s*(?!\(\d\))(?P<title>.*)$"
)
DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})")
CAPITAL_RE = re.compile(r"SERMAYE(?:SI)?[^0-9]{0,80}([0-9][0-9\.\,\s]{1,24}\s*TL)")
MERSIS_RE = re.compile(r"MERSIS(?:\s+NO| NUMARASI)?\s*[:\-]?\s*([0-9]{8,16})")
SICIL_NO_RE = re.compile(r"TICARET SICIL(?:I)?(?:/DOSYA)?\s*(?:NO|NUMARASI)\s*[:\-]?\s*([0-9A-Z/\-]+)")
SICIL_MUDURLUGU_RE = re.compile(r"([A-Z. ]+ TICARET SICIL(?:I)? MUDURLUGU)")
ADDRESS_RE = re.compile(r"ADRES(?:I)?\s*[:\-]?\s*(.+)")
GAZETTE_RE = re.compile(r"SAYI\s*:?\s*(\d+)")

APPOINTMENT_KEYWORDS = ("ATANDI", "ATANMISTIR", "SECILDI", "SECILMISTIR", "GOREVLENDIRILDI", "GOREVLENDIRILMISTIR")
REMOVAL_KEYWORDS = ("GOREVDEN ALINDI", "ISTIFA", "AYRILDI", "UYELIGI SONA", "TIYALIGI SONA")

# Matches the "İlk N Yıl için ... Temsile Yetkili" pattern — representative, NOT board member
_ILK_YIL_TEMSILE_YETKILI_RE = re.compile(
    r"ILK \d+ YIL ICIN\b.{0,600}?TEMSILE YETKILI OLARAK SECILMISTIR",
    re.DOTALL,
)


@dataclass(frozen=True)
class AuthoritativeDoc:
    meta: DocumentMeta
    text: str
    source_pages: list[int]
    provider: str
    status: str
    gazette_number: str | None
    notes: list[str]


def load_authoritative_documents(cache: CacheLayout) -> tuple[list[AuthoritativeDoc], list[dict]]:
    documents = discover_documents(cache.paths)
    adjudicated_dir = cache.paths.target_only / "adjudicated"
    manual_review_items: list[dict] = []
    authoritative: list[AuthoritativeDoc] = []
    for document in documents:
        consensus = cache.load_consensus(document)
        override_path = adjudicated_dir / f"{document.doc_id}.md"
        if override_path.exists():
            text = override_path.read_text(encoding="utf-8").strip()
            authoritative.append(
                AuthoritativeDoc(
                    meta=document,
                    text=text,
                    source_pages=consensus.source_pages if consensus else list(range(1, document.page_count + 1)),
                    provider="adjudicated",
                    status="adjudicated",
                    gazette_number=_extract_gazette_number(cache, document, (consensus.providers if consensus else [])),
                    notes=["manual_override"],
                )
            )
            continue
        if consensus is None:
            manual_review_items.append(
                {
                    "doc_id": document.doc_id,
                    "status": "missing_consensus",
                    "reason": "extract-target command has not produced consensus output.",
                    "pdf_path": document.relative_pdf_path,
                }
            )
            continue
        if consensus.status == "manual_review":
            # Still include in authoritative docs but also flag for review
            manual_review_items.append(
                {
                    "doc_id": document.doc_id,
                    "status": consensus.status,
                    "reason": ", ".join(consensus.notes),
                    "pdf_path": document.relative_pdf_path,
                }
            )
        if not consensus.normalized_text.strip():
            manual_review_items.append(
                {
                    "doc_id": document.doc_id,
                    "status": "empty_consensus_text",
                    "reason": ", ".join(consensus.notes),
                    "pdf_path": document.relative_pdf_path,
                }
            )
        authoritative.append(
            AuthoritativeDoc(
                meta=document,
                text=consensus.normalized_text,
                source_pages=consensus.source_pages,
                provider="+".join(consensus.providers) or "mistral",
                status=consensus.status,
                gazette_number=_extract_gazette_number(cache, document, consensus.providers),
                notes=consensus.notes,
            )
        )
        if consensus.status == "single_provider":
            manual_review_items.append(
                {
                    "doc_id": document.doc_id,
                    "status": consensus.status,
                    "reason": "Only one provider was available; verify before relying on final outputs.",
                    "pdf_path": document.relative_pdf_path,
                }
            )
    return authoritative, manual_review_items


def consolidate(cache: CacheLayout, *, replicate_token: str | None = None, gemini_api_key: str | None = None) -> ConsolidationResult:
    authoritative_docs, manual_review_items = load_authoritative_documents(cache)
    company_info = CurrentCompanyInfo()
    article_versions: dict[str, ArticleVersion] = {}
    board_ledger: dict[str, BoardMemberRecord] = {}

    use_llm = bool(replicate_token)
    llm_usage_log: list[dict] = []

    for document in authoritative_docs:
        source = SourceRef(
            publication_date=document.meta.publication_date.isoformat(),
            gazette_number=document.gazette_number,
            pdf_path=document.meta.relative_pdf_path,
            pages=document.source_pages,
            provider=document.provider,
        )

        if use_llm:
            # --- LLM path: prefer cleaned document text, fall back to raw OCR ---
            clean_text = document.text.strip()
            raw_text = _get_raw_ocr_text(cache, document.meta, document.provider.split(":")) if not clean_text else None
            llm_text = clean_text or raw_text or ""
            # Gazette header fields (MERSIS, sicil) — reliable regex from raw OCR
            header_fields = _extract_gazette_header_fields(cache, document.meta, document.provider.split("+"))
            _apply_field(company_info, "mersis_numarasi", header_fields.get("mersis_numarasi"), source, only_if_empty=True)
            _apply_field(company_info, "ticaret_sicil_mudurlugu", header_fields.get("ticaret_sicil_mudurlugu"), source, only_if_empty=True)
            _apply_field(company_info, "ticaret_sicil_numarasi", header_fields.get("ticaret_sicil_numarasi"), source, only_if_empty=True)
            print(f"[consolidate-llm] {document.meta.doc_id}: parsing with Gemini Flash...")
            parsed, metrics = _llm_parse_document(llm_text, replicate_token, gemini_api_key)
            if parsed:
                _apply_llm_parsed(parsed, company_info, article_versions, board_ledger,
                                  manual_review_items, source,
                                  doc_category=document.meta.category)
                llm_usage_log.append({"doc_id": document.meta.doc_id, **metrics})
                print(f"[consolidate-llm] {document.meta.doc_id}: OK "
                      f"(articles={len(parsed.get('articles', []))}, "
                      f"board={len(parsed.get('board_events', []))}, "
                      f"cost=${metrics.get('cost_usd', 0):.6f})")
            else:
                # LLM failed — fall back to regex for this document
                if document.text.strip():
                    _regex_extract_document(cache, document, company_info, article_versions,
                                            board_ledger, manual_review_items, source)
        else:
            # --- Regex path (original) ---
            if not document.text.strip():
                continue
            _regex_extract_document(cache, document, company_info, article_versions,
                                    board_ledger, manual_review_items, source)

    if "6" in article_versions and not use_llm:
        capital = extract_capital(article_versions["6"].body)
        _apply_field(company_info, "mevcut_sermaye", capital, article_versions["6"].source)

    # Fill in transparent placeholders for any article numbers missing from the
    # consecutive range 1..max.  This can happen when an OCR page is physically
    # absent from the supplied PDFs.  A placeholder is always preferable to
    # silently omitting the article, as it preserves zero-hallucination guarantees.
    if article_versions:
        max_no = max(int(k) for k in article_versions)
        for missing_no in range(1, max_no + 1):
            key = str(missing_no)
            if key not in article_versions:
                # Use the earliest available founding source as attribution.
                fallback_source = min(
                    article_versions.values(),
                    key=lambda v: v.source.publication_date,
                ).source
                gazette_ref = (
                    f" (TTSG Sayı: {fallback_source.gazette_number})"
                    if fallback_source.gazette_number
                    else ""
                )
                article_versions[key] = ArticleVersion(
                    article_no=key,
                    title="",
                    body=(
                        f"[Madde {missing_no}'nin içeriği temin edilememiştir."
                        f" Kaynak belgede{gazette_ref} bu madde,"
                        f" sağlanan PDF dosyasına dahil edilmemiş sayfada yer almaktadır.]"
                    ),
                    source=fallback_source,
                )

    _prune_temsile_yetkili_from_ledger(board_ledger, authoritative_docs)

    result = ConsolidationResult(
        company_info=company_info,
        board_members=sorted(board_ledger.values(), key=lambda item: slugify(item.name_or_title)),
        articles=sorted(article_versions.values(), key=lambda item: int(item.article_no)),
        manual_review_items=manual_review_items,
    )
    _write_consolidation_outputs(cache, result)

    # Persist LLM usage metrics for downstream benchmarking
    if llm_usage_log:
        write_json(cache.paths.final / "llm_usage.json", llm_usage_log)
        total_cost = sum(entry.get("cost_usd", 0) for entry in llm_usage_log)
        total_in = sum(entry.get("input_tokens", 0) for entry in llm_usage_log)
        total_out = sum(entry.get("output_tokens", 0) for entry in llm_usage_log)
        print(f"[consolidate-llm] Total LLM cost: ${total_cost:.6f} "
              f"({total_in} input + {total_out} output tokens, {len(llm_usage_log)} calls)")

    return result


# ---------------------------------------------------------------------------
# LLM-based document parsing (Gemini Flash via Replicate)
# ---------------------------------------------------------------------------

def _get_raw_ocr_text(cache: CacheLayout, document: DocumentMeta, providers: list[str]) -> str | None:
    """Read raw OCR pages for a document and merge into a single string."""
    candidates = providers or ["mistral", "gemini"]
    for provider in candidates:
        ocr_doc = cache.load_ocr(provider, document)
        if ocr_doc and ocr_doc.pages:
            return "\n\n".join(page.raw_markdown for page in ocr_doc.pages)
    return None


def _llm_parse_document(text: str, replicate_token: str, gemini_api_key: str | None = None) -> tuple[dict | None, dict]:
    """Call Gemini Flash via Replicate to extract structured data from OCR text.

    Returns (parsed_dict_or_None, metrics_dict).
    """
    empty_metrics: dict = {"input_tokens": 0, "output_tokens": 0, "predict_time_s": 0.0, "cost_usd": 0.0}
    if not text.strip():
        return None, empty_metrics
    full_prompt = f"{_LLM_EXTRACT_PROMPT}\n---\n\n{text}"
    input_tokens = count_tokens_gemini(full_prompt, gemini_api_key) if gemini_api_key else 0
    payload = {"input": {"prompt": full_prompt}}
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(_LLM_MAX_RETRIES):
        request = urllib.request.Request(
            REPLICATE_PREDICTIONS_ENDPOINT,
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
            if exc.code == 429 and attempt < _LLM_MAX_RETRIES - 1:
                print(f"[llm-parse] 429 rate limit, retrying in 30s (attempt {attempt + 1})")
                time.sleep(30)
                continue
            print(f"[llm-parse] Replicate error {exc.code}: {body[:200]}")
            return None, empty_metrics
        except urllib.error.URLError as exc:
            print(f"[llm-parse] Network error: {exc}")
            return None, empty_metrics
        # Poll if needed
        if result.get("status") not in ("succeeded",):
            result = _llm_poll(result, replicate_token)
            if result is None:
                return None, empty_metrics
        # Extract metrics
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
        return _parse_llm_json(output), metrics
    return None, empty_metrics


def _llm_poll(prediction: dict, token: str) -> dict | None:
    """Poll a Replicate prediction until done."""
    poll_url = prediction.get("urls", {}).get("get")
    if not poll_url:
        print("[llm-parse] No poll URL, skipping")
        return None
    deadline = time.perf_counter() + _LLM_POLL_TIMEOUT
    while time.perf_counter() < deadline:
        time.sleep(_LLM_POLL_INTERVAL)
        request = urllib.request.Request(
            poll_url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[llm-parse] Poll error: {exc}")
            return None
        status = result.get("status")
        if status == "succeeded":
            return result
        if status in ("failed", "canceled"):
            print(f"[llm-parse] Prediction {status}: {result.get('error', '')}")
            return None
    print(f"[llm-parse] Timed out after {_LLM_POLL_TIMEOUT}s")
    return None





def _parse_llm_json(raw_output: str) -> dict | None:
    """Extract JSON from LLM output (may have markdown fences)."""
    text = raw_output.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json) and last (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the output
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    print(f"[llm-parse] Could not parse JSON from LLM output: {text[:200]}")
    return None


def _apply_llm_parsed(
    parsed: dict,
    company_info: CurrentCompanyInfo,
    article_versions: dict[str, ArticleVersion],
    board_ledger: dict[str, BoardMemberRecord],
    manual_review_items: list[dict],
    source: SourceRef,
    *,
    doc_category: str = "",
) -> None:
    """Apply LLM-parsed structured data to the consolidation state."""
    # Documents that never carry a valid capital change
    _NO_CAPITAL_CATEGORIES = {"denetci-degisikligi", "yonetim-kurulu"}
    ci = parsed.get("company_info") or {}
    _apply_field(company_info, "ticaret_unvani", ci.get("ticaret_unvani"), source)
    _apply_field(company_info, "adres", ci.get("adres"), source)
    if doc_category not in _NO_CAPITAL_CATEGORIES:
        _apply_field(company_info, "mevcut_sermaye", ci.get("sermaye"), source)
    _apply_field(company_info, "denetci", ci.get("denetci"), source)
    _apply_field(company_info, "kurulus_tarihi", ci.get("kurulus_tarihi"), source, only_if_empty=True)
    # Detect şirket türü from unvanı
    unvan = ci.get("ticaret_unvani") or ""
    if "ANONİM" in unvan.upper() or "ANONIM" in normalize_text(unvan):
        _apply_field(company_info, "sirket_turu", "Anonim Şirket", source)
    elif "LİMİTED" in unvan.upper() or "LIMITED" in normalize_text(unvan):
        _apply_field(company_info, "sirket_turu", "Limited Şirket", source)

    # Articles — only for documents where esas sözleşme changes are expected
    _ARTICLE_CATEGORIES = {"kurulus", "esas-sozlesme-degisikligi"}
    if doc_category in _ARTICLE_CATEGORIES:
        for art in parsed.get("articles") or []:
            no = str(art.get("article_no", "")).strip()
            if not no or not no.isdigit():
                continue
            article_versions[no] = ArticleVersion(
                article_no=no,
                title=art.get("title") or "",
                body=art.get("body") or "",
                source=source,
            )

    # Board events
    for ev in parsed.get("board_events") or []:
        action = (ev.get("action") or "").strip().lower()
        name = (ev.get("name") or "").strip()
        if not name:
            continue
        name_key = normalize_text(name)
        if action == "appoint":
            board_ledger[name_key] = BoardMemberRecord(
                name_or_title=name,
                role=ev.get("role") or None,
                representative=ev.get("representative") or None,
                role_end_date=ev.get("role_end_date"),
                appointed_ttsg_date=source.publication_date,
                appointed_ttsg_number=source.gazette_number,
                pdf_link=source.pdf_path,
                status="active",
                source=source,
            )
        elif action == "remove":
            board_ledger.pop(name_key, None)


def _regex_extract_document(
    cache: CacheLayout,
    document: AuthoritativeDoc,
    company_info: CurrentCompanyInfo,
    article_versions: dict[str, ArticleVersion],
    board_ledger: dict[str, BoardMemberRecord],
    manual_review_items: list[dict],
    source: SourceRef,
) -> None:
    """Original regex-based extraction path (fallback)."""
    field_values = extract_company_fields(document.text)
    _apply_field(company_info, "ticaret_unvani", field_values.get("ticaret_unvani"), source)
    _apply_field(company_info, "sirket_turu", field_values.get("sirket_turu"), source)
    _apply_field(company_info, "adres", field_values.get("adres"), source)

    header_fields = _extract_gazette_header_fields(cache, document.meta, document.provider.split("+"))
    _apply_field(company_info, "mersis_numarasi", header_fields.get("mersis_numarasi"), source, only_if_empty=True)
    _apply_field(company_info, "ticaret_sicil_mudurlugu", header_fields.get("ticaret_sicil_mudurlugu"), source, only_if_empty=True)
    _apply_field(company_info, "ticaret_sicil_numarasi", header_fields.get("ticaret_sicil_numarasi"), source, only_if_empty=True)
    if document.meta.category == "kurulus":
        _apply_field(company_info, "kurulus_tarihi", extract_founding_date(document.text), source, only_if_empty=True)

    if document.meta.category in {"kurulus", "esas-sozlesme-degisikligi"}:
        for article in parse_articles(document.text, source):
            article_versions[article.article_no] = article

    capital = extract_capital(document.text)
    _apply_field(company_info, "mevcut_sermaye", capital, source)

    denetci = extract_auditor(document.text)
    _apply_field(company_info, "denetci", denetci, source)

    board_events, board_review_items = extract_board_events(document.text, source)
    manual_review_items.extend(board_review_items)
    for event in board_events:
        if event["action"] == "appoint":
            board_ledger[event["member"]] = BoardMemberRecord(
                name_or_title=event["member"],
                role=None,
                representative=None,
                role_end_date=event.get("role_end_date"),
                appointed_ttsg_date=source.publication_date,
                appointed_ttsg_number=source.gazette_number,
                pdf_link=source.pdf_path,
                status="active",
                source=source,
            )
        elif event["action"] == "remove":
            board_ledger.pop(event["member"], None)


# ---------------------------------------------------------------------------
# Regex-based extraction functions (used as fallback)
# ---------------------------------------------------------------------------

def extract_company_fields(text: str) -> dict[str, str | None]:
    normalized = normalize_text(text)
    heading = None
    for line in text.splitlines():
        if TARGET_COMPANY in normalize_text(line):
            clean = re.sub(r'^#+\s*', '', line.strip())
            clean = re.sub(r'(?i)^Ticaret\s+[UÜuü]nvan[ıiIİ]\s*:\s*', '', clean)
            clean = clean.replace('**', '').strip()
            heading = clean
            break
    company_type = None
    if heading:
        normalized_heading = normalize_text(heading)
        if "ANONIM SIRKETI" in normalized_heading:
            company_type = "Anonim Şirket"
        elif "LIMITED SIRKETI" in normalized_heading:
            company_type = "Limited Şirket"

    address = None
    for line in text.splitlines():
        stripped = line.strip()
        if ADDRESS_RE.search(normalize_text(stripped)):
            clean = re.sub(r'^#+\s*', '', stripped)
            clean = clean.replace('**', '')
            clean = re.sub(r'(?i)^adres\s*:?\s*', '', clean)
            address = clean.strip()
            break

    return {
        "ticaret_unvani": heading,
        "sirket_turu": company_type,
        "adres": address,
    }


def parse_articles(text: str, source: SourceRef) -> list[ArticleVersion]:
    # Truncate at İç Yönerge SECTION HEADING to avoid parsing directive articles.
    # Only trigger on lines that are actual headings (start with #), not on body text
    # that merely mentions "iç yönerge" in passing (e.g. "bir iç yönergeeye göre").
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "IC YONERGE" in normalize_text(stripped):
            text = "".join(lines[:i])
            break
    matches = list(ARTICLE_HEADING_RE.finditer(text))
    articles: list[ArticleVersion] = []
    seen: set[str] = set()
    max_no = 0  # highest article number seen so far; used for forward-jump guard
    for index, match in enumerate(matches):
        article_no = int(match.group("num"))
        title = (match.group("title") or "").strip()
        # Skip only forward-jumping sub-items: if this number jumps > 5 beyond the max
        # article seen so far it is likely a numbered list item inside an article body.
        # Use max_no (not article_no) so that out-of-order kuruluş articles never
        # accidentally raise the threshold (e.g. 10 → 4 → 5 → 11 should keep max=10).
        if max_no > 0 and article_no > max_no and article_no - max_no > 5:
            continue
        # Whether the match used the explicit "Madde X-" prefix form.
        # In older Turkish gazette format articles have no separate title; the body
        # starts inline on the same line as "Madde 6-", so the regex captures the
        # entire body-line as `title`.  When the madde prefix is present we fold
        # that captured text back into the body instead of discarding the article.
        has_madde_prefix = bool(re.match(r"^(?:#+\s+)?madde\s", match.group(0), re.IGNORECASE))
        # Skip if title looks like a full sentence (real articles have short titles).
        # Exception: madde-prefix articles never have a separate title.
        if len(title) > 120 and not has_madde_prefix:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Fold inline body text (captured as title) into body; clear the title.
        if has_madde_prefix and len(title) > 120:
            body = title + ("\n" + body if body else "")
            title = ""
        key = str(article_no)
        # Deduplication: for Madde-prefix articles (esas-sozlesme format) keep the
        # FIRST occurrence — contamination from neighboring companies in the same
        # gazette page always appears after the target company's section.
        # For non-madde articles (kuruluş format, ## N. Title) the same article may
        # appear twice due to OCR multi-pass; prefer the longer body there.
        if key in seen:
            if not has_madde_prefix:
                existing = next(a for a in reversed(articles) if a.article_no == key)
                if len(body) > len(existing.body):
                    articles.remove(existing)
                    articles.append(ArticleVersion(article_no=key, title=title, body=body, source=source))
        else:
            articles.append(ArticleVersion(article_no=key, title=title, body=body, source=source))
            seen.add(key)
        if article_no > max_no:
            max_no = article_no
    return articles


def extract_capital(text: str) -> str | None:
    normalized = normalize_text(text)
    matches = CAPITAL_RE.findall(normalized)
    if not matches:
        return None
    return _normalize_money(matches[-1])


def extract_founding_date(text: str) -> str | None:
    normalized = normalize_text(text)
    match = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})\s+TARIHINDE\s+TESCIL", normalized)
    if match:
        return _normalize_date(match.group(1))
    return None


def extract_auditor(text: str) -> str | None:
    normalized = normalize_text(text)
    if "DENETCI" not in normalized:
        return None
    appointment = re.search(
        r"([A-Z0-9 .,&/-]+?)\s+DENETCI\s+OLARAK\s+(?:ATANDI|SECILDI|BELIRLENDI)",
        normalized,
    )
    if appointment:
        return appointment.group(1).strip(" .,")
    direct = re.search(r"DENETCI(?:SI)?\s*[:\-]\s*([A-Z0-9 .,&/-]+)", normalized)
    if direct:
        return direct.group(1).strip(" .,")
    # Try markdown table in DENETÇİLER section
    return _extract_auditor_from_table(text)


def _extract_auditor_from_table(text: str) -> str | None:
    """Extract auditor name from markdown table in DENETÇİLER section."""
    in_section = False
    for line in text.splitlines():
        norm = normalize_text(line)
        if "DENETCI" in norm and ("YENI" in norm or "DENETCILER" in norm):
            in_section = True
            continue
        if in_section and line.strip().startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.strip().split('|')]
            cells = [c for c in cells if c]
            if len(cells) >= 3:
                name = cells[2].strip()
                if name and not any(h in normalize_text(name) for h in ("ADI SOYADI", "FIRMA ADI")):
                    return name
    return None


def _prune_temsile_yetkili_from_ledger(
    ledger: dict[str, "BoardMemberRecord"],
    docs: list["AuthoritativeDoc"],
) -> None:
    """Remove members who were only designated 'İlk N Yıl için Temsile Yetkili' (representative,
    not an actual board appointment) from their source document."""
    doc_text_by_date: dict[str, str] = {
        d.meta.publication_date.isoformat(): normalize_text(d.text) for d in docs
    }
    to_remove = []
    for key, member in ledger.items():
        norm_text = doc_text_by_date.get(member.source.publication_date)
        if not norm_text:
            continue
        norm_name = normalize_text(member.name_or_title)
        for segment_match in _ILK_YIL_TEMSILE_YETKILI_RE.finditer(norm_text):
            if norm_name in segment_match.group(0):
                to_remove.append(key)
                break
    for key in to_remove:
        del ledger[key]


def extract_board_events(text: str, source: SourceRef) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    review_items: list[dict] = []
    if "YONETIM KURULU" not in normalize_text(text):
        return events, review_items
    clauses = _split_clauses(text)
    for clause in clauses:
        normalized = normalize_text(clause)
        has_appoint = any(keyword in normalized for keyword in APPOINTMENT_KEYWORDS)
        has_remove = any(keyword in normalized for keyword in REMOVAL_KEYWORDS)
        if not has_appoint and not has_remove:
            continue
        found_event = False
        if has_appoint:
            member = _extract_appointment_name(normalized)
            if member:
                role_end_date = _extract_role_end_date(normalized)
                events.append({"action": "appoint", "member": member, "role_end_date": role_end_date})
                found_event = True
        if has_remove:
            member = _extract_removal_name(normalized)
            if member:
                events.append({"action": "remove", "member": member})
                found_event = True
        if not found_event:
            review_items.append(
                {
                    "kind": "board_member_unresolved",
                    "publication_date": source.publication_date,
                    "pdf_path": source.pdf_path,
                    "clause": clause,
                }
            )
    return events, review_items


_INVALID_NAME_FRAGMENTS = {
    "TARIHINE KADAR", "UYEDEN OLUSAN", "YAYINLANMIS OLAN",
    "DE YAYINLANMIS", "UYELIGI SONA", "KARAR UYARINCA",
    "YONETIM KURULU", "MADDE", "SUREYLE", "OLARAK SECILMISTIR",
    "OLAN IC", "BIR YONETIM",
}


def _is_valid_member_name(name: str) -> bool:
    """Reject obviously invalid board member names (sentence fragments)."""
    if not name or len(name) < 4:
        return False
    words = name.split()
    if len(words) < 2:
        return False
    for frag in _INVALID_NAME_FRAGMENTS:
        if frag in name:
            return False
    return True


def _extract_appointment_name(normalized: str) -> str | None:
    """Extract member name from an appointment clause."""
    # Person name after IKAMET EDEN
    match = re.search(
        r"IKAMET EDEN[, ]+(?P<name>[A-Z][A-Z .]+?)\s*[;\s,]*"
        r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}\s+TARIHINE\s+KADAR\s+)?"
        r"YONETIM KURULU",
        normalized,
    )
    if match:
        name = match.group("name").strip(" ,.")
        if _is_valid_member_name(name):
            return name
    # Company name (ends with SIRKETI)
    match = re.search(
        r"(?P<name>[A-Z][A-Z0-9 .,&/'()-]*?(?:ANONIM SIRKETI|LIMITED SIRKETI|SIRKETI))"
        r"[;\s,]*(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}\s+TARIHINE\s+KADAR\s+)?"
        r"YONETIM KURULU",
        normalized,
    )
    if match:
        name = match.group("name").strip(" ,.")
        if _is_valid_member_name(name):
            return name
    # Direct person name (no identity prefix)
    match = re.search(
        r"(?P<name>[A-Z][A-Z ]+?)\s+"
        r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}\s+TARIHINE\s+KADAR\s+)?"
        r"YONETIM KURULU",
        normalized,
    )
    if match:
        name = match.group("name").strip(" ,.")
        if _is_valid_member_name(name):
            return name
    return None


def _extract_removal_name(normalized: str) -> str | None:
    """Extract member name from a removal clause."""
    # Company name + possessive + ONCEKI UYELIGI/TIYALIGI SONA
    match = re.search(
        r"(?P<name>[A-Z][A-Z0-9 .,&/'()-]*?(?:ANONIM SIRKETI|LIMITED SIRKETI|SIRKETI))"
        r"['\u2019]?(?:IN|NIN|NUN|NI)\s+ONCEKI\s+(?:UYELIGI|TIYALIGI)\s+SONA",
        normalized,
    )
    if match:
        return match.group("name").strip(" ,.")
    # Person name after IKAMET EDEN + possessive
    match = re.search(
        r"IKAMET EDEN[, ]+(?P<name>[A-Z][A-Z .]+?)\s*"
        r"['\u2019]?(?:IN|NIN|NUN|NI)\s+ONCEKI\s+(?:UYELIGI|TIYALIGI)\s+SONA",
        normalized,
    )
    if match:
        return match.group("name").strip(" ,.")
    # Generic removal keywords
    match = re.search(
        r"(?P<name>[A-Z0-9 .,&/'()-]+?)\s+(?:ISTIFA|GOREVDEN ALINDI|AYRILDI)",
        normalized,
    )
    if match:
        return match.group("name").strip(" ,.")
    return None


def _extract_role_end_date(normalized: str) -> str | None:
    match = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})\s+TARIHINE\s+KADAR", normalized)
    if match:
        return _normalize_date(match.group(1))
    return None


def _split_clauses(text: str) -> Iterable[str]:
    for piece in re.split(r"[\n;]+", text):
        candidate = piece.strip()
        if candidate:
            yield candidate


def _extract_gazette_number(cache: CacheLayout, document: DocumentMeta, providers: list[str]) -> str | None:
    candidates = providers or ["mistral", "gemini"]
    for provider in candidates:
        ocr_document = cache.load_ocr(provider, document)
        if not ocr_document:
            continue
        combined = "\n".join(page.raw_markdown for page in ocr_document.pages)
        normalized = normalize_text(combined)
        match = GAZETTE_RE.search(normalized)
        if match:
            return match.group(1)
    return None


def _extract_gazette_header_fields(cache: CacheLayout, document: DocumentMeta, providers: list[str]) -> dict[str, str | None]:
    """Extract MERSIS, sicil müdürlüğü, and sicil numarası from the gazette header above the target section."""
    candidates = providers or ["mistral", "gemini"]
    for provider in candidates:
        ocr_document = cache.load_ocr(provider, document)
        if not ocr_document:
            continue
        combined = "\n".join(page.raw_markdown for page in ocr_document.pages)
        normalized = normalize_text(combined)
        target_pos = normalized.find(TARGET_COMPANY)
        if target_pos == -1:
            continue
        header_text = normalized[:target_pos]
        mersis = _first_group(MERSIS_RE, header_text)
        sicil_no = _first_group(SICIL_NO_RE, header_text)
        mudurluk_match = SICIL_MUDURLUGU_RE.search(header_text)
        mudurluk = mudurluk_match.group(1).title().strip() if mudurluk_match else None
        if mersis or sicil_no or mudurluk:
            return {"mersis_numarasi": mersis, "ticaret_sicil_numarasi": sicil_no, "ticaret_sicil_mudurlugu": mudurluk}
    return {}


def _apply_field(
    company_info: CurrentCompanyInfo,
    field_name: str,
    value: str | None,
    source: SourceRef,
    *,
    only_if_empty: bool = False,
) -> None:
    if not value:
        return
    if only_if_empty and getattr(company_info, field_name):
        return
    setattr(company_info, field_name, value)
    company_info.source_map[field_name] = source


def _first_group(pattern: re.Pattern[str], normalized: str) -> str | None:
    match = pattern.search(normalized)
    if match:
        return match.group(1).strip()
    return None


def _normalize_money(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace(" ,", ",").replace(" .", ".")).strip()


def _normalize_date(value: str) -> str:
    return value.replace("/", ".").replace("-", ".")


def _write_consolidation_outputs(cache: CacheLayout, result: ConsolidationResult) -> None:
    cache.paths.final.mkdir(parents=True, exist_ok=True)
    write_json(cache.paths.final / "company_info.json", result.company_info.to_dict())
    write_json(cache.paths.final / "board_members.json", [item.to_dict() for item in result.board_members])
    write_json(cache.paths.final / "articles.json", [item.to_dict() for item in result.articles])
    write_json(cache.paths.final / "consolidation.json", result.to_dict())

    _write_company_info_csv(cache.paths.final / "company_info.csv", result.company_info)
    _write_board_csv(cache.paths.final / "board_members.csv", result.board_members)
    (cache.paths.final / "company_info.md").write_text(_company_info_markdown(result.company_info), encoding="utf-8")
    (cache.paths.final / "board_members.md").write_text(_board_markdown(result.board_members), encoding="utf-8")
    (cache.paths.final / "articles.md").write_text(_articles_markdown(result.articles), encoding="utf-8")


def _write_company_info_csv(path: Path, info: CurrentCompanyInfo) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alan", "deger", "kaynak_pdf", "kaynak_tarih", "kaynak_sayi"])
        for field_name, label in (
            ("ticaret_unvani", "Ticaret Unvanı"),
            ("sirket_turu", "Şirket Türü"),
            ("mersis_numarasi", "MERSİS Numarası"),
            ("ticaret_sicil_mudurlugu", "Ticaret Sicil Müdürlüğü"),
            ("ticaret_sicil_numarasi", "Ticaret Sicil Numarası"),
            ("adres", "Adres"),
            ("mevcut_sermaye", "Mevcut Sermaye"),
            ("kurulus_tarihi", "Kuruluş Tarihi"),
            ("denetci", "Denetçi"),
        ):
            source = info.source_map.get(field_name)
            writer.writerow(
                [
                    label,
                    getattr(info, field_name),
                    source.pdf_path if source else "",
                    source.publication_date if source else "",
                    source.gazette_number if source else "",
                ]
            )


def _write_board_csv(path: Path, members: list[BoardMemberRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uye", "unvan", "temsilci", "gorev_bitis_tarihi", "atandigi_tarih", "atandigi_sayi", "pdf_link"])
        for item in members:
            writer.writerow(
                [
                    item.name_or_title,
                    item.role or "",
                    item.representative or "",
                    item.role_end_date or "",
                    item.appointed_ttsg_date,
                    item.appointed_ttsg_number or "",
                    item.pdf_link,
                ]
            )


def _company_info_markdown(info: CurrentCompanyInfo) -> str:
    headers = ["Alan", "Değer", "Kaynak Tarih / Sayı", "Kaynak PDF"]
    rows = []
    for field_name, label in (
        ("ticaret_unvani", "Ticaret Unvanı"),
        ("sirket_turu", "Şirket Türü"),
        ("mersis_numarasi", "MERSİS Numarası"),
        ("ticaret_sicil_mudurlugu", "Ticaret Sicil Müdürlüğü"),
        ("ticaret_sicil_numarasi", "Ticaret Sicil Numarası"),
        ("adres", "Adres"),
        ("mevcut_sermaye", "Mevcut Sermaye"),
        ("kurulus_tarihi", "Kuruluş Tarihi"),
        ("denetci", "Denetçi"),
    ):
        source = info.source_map.get(field_name)
        source_label = ""
        source_pdf = ""
        if source:
            source_label = source.publication_date + (f" / {source.gazette_number}" if source.gazette_number else "")
            source_pdf = f"[{source.pdf_path}]({source.pdf_path})"
        rows.append([label, getattr(info, field_name) or "", source_label, source_pdf])
    return _markdown_table(headers, rows)


def _board_markdown(members: list[BoardMemberRecord]) -> str:
    headers = ["Yönetim Kurulu Üyesi", "Unvan", "Görev Bitiş Tarihi", "Atandığı TTSG Tarih / Sayı", "Kaynak PDF"]
    rows = []
    for member in members:
        display = member.name_or_title
        if member.representative:
            display += f" (adına hareket edecek gerçek kişi: {member.representative})"
        rows.append(
            [
                display,
                member.role or "",
                member.role_end_date or "",
                member.appointed_ttsg_date
                + (f" / {member.appointed_ttsg_number}" if member.appointed_ttsg_number else ""),
                f"[{member.pdf_link}]({member.pdf_link})",
            ]
        )
    return _markdown_table(headers, rows)


def _articles_markdown(articles: list[ArticleVersion]) -> str:
    parts = ["# Konsolide Esas Sözleşme", ""]
    for article in articles:
        source = article.source
        parts.append(f"## Madde {article.article_no} - {article.title or ''}".strip())
        parts.append(f"Kaynak: {source.publication_date}" + (f" / {source.gazette_number}" if source.gazette_number else ""))
        parts.append("")
        parts.append(article.body.strip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _manual_review_markdown(items: list[dict]) -> str:
    parts = ["# Manual Review Items", ""]
    for item in items:
        doc_id = item.get("doc_id", item.get("kind", "unknown"))
        status = item.get("status", "")
        reason = item.get("reason", item.get("clause", ""))
        pdf_path = item.get("pdf_path", "")
        line = f"- **{doc_id}** ({status}): {reason}"
        if pdf_path:
            line += f" — [{pdf_path}]({pdf_path})"
        parts.append(line)
    return "\n".join(parts).strip() + "\n"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"
