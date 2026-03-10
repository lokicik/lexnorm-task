from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import Paths
from .models import DocumentMeta
from .utils import slugify


FILENAME_RE = re.compile(r"^(?P<index>\d+)\)\s+(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<title>.+)\.pdf$", re.IGNORECASE)


def _approx_page_count(pdf_path: Path) -> int:
    payload = pdf_path.read_bytes()
    matches = [int(value) for value in re.findall(rb"/Count\s+(\d+)", payload)]
    if matches:
        return max(matches)
    page_markers = len(re.findall(rb"/Type/Page\b", payload))
    return page_markers or 1


def _category_for(title: str) -> str:
    lowered = title.lower()
    lowered = lowered.replace("-", " ")
    if "kurulus" in lowered:
        return "kurulus"
    if "esas sozlesme" in lowered:
        return "esas-sozlesme-degisikligi"
    if "denet" in lowered:
        return "denetci-degisikligi"
    if "yonetim kurulu" in lowered:
        return "yonetim-kurulu"
    return "diger"


def discover_documents(paths: Paths) -> list[DocumentMeta]:
    documents: list[DocumentMeta] = []
    for pdf_path in sorted(paths.data.glob("*.pdf")):
        match = FILENAME_RE.match(pdf_path.name)
        if not match:
            continue
        publication_date = datetime.strptime(match.group("date"), "%d-%m-%Y").date()
        title = match.group("title")
        category = _category_for(slugify(title))
        index = int(match.group("index"))
        doc_id = f"{index:02d}-{publication_date.isoformat()}-{slugify(title)}"
        documents.append(
            DocumentMeta(
                doc_id=doc_id,
                index=index,
                publication_date=publication_date,
                title=title,
                category=category,
                pdf_path=pdf_path,
                relative_pdf_path=str(pdf_path.relative_to(paths.root)),
                page_count=_approx_page_count(pdf_path),
            )
        )
    return sorted(documents, key=lambda item: (item.publication_date, item.index))
