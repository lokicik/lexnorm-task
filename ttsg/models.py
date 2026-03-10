from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentMeta:
    doc_id: str
    index: int
    publication_date: date
    title: str
    category: str
    pdf_path: Path
    relative_pdf_path: str
    page_count: int


@dataclass
class OCRPage:
    provider: str
    doc_id: str
    page: int
    source_image: str
    raw_markdown: str
    runtime_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    sha256: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRDocument:
    provider: str
    doc_id: str
    pages: list[OCRPage]
    created_at: str
    input_sha256: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "doc_id": self.doc_id,
            "created_at": self.created_at,
            "input_sha256": self.input_sha256,
            "notes": self.notes,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass
class SegmentMatch:
    start_index: int
    end_index: int
    text: str
    status: str
    reason: str
    source_pages: list[int]
    provider: str
    doc_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusDocument:
    doc_id: str
    normalized_text: str
    status: str
    providers: list[str]
    source_pages: list[int]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceRef:
    publication_date: str
    gazette_number: str | None
    pdf_path: str
    pages: list[int]
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoardMemberRecord:
    name_or_title: str
    role: str | None
    representative: str | None
    role_end_date: str | None
    appointed_ttsg_date: str
    appointed_ttsg_number: str | None
    pdf_link: str
    status: str
    source: SourceRef

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.to_dict()
        return data


@dataclass
class CurrentCompanyInfo:
    ticaret_unvani: str | None = None
    sirket_turu: str | None = None
    mersis_numarasi: str | None = None
    ticaret_sicil_mudurlugu: str | None = None
    ticaret_sicil_numarasi: str | None = None
    adres: str | None = None
    mevcut_sermaye: str | None = None
    kurulus_tarihi: str | None = None
    denetci: str | None = None
    source_map: dict[str, SourceRef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticaret_unvani": self.ticaret_unvani,
            "sirket_turu": self.sirket_turu,
            "mersis_numarasi": self.mersis_numarasi,
            "ticaret_sicil_mudurlugu": self.ticaret_sicil_mudurlugu,
            "ticaret_sicil_numarasi": self.ticaret_sicil_numarasi,
            "adres": self.adres,
            "mevcut_sermaye": self.mevcut_sermaye,
            "kurulus_tarihi": self.kurulus_tarihi,
            "denetci": self.denetci,
            "source_map": {key: value.to_dict() for key, value in self.source_map.items()},
        }


@dataclass
class ArticleVersion:
    article_no: str
    title: str | None
    body: str
    source: SourceRef

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.to_dict()
        return data


@dataclass
class ConsolidationResult:
    company_info: CurrentCompanyInfo
    board_members: list[BoardMemberRecord]
    articles: list[ArticleVersion]
    manual_review_items: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_info": self.company_info.to_dict(),
            "board_members": [item.to_dict() for item in self.board_members],
            "articles": [item.to_dict() for item in self.articles],
            "manual_review_items": self.manual_review_items,
        }
