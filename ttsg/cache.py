from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Paths
from .models import ConsolidationResult, ConsensusDocument, DocumentMeta, OCRDocument, SegmentMatch
from .utils import read_json, sha256_file, write_json


@dataclass(frozen=True)
class CacheLayout:
    paths: Paths

    def raster_dir(self, document: DocumentMeta, dpi: int = 300) -> Path:
        pdf_sha = sha256_file(document.pdf_path)[:16]
        return self.paths.cache / "rasterized" / f"{document.doc_id}-{pdf_sha}-{dpi}dpi"

    def preprocessed_dir(self, document: DocumentMeta, dpi: int = 300) -> Path:
        pdf_sha = sha256_file(document.pdf_path)[:16]
        return self.paths.cache / "preprocessed" / f"{document.doc_id}-{pdf_sha}-{dpi}dpi"

    def tiles_dir(self, document: DocumentMeta, dpi: int = 400, num_tiles: int = 5, overlap_pct: float = 0.03) -> Path:
        pdf_sha = sha256_file(document.pdf_path)[:16]
        olap = int(overlap_pct * 100)
        return self.paths.cache / "tiles" / f"{document.doc_id}-{pdf_sha}-{dpi}dpi-t{num_tiles}-o{olap}"

    def ocr_path(self, provider: str, document: DocumentMeta) -> Path:
        return self.paths.cache / "ocr" / provider / f"{document.doc_id}.json"

    def extraction_path(self, provider: str, document: DocumentMeta) -> Path:
        return self.paths.cache / "extracted" / provider / f"{document.doc_id}.json"

    def consensus_path(self, document: DocumentMeta) -> Path:
        return self.paths.cache / "consensus" / f"{document.doc_id}.json"

    def consolidation_path(self) -> Path:
        return self.paths.final / "consolidation.json"

    def benchmark_path(self) -> Path:
        return self.paths.benchmark / "benchmark.json"

    def load_ocr(self, provider: str, document: DocumentMeta) -> OCRDocument | None:
        path = self.ocr_path(provider, document)
        if not path.exists():
            return None
        payload = read_json(path)
        pages = [_ocr_page_from_dict(item) for item in payload.get("pages", [])]
        return OCRDocument(
            provider=payload["provider"],
            doc_id=payload["doc_id"],
            created_at=payload["created_at"],
            input_sha256=payload["input_sha256"],
            notes=payload.get("notes", []),
            pages=pages,
        )

    def save_ocr(self, provider: str, document: DocumentMeta, ocr_document: OCRDocument) -> Path:
        path = self.ocr_path(provider, document)
        write_json(path, ocr_document.to_dict())
        return path

    def load_extraction(self, provider: str, document: DocumentMeta) -> SegmentMatch | None:
        path = self.extraction_path(provider, document)
        if not path.exists():
            return None
        payload = read_json(path)
        return SegmentMatch(**payload)

    def save_extraction(self, provider: str, document: DocumentMeta, segment: SegmentMatch) -> Path:
        path = self.extraction_path(provider, document)
        write_json(path, segment.to_dict())
        return path

    def load_consensus(self, document: DocumentMeta) -> ConsensusDocument | None:
        path = self.consensus_path(document)
        if not path.exists():
            return None
        return ConsensusDocument(**read_json(path))

    def save_consensus(self, document: DocumentMeta, consensus: ConsensusDocument) -> Path:
        path = self.consensus_path(document)
        write_json(path, consensus.to_dict())
        return path

    def save_consolidation(self, result: ConsolidationResult) -> Path:
        path = self.consolidation_path()
        write_json(path, result.to_dict())
        return path

    def load_consolidation(self) -> ConsolidationResult | None:
        path = self.consolidation_path()
        if not path.exists():
            return None
        payload = read_json(path)
        from .models import ArticleVersion, BoardMemberRecord, CurrentCompanyInfo, SourceRef

        company_info = CurrentCompanyInfo(
            ticaret_unvani=payload["company_info"].get("ticaret_unvani"),
            sirket_turu=payload["company_info"].get("sirket_turu"),
            mersis_numarasi=payload["company_info"].get("mersis_numarasi"),
            ticaret_sicil_mudurlugu=payload["company_info"].get("ticaret_sicil_mudurlugu"),
            ticaret_sicil_numarasi=payload["company_info"].get("ticaret_sicil_numarasi"),
            adres=payload["company_info"].get("adres"),
            mevcut_sermaye=payload["company_info"].get("mevcut_sermaye"),
            kurulus_tarihi=payload["company_info"].get("kurulus_tarihi"),
            denetci=payload["company_info"].get("denetci"),
            source_map={
                key: SourceRef(**value) for key, value in payload["company_info"].get("source_map", {}).items()
            },
        )
        board_members = [
            BoardMemberRecord(
                name_or_title=item["name_or_title"],
                role=item.get("role"),
                representative=item.get("representative"),
                role_end_date=item.get("role_end_date"),
                appointed_ttsg_date=item["appointed_ttsg_date"],
                appointed_ttsg_number=item.get("appointed_ttsg_number"),
                pdf_link=item["pdf_link"],
                status=item["status"],
                source=SourceRef(**item["source"]),
            )
            for item in payload.get("board_members", [])
        ]
        articles = [
            ArticleVersion(
                article_no=item["article_no"],
                title=item.get("title"),
                body=item["body"],
                source=SourceRef(**item["source"]),
            )
            for item in payload.get("articles", [])
        ]
        return ConsolidationResult(
            company_info=company_info,
            board_members=board_members,
            articles=articles,
            manual_review_items=payload.get("manual_review_items", []),
        )


def _ocr_page_from_dict(payload: dict) -> "OCRPage":
    from .models import OCRPage

    return OCRPage(
        provider=payload["provider"],
        doc_id=payload["doc_id"],
        page=payload["page"],
        source_image=payload["source_image"],
        raw_markdown=payload["raw_markdown"],
        runtime_ms=payload["runtime_ms"],
        usage=payload.get("usage", {}),
        cost_usd=payload.get("cost_usd"),
        sha256=payload.get("sha256", ""),
        notes=payload.get("notes", []),
    )
