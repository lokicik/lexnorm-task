from __future__ import annotations

import html
import os
import zipfile
from pathlib import Path
from urllib.parse import quote

from .cache import CacheLayout
from .models import ConsolidationResult
from .utils import read_json


def export_docx(cache: CacheLayout, result: ConsolidationResult | None = None) -> Path:
    consolidation = result or cache.load_consolidation()
    if consolidation is None:
        raise RuntimeError("No consolidation output found. Run `python -m ttsg consolidate` first.")

    link_registry: dict[str, str] = {}
    for member in consolidation.board_members:
        if member.pdf_link and member.pdf_link not in link_registry:
            link_registry[member.pdf_link] = f"rId{len(link_registry) + 2}"

    output_path = cache.paths.deliverables / "ttsg_report.docx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr(
            "word/document.xml",
            _document_xml(consolidation, link_registry, cache.paths.target_only, cache.paths.final, cache.paths.cache),
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            _document_rels_xml(link_registry, cache.paths.root, cache.paths.deliverables),
        )
        archive.writestr("word/styles.xml", _styles_xml())
    return output_path


def _document_xml(
    result: ConsolidationResult,
    link_registry: dict[str, str],
    target_only_dir: Path,
    final_dir: Path,
    cache_dir: Path | None = None,
) -> str:
    body_parts = []
    body_parts.append(_paragraph("Parla Enerji Yatırımları Anonim Şirketi TTSG Raporu", bold=True))
    body_parts.append(_paragraph(""))
    body_parts.append(_paragraph("Güncel Şirket Bilgileri", bold=True))
    company_rows = [
        ("Ticaret Unvanı", result.company_info.ticaret_unvani or ""),
        ("Şirket Türü", result.company_info.sirket_turu or ""),
        ("MERSİS Numarası", result.company_info.mersis_numarasi or ""),
        ("Ticaret Sicil Müdürlüğü", result.company_info.ticaret_sicil_mudurlugu or ""),
        ("Ticaret Sicil Numarası", result.company_info.ticaret_sicil_numarasi or ""),
        ("Adres", result.company_info.adres or ""),
        ("Mevcut Sermaye", result.company_info.mevcut_sermaye or ""),
        ("Kuruluş Tarihi", result.company_info.kurulus_tarihi or ""),
        ("Denetçi", result.company_info.denetci or ""),
    ]
    body_parts.append(_table(["Alan", "Değer"], company_rows))

    body_parts.append(_paragraph("Yönetim Kurulu Üyeleri", bold=True))
    board_header = _table_row(("Üye", "Unvan", "Görev Bitiş Tarihi", "Atandığı Tarih / Sayı", "Kaynak PDF"), header=True)
    board_data_rows = []
    for member in result.board_members:
        display = _display_name(member)
        date_str = member.appointed_ttsg_date + (f" / {member.appointed_ttsg_number}" if member.appointed_ttsg_number else "")
        rid = link_registry.get(member.pdf_link)
        pdf_cell = _hyperlink_cell(member.pdf_link, rid) if rid else _plain_cell(member.pdf_link)
        board_data_rows.append(
            "<w:tr>"
            + _plain_cell(display)
            + _plain_cell(member.role or "")
            + _plain_cell(member.role_end_date or "")
            + _plain_cell(date_str)
            + pdf_cell
            + "</w:tr>"
        )
    body_parts.append(_table_xml([board_header] + board_data_rows))

    body_parts.append(_paragraph("Konsolide Esas Sözleşme", bold=True))
    for article in result.articles:
        heading = f"Madde {article.article_no}"
        if article.title:
            heading += f" - {article.title}"
        body_parts.append(_paragraph(heading, bold=True))
        source_line = article.source.publication_date
        if article.source.gazette_number:
            source_line += f" / {article.source.gazette_number}"
        body_parts.append(_paragraph(f"Kaynak: {source_line}"))
        for paragraph in article.body.split("\n"):
            if paragraph.strip():
                body_parts.append(_paragraph(paragraph.strip()))

    body_parts.append(_paragraph("Belge Bazlı Metin Çıkarımı", bold=True))
    for md_file in sorted(target_only_dir.glob("*.md")):
        body_parts.append(_paragraph(md_file.stem, bold=True))
        for line in md_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body_parts.append(_paragraph(stripped.lstrip("# "), bold=True))
            else:
                body_parts.append(_paragraph(stripped))
        body_parts.append(_paragraph(""))

    body_parts.append(_paragraph("Benchmark Özeti", bold=True))

    grand_total: dict[str, float] = {}

    # Mistral OCR cost — read from .cache/ocr/mistral/*.json
    if cache_dir is not None:
        ocr_dir = cache_dir / "ocr" / "mistral"
        if ocr_dir.is_dir():
            body_parts.append(_paragraph("Mistral OCR Maliyeti", bold=True))
            ocr_rows = []
            total_ocr_cost = 0.0
            total_ocr_pages = 0
            total_ocr_ms = 0
            for ocr_file in sorted(ocr_dir.glob("*.json")):
                payload = read_json(ocr_file)
                pages = payload.get("pages", [])
                doc_cost = sum(p.get("cost_usd") or 0 for p in pages)
                doc_pages = len(pages)
                doc_ms = sum(p.get("runtime_ms") or 0 for p in pages)
                ocr_rows.append((
                    payload.get("doc_id", ocr_file.stem),
                    str(doc_pages),
                    f"{doc_ms / 1000:.1f}",
                    f"${doc_cost:.6f}",
                ))
                total_ocr_cost += doc_cost
                total_ocr_pages += doc_pages
                total_ocr_ms += doc_ms
            ocr_rows.append(("TOPLAM", str(total_ocr_pages), f"{total_ocr_ms / 1000:.1f}", f"${total_ocr_cost:.6f}"))
            body_parts.append(_table(["Belge", "Sayfa", "Süre (s)", "Maliyet (USD)"], ocr_rows))
            grand_total["Mistral OCR"] = total_ocr_cost

    def _llm_cost_table(label: str, path: Path) -> None:
        if not path.exists():
            return
        entries: list[dict] = read_json(path)
        body_parts.append(_paragraph(label, bold=True))
        rows = []
        for e in entries:
            rows.append((
                e.get("doc_id", ""),
                str(e.get("input_tokens", "")),
                str(e.get("output_tokens", "")),
                f"{e.get('predict_time_s', 0):.1f}",
                f"${e.get('cost_usd', 0):.6f}",
            ))
        total_cost = sum(e.get("cost_usd", 0) for e in entries)
        total_in = sum(e.get("input_tokens", 0) for e in entries)
        total_out = sum(e.get("output_tokens", 0) for e in entries)
        total_time = sum(e.get("predict_time_s", 0) for e in entries)
        rows.append(("TOPLAM", str(total_in), str(total_out), f"{total_time:.1f}", f"${total_cost:.6f}"))
        body_parts.append(_table(["Belge", "Giriş Token", "Çıkış Token", "Süre (s)", "Maliyet (USD)"], rows))
        grand_total[label] = total_cost

    _llm_cost_table("LLM Metin Temizleme Maliyeti", final_dir / "llm_clean_usage.json")
    _llm_cost_table("LLM Konsolidasyon Maliyeti", final_dir / "llm_usage.json")

    if grand_total:
        body_parts.append(_paragraph("Toplam Pipeline Maliyeti", bold=True))
        summary_rows = [(label, f"${cost:.6f}") for label, cost in grand_total.items()]
        summary_rows.append(("GENEL TOPLAM", f"${sum(grand_total.values()):.6f}"))
        body_parts.append(_table(["Aşama", "Maliyet (USD)"], summary_rows))

    body_parts.append(
        "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""
        " xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<w:body>{''.join(body_parts)}</w:body>"
        "</w:document>"
    )


def _paragraph(text: str, *, bold: bool = False) -> str:
    escaped = html.escape(text)
    run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:p><w:r>{run_props}<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def _table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    row_xml = [_table_row(tuple(headers), header=True)]
    for row in rows:
        row_xml.append(_table_row(row))
    return "<w:tbl><w:tblPr><w:tblBorders>" \
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>" \
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>" \
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>" \
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>" \
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>" \
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/></w:tblBorders></w:tblPr>" \
        + "".join(row_xml) + "</w:tbl>"


def _table_row(cells: tuple[str, ...], *, header: bool = False) -> str:
    cell_xml = []
    for cell in cells:
        escaped = html.escape(cell)
        run_props = "<w:rPr><w:b/></w:rPr>" if header else ""
        cell_xml.append(
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r>{run_props}<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
            "</w:tc>"
        )
    return "<w:tr>" + "".join(cell_xml) + "</w:tr>"


def _content_types_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/word/styles.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
        "</Types>"
    )


def _root_rels_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "</Relationships>"
    )


def _styles_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"/>"
    )


def _display_name(member) -> str:
    if member.representative:
        return f"{member.name_or_title} (adına hareket edecek gerçek kişi: {member.representative})"
    return member.name_or_title


def _plain_cell(text: str, *, bold: bool = False) -> str:
    escaped = html.escape(text)
    run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
        f"<w:p><w:r>{run_props}<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
        "</w:tc>"
    )


def _hyperlink_cell(text: str, rid: str) -> str:
    escaped = html.escape(text)
    return (
        "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
        "<w:p>"
        f'<w:hyperlink r:id="{rid}" w:history="1">'
        "<w:r><w:rPr><w:color w:val=\"0563C1\"/><w:u w:val=\"single\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escaped}</w:t></w:r>"
        "</w:hyperlink>"
        "</w:p>"
        "</w:tc>"
    )


def _table_xml(rows: list[str]) -> str:
    return (
        "<w:tbl><w:tblPr><w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "</w:tblBorders></w:tblPr>"
        + "".join(rows)
        + "</w:tbl>"
    )


def _document_rels_xml(link_registry: dict[str, str], root: Path, deliverables: Path) -> str:
    rels = []
    for pdf_path, rid in link_registry.items():
        abs_pdf = (root / pdf_path).resolve()
        rel = Path(os.path.relpath(abs_pdf, deliverables.resolve())).as_posix()
        target = html.escape(quote(rel, safe="/"))
        rels.append(
            f'<Relationship Id="{rid}"'
            f' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"'
            f' Target="{target}" TargetMode="External"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )
