from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from ttsg.cache import CacheLayout
from ttsg.config import Paths
from ttsg.consolidation import consolidate, extract_company_fields, parse_articles
from ttsg.documents import discover_documents
from ttsg.extract import build_consensus, extract_target_segment
from ttsg.models import ConsensusDocument, OCRDocument, OCRPage, SourceRef


def _fake_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Pages/Count 1>>endobj\n")


class ExtractionTests(unittest.TestCase):
    def test_extract_target_segment_stops_at_next_company_heading(self) -> None:
        page = OCRPage(
            provider="mistral",
            doc_id="doc-1",
            page=1,
            source_image="page-0001.png",
            raw_markdown="\n".join(
                [
                    "DIGER ORNEK ANONIM SIRKETI",
                    "Baska satir",
                    "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                    "ADRES: ORNEK MAHALLESI ANKARA",
                    "SIRKETIN SERMAYESI 500.000 TL'DIR.",
                    "ACME ANONIM SIRKETI",
                    "Baska ilan",
                ]
            ),
            runtime_ms=1,
        )
        document = OCRDocument(
            provider="mistral",
            doc_id="doc-1",
            pages=[page],
            created_at="2026-03-09T00:00:00Z",
            input_sha256="abc",
        )

        segment = extract_target_segment(document)

        self.assertEqual(segment.status, "accepted")
        self.assertIn("PARLA ENERJI YATIRIMLARI ANONIM SIRKETI", segment.text)
        self.assertNotIn("ACME ANONIM SIRKETI", segment.text)

    def test_build_consensus_marks_manual_review_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            review_dir = Path(tempdir)
            left = extract_target_segment(
                OCRDocument(
                    provider="mistral",
                    doc_id="doc-1",
                    pages=[
                        OCRPage(
                            provider="mistral",
                            doc_id="doc-1",
                            page=1,
                            source_image="left.png",
                            raw_markdown="PARLA ENERJI YATIRIMLARI ANONIM SIRKETI\nSATIR A",
                            runtime_ms=1,
                        )
                    ],
                    created_at="x",
                    input_sha256="x",
                )
            )
            right = extract_target_segment(
                OCRDocument(
                    provider="gemini",
                    doc_id="doc-1",
                    pages=[
                        OCRPage(
                            provider="gemini",
                            doc_id="doc-1",
                            page=1,
                            source_image="right.png",
                            raw_markdown="PARLA ENERJI YATIRIMLARI ANONIM SIRKETI\nSATIR B",
                            runtime_ms=1,
                        )
                    ],
                    created_at="x",
                    input_sha256="x",
                )
            )

            consensus = build_consensus("doc-1", [left, right], review_dir=review_dir)

            self.assertEqual(consensus.status, "accepted_primary")
            self.assertTrue((review_dir / "doc-1.md").exists())


class ConsolidationTests(unittest.TestCase):
    def test_parse_articles_and_company_fields(self) -> None:
        source = SourceRef(
            publication_date="2022-12-01",
            gazette_number="10716",
            pdf_path="data/1.pdf",
            pages=[1, 2],
            provider="mistral",
        )
        text = "\n".join(
            [
                "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                "MERSIS NO: 0123456789012345",
                "ANKARA TICARET SICIL MUDURLUGU",
                "TICARET SICIL NO: 12345",
                "ADRES: ORNEK MAHALLESI ANKARA",
                "1. KURULUS",
                "Kurulus metni",
                "2. SIRKETIN UNVANI",
                "Sirketin unvani Parla Enerji dir.",
                "6. SERMAYE",
                "Sirketin sermayesi 500.000 TL'dir.",
            ]
        )

        fields = extract_company_fields(text)
        articles = parse_articles(text, source)

        self.assertEqual(fields["ticaret_unvani"], "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI")
        self.assertEqual(fields["sirket_turu"], "Anonim \u015eirket")
        self.assertEqual(len(articles), 3)
        self.assertEqual(articles[0].article_no, "1")
        self.assertEqual(articles[1].article_no, "2")
        self.assertEqual(articles[2].article_no, "6")

    def test_consolidate_and_export_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            data_dir = root / "data"
            data_dir.mkdir()

            pdfs = [
                "1) 01-12-2022 Kurulus İlanı.pdf",
                "2) 21-06-2023 Esas Sözleşme Değişikliği–Sermaye Artırımı.pdf",
                "3) 10-10-2023 Yönetim Kurulu Atama Görevden Alma, Temsilci ve Yetkili Kişi Atanması.pdf",
                "4) 22-12-2025 Denetçi Değişikliği.pdf",
            ]
            for filename in pdfs:
                _fake_pdf(data_dir / filename)

            paths = Paths.from_root(root)
            paths.ensure_runtime_dirs()
            cache = CacheLayout(paths)
            documents = discover_documents(paths)

            texts = {
                1: "\n".join(
                    [
                        "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                        "MERSIS NO: 0123456789012345",
                        "ANKARA TICARET SICIL MUDURLUGU",
                        "TICARET SICIL NO: 12345",
                        "ADRES: ORNEK MAHALLESI ANKARA",
                        "01.12.2022 TARIHINDE TESCIL EDILMISTIR",
                        "1. KURULUS",
                        "Kurulus metni",
                        "6. SERMAYE",
                        "Sirketin sermayesi 500.000 TL'dir.",
                        "8. YONETIM KURULU",
                        "AHMET YILMAZ 13.09.2027 TARIHINE KADAR YONETIM KURULU UYESI OLARAK SECILDI.",
                    ]
                ),
                2: "\n".join(
                    [
                        "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                        "6. SERMAYE",
                        "Sirketin sermayesi 750.000 TL'dir.",
                    ]
                ),
                3: "\n".join(
                    [
                        "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                        "AHMET YILMAZ GOREVDEN ALINDI.",
                        "MEHMET DEMIR 13.09.2028 TARIHINE KADAR YONETIM KURULU UYESI OLARAK ATANDI.",
                    ]
                ),
                4: "\n".join(
                    [
                        "PARLA ENERJI YATIRIMLARI ANONIM SIRKETI",
                        "XYZ BAGIMSIZ DENETIM A.S. DENETCI OLARAK ATANDI.",
                    ]
                ),
            }

            for document in documents:
                consensus = ConsensusDocument(
                    doc_id=document.doc_id,
                    normalized_text=texts[document.index],
                    status="accepted",
                    providers=["mistral"],
                    source_pages=[1],
                    notes=[],
                )
                cache.save_consensus(document, consensus)

            result = consolidate(cache)

            self.assertEqual(result.company_info.mevcut_sermaye, "750.000 TL")
            self.assertEqual(result.company_info.denetci, "XYZ BAGIMSIZ DENETIM A.S")
            self.assertEqual([member.name_or_title for member in result.board_members], ["MEHMET DEMIR"])
            article_map = {article.article_no: article for article in result.articles}
            self.assertEqual(article_map["6"].body, "Sirketin sermayesi 750.000 TL'dir.")

            from ttsg.export_docx import export_docx

            output_path = export_docx(cache, result)
            self.assertTrue(output_path.exists())
            with zipfile.ZipFile(output_path) as archive:
                self.assertIn("word/document.xml", archive.namelist())


if __name__ == "__main__":
    unittest.main()
