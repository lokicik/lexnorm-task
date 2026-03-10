from __future__ import annotations

import argparse

from .commands import run_consolidate, run_export_docx, run_extract_target, run_ocr, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ttsg", description="TTSG OCR pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr", help="Run OCR for one provider.")
    ocr_parser.add_argument("--provider", choices=("mistral", "gemini"), required=True)
    ocr_parser.add_argument("--force", action="store_true", help="Ignore existing OCR cache.")
    ocr_parser.add_argument("--prefer-replicate", action="store_true", help="Use Replicate as primary backend for Gemini.")
    ocr_parser.add_argument("--dpi", type=int, default=400, help="Rasterization DPI (default: 400).")
    ocr_parser.add_argument("--no-crop", action="store_true", help="Disable margin cropping.")
    ocr_parser.add_argument("--tiles", type=int, default=5, help="Split each page into N horizontal tiles for higher accuracy (0=off).")
    ocr_parser.add_argument("--overlap", type=float, default=0.03, help="Tile overlap fraction (default: 0.03).")
    ocr_parser.set_defaults(handler=run_ocr)

    extract_parser = subparsers.add_parser("extract-target", help="Extract target-company text from OCR outputs.")
    extract_parser.add_argument(
        "--providers",
        nargs="*",
        default=("mistral",),
        help="Providers to consider for extraction/consensus.",
    )
    extract_parser.add_argument("--no-clean", action="store_true", help="Skip LLM-based text cleaning (Gemini Flash).")
    extract_parser.set_defaults(handler=run_extract_target)

    consolidate_parser = subparsers.add_parser("consolidate", help="Build company tables and consolidated AoA.")
    consolidate_parser.set_defaults(handler=run_consolidate)

    export_parser = subparsers.add_parser("export-docx", help="Export final outputs to DOCX.")
    export_parser.set_defaults(handler=run_export_docx)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline: OCR, extract, consolidate, export.")
    pipeline_parser.add_argument(
        "--providers",
        nargs="*",
        default=("mistral",),
        help="Providers to run OCR and extraction for.",
    )
    pipeline_parser.add_argument("--force", action="store_true", help="Ignore existing OCR cache.")
    pipeline_parser.add_argument("--prefer-replicate", action="store_true", help="Use Replicate as primary backend for Gemini.")
    pipeline_parser.add_argument("--dpi", type=int, default=400, help="Rasterization DPI (default: 400).")
    pipeline_parser.add_argument("--no-crop", action="store_true", help="Disable margin cropping.")
    pipeline_parser.add_argument("--tiles", type=int, default=5, help="Split each page into N horizontal tiles for higher accuracy (0=off).")
    pipeline_parser.add_argument("--overlap", type=float, default=0.03, help="Tile overlap fraction (default: 0.03).")
    pipeline_parser.add_argument("--no-clean", action="store_true", help="Skip LLM-based text cleaning (Gemini Flash).")
    pipeline_parser.set_defaults(handler=run_pipeline)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args) or 0)
