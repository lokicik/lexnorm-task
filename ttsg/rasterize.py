from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .cache import CacheLayout
from .models import DocumentMeta


class RasterizationError(RuntimeError):
    pass


def rasterize_document(cache: CacheLayout, document: DocumentMeta, dpi: int = 300, force: bool = False) -> list[Path]:
    output_dir = cache.raster_dir(document, dpi=dpi)
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("page-*.png"))
    if len(existing) == document.page_count:
        return existing

    if _rasterize_with_pypdfium(document.pdf_path, output_dir, dpi):
        pages = sorted(output_dir.glob("page-*.png"))
        if len(pages) == document.page_count:
            return pages

    backend = _choose_backend()
    if backend == "pdftoppm":
        prefix = output_dir / "page"
        _run(["pdftoppm", "-png", "-r", str(dpi), str(document.pdf_path), str(prefix)])
        for generated in sorted(output_dir.glob("page-*.png")):
            page_no = int(generated.stem.split("-")[-1])
            generated.rename(output_dir / f"page-{page_no:04d}.png")
    elif backend == "mutool":
        _run(["mutool", "draw", "-r", str(dpi), "-o", str(output_dir / "page-%04d.png"), str(document.pdf_path)])
    elif backend == "magick":
        _run(
            [
                "magick",
                "-density",
                str(dpi),
                str(document.pdf_path),
                str(output_dir / "page-%04d.png"),
            ]
        )
    elif backend == "swift":
        script = cache.paths.tools / "pdf_rasterizer.swift"
        if not script.exists():
            raise RasterizationError(f"Missing rasterizer script: {script}")
        _run(
            [
                "swift",
                str(script),
                "--input",
                str(document.pdf_path),
                "--output-dir",
                str(output_dir),
                "--dpi",
                str(dpi),
            ]
        )
    else:
        raise RasterizationError("No supported PDF rasterization backend found.")

    pages = sorted(output_dir.glob("page-*.png"))
    if len(pages) != document.page_count:
        raise RasterizationError(
            f"Rasterized {len(pages)} pages for {document.pdf_path.name}, expected {document.page_count}."
        )
    return pages


def _choose_backend() -> str:
    if shutil.which("pdftoppm"):
        return "pdftoppm"
    if shutil.which("mutool"):
        return "mutool"
    if shutil.which("magick"):
        return "magick"
    if os.uname().sysname == "Darwin" and shutil.which("swift"):
        return "swift"
    return ""


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RasterizationError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}"
        )


def _rasterize_with_pypdfium(pdf_path: Path, output_dir: Path, dpi: int) -> bool:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        return False

    scale = dpi / 72.0
    document = pdfium.PdfDocument(str(pdf_path))
    for index, page in enumerate(document, start=1):
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(output_dir / f"page-{index:04d}.png")
    return True
