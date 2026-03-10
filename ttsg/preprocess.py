from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


def crop_margins(
    image_path: Path,
    output_path: Path,
    *,
    top_pct: float = 0.04,
    bottom_pct: float = 0.02,
    left_pct: float = 0.04,
    right_pct: float = 0.04,
) -> Path:
    with Image.open(image_path) as img:
        w, h = img.size
        left = int(w * left_pct)
        upper = int(h * top_pct)
        right = w - int(w * right_pct)
        lower = h - int(h * bottom_pct)
        cropped = img.crop((left, upper, right, lower))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    return output_path


def preprocess_pages(
    image_paths: list[Path],
    output_dir: Path,
    *,
    crop: bool = True,
    force: bool = False,
) -> list[Path]:
    if not crop:
        return image_paths

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("page-*.png"))
    if len(existing) == len(image_paths):
        return existing

    results: list[Path] = []
    for image_path in image_paths:
        out = output_dir / image_path.name
        crop_margins(image_path, out)
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Tile splitting – split each page into horizontal strips for higher OCR
# accuracy.  Tiles overlap by *overlap_pct* so that text at the boundary is
# fully captured in at least one tile.
# ---------------------------------------------------------------------------

def split_tiles(
    image_path: Path,
    output_dir: Path,
    *,
    num_tiles: int = 5,
    overlap_pct: float = 0.03,
) -> list[Path]:
    """Split *image_path* into *num_tiles* horizontal strips with overlap.

    Returns a list of tile image paths in top-to-bottom order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem  # e.g. "page-0001"

    with Image.open(image_path) as img:
        w, h = img.size
        # Each tile's net height (without overlap)
        net_h = h / num_tiles
        overlap_px = int(h * overlap_pct)
        tiles: list[Path] = []
        for i in range(num_tiles):
            upper = max(0, int(i * net_h) - (overlap_px if i > 0 else 0))
            lower = min(h, int((i + 1) * net_h) + (overlap_px if i < num_tiles - 1 else 0))
            tile = img.crop((0, upper, w, lower))
            tile_path = output_dir / f"{stem}-tile-{i + 1}.png"
            tile.save(tile_path)
            tiles.append(tile_path)
    return tiles


def tile_pages(
    image_paths: list[Path],
    output_dir: Path,
    *,
    num_tiles: int = 5,
    overlap_pct: float = 0.03,
    force: bool = False,
) -> list[list[Path]]:
    """Split every page into tiles.  Returns a list-of-lists (one per page)."""
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick cache check – if the expected number of tile files exist, reuse
    expected = len(image_paths) * num_tiles
    existing = sorted(output_dir.glob("page-*-tile-*.png"))
    if len(existing) == expected:
        result: list[list[Path]] = []
        for image_path in image_paths:
            stem = image_path.stem
            page_tiles = sorted(output_dir.glob(f"{stem}-tile-*.png"))
            result.append(page_tiles)
        return result

    result = []
    for image_path in image_paths:
        tiles = split_tiles(
            image_path, output_dir, num_tiles=num_tiles, overlap_pct=overlap_pct,
        )
        result.append(tiles)
    return result
