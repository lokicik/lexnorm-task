from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ttsg.preprocess import crop_margins, preprocess_pages, split_tiles, tile_pages


class TestCropMargins(unittest.TestCase):
    def test_crop_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img = Image.new("RGB", (1000, 2000), color="white")
            src = tmp_path / "input.png"
            img.save(src)

            dst = tmp_path / "output.png"
            crop_margins(src, dst, top_pct=0.04, bottom_pct=0.02, left_pct=0.04, right_pct=0.04)

            with Image.open(dst) as result:
                w, h = result.size
                self.assertEqual(w, 1000 - 40 - 40)  # 920
                self.assertEqual(h, 2000 - 80 - 40)  # 1880

    def test_no_crop_returns_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = []
            for i in range(3):
                img = Image.new("RGB", (100, 100), color="white")
                p = tmp_path / f"page-{i:04d}.png"
                img.save(p)
                paths.append(p)

            out_dir = tmp_path / "preprocessed"
            result = preprocess_pages(paths, out_dir, crop=False)
            self.assertEqual(result, paths)
            self.assertFalse(out_dir.exists())

    def test_preprocess_creates_cropped_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = []
            for i in range(2):
                img = Image.new("RGB", (500, 1000), color="white")
                p = tmp_path / f"page-{i:04d}.png"
                img.save(p)
                paths.append(p)

            out_dir = tmp_path / "preprocessed"
            result = preprocess_pages(paths, out_dir, crop=True)
            self.assertEqual(len(result), 2)
            for p in result:
                self.assertTrue(p.exists())
                with Image.open(p) as img:
                    self.assertLess(img.size[0], 500)
                    self.assertLess(img.size[1], 1000)


class TestSplitTiles(unittest.TestCase):
    def test_split_5_tiles_3pct_overlap(self) -> None:
        """5 tiles at 3% overlap should produce 5 files with correct dimensions."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            h = 4676  # approx A4 @ 400 DPI
            w = 3307
            img = Image.new("RGB", (w, h), color="white")
            src = tmp_path / "page-0001.png"
            img.save(src)

            out_dir = tmp_path / "tiles"
            tiles = split_tiles(src, out_dir, num_tiles=5, overlap_pct=0.03)
            self.assertEqual(len(tiles), 5)
            for t in tiles:
                self.assertTrue(t.exists())

            # First and last tiles should be smaller (no overlap on one side)
            with Image.open(tiles[0]) as t0:
                self.assertEqual(t0.size[0], w)
                # First tile: net_h + overlap on bottom only
                expected_h0 = int(1 * h / 5) + int(h * 0.03)
                self.assertEqual(t0.size[1], expected_h0)

            # Middle tile should have overlap on both sides
            with Image.open(tiles[2]) as t2:
                self.assertEqual(t2.size[0], w)
                self.assertGreater(t2.size[1], h // 5)

    def test_tile_pages_caching(self) -> None:
        """tile_pages should reuse cached tiles on second call."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = []
            for i in range(2):
                img = Image.new("RGB", (200, 1000), color="white")
                p = tmp_path / f"page-{i:04d}.png"
                img.save(p)
                paths.append(p)

            out_dir = tmp_path / "tiles"
            result1 = tile_pages(paths, out_dir, num_tiles=5, overlap_pct=0.03)
            self.assertEqual(len(result1), 2)
            self.assertEqual(len(result1[0]), 5)

            # Second call should hit cache
            result2 = tile_pages(paths, out_dir, num_tiles=5, overlap_pct=0.03)
            self.assertEqual(len(result2), 2)
            self.assertEqual(result2[0], result1[0])


if __name__ == "__main__":
    unittest.main()
