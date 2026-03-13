"""Tests for Phase 1 collage assembly."""

import os

import pytest
from PIL import Image

from src.phase1_paths.collage import (
    _find_month_file,
    _letterbox_fit,
    _ppt_collage,
    _simple_grid,
)


class TestSimpleGrid:
    def test_2x2_grid(self):
        imgs = [Image.new("RGB", (40, 30), c) for c in
                [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]]
        result = _simple_grid(imgs, cols=2, border=4)
        expected_w = 2 * 40 + 3 * 4
        expected_h = 2 * 30 + 3 * 4
        assert result.size == (expected_w, expected_h)

    def test_single_column(self):
        imgs = [Image.new("RGB", (50, 50)) for _ in range(3)]
        result = _simple_grid(imgs, cols=1, border=2)
        assert result.size[0] == 50 + 2 * 2
        assert result.size[1] == 3 * 50 + 4 * 2

    def test_4x3_grid(self):
        imgs = [Image.new("RGB", (20, 20)) for _ in range(12)]
        result = _simple_grid(imgs, cols=4, border=6)
        assert result.size == (4 * 20 + 5 * 6, 3 * 20 + 4 * 6)


class TestLetterboxFit:
    def test_wider_image_fits(self):
        img = Image.new("RGB", (200, 100))
        result = _letterbox_fit(img, 100, 100)
        assert result.size == (100, 100)

    def test_taller_image_fits(self):
        img = Image.new("RGB", (100, 200))
        result = _letterbox_fit(img, 100, 100)
        assert result.size == (100, 100)

    def test_exact_fit(self):
        img = Image.new("RGB", (100, 100))
        result = _letterbox_fit(img, 100, 100)
        assert result.size == (100, 100)


class TestPptCollage:
    def test_produces_correct_size(self):
        imgs = [Image.new("RGB", (100, 80)) for _ in range(6)]
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        result = _ppt_collage(imgs, "Test Title", labels, out_w=800, out_h=600)
        assert result.size == (800, 600)

    def test_fewer_labels_than_images(self):
        imgs = [Image.new("RGB", (100, 80)) for _ in range(6)]
        result = _ppt_collage(imgs, "T", ["A", "B"], out_w=800, out_h=600)
        assert result.size == (800, 600)


class TestFindMonthFile:
    def test_finds_existing_file(self, tmp_path):
        (tmp_path / "overview_03_Mar_eco.png").write_bytes(b"\x89PNG")
        result = _find_month_file(str(tmp_path), "overview", 3)
        assert result is not None
        assert "overview_03_Mar" in result

    def test_returns_none_for_missing(self, tmp_path):
        result = _find_month_file(str(tmp_path), "overview", 5)
        assert result is None

    def test_picks_first_sorted_match(self, tmp_path):
        (tmp_path / "v_01_Jan_a.png").write_bytes(b"\x89PNG")
        (tmp_path / "v_01_Jan_b.png").write_bytes(b"\x89PNG")
        result = _find_month_file(str(tmp_path), "v", 1)
        assert "v_01_Jan_a.png" in result
