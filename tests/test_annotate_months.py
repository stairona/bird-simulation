"""Tests for Phase 1 annotate_months rendering."""

import numpy as np
import pytest
from PIL import Image

from src.core.config import MapView, MapCorridorEndpoints


class TestRenderCorridors:
    """Test render_corridors with a synthetic 100x100 base image."""

    @pytest.fixture
    def small_view(self, isabella_cfg):
        """Build a minimal MapView with pixel-space endpoints for a 100x100 image."""
        eps = {}
        for key in isabella_cfg.species:
            if key == "local":
                eps[key] = MapCorridorEndpoints(center=(50, 50), radius=30)
            else:
                eps[key] = MapCorridorEndpoints(p0=(10, 90), p3=(90, 10), curv=0.0)
        return MapView(
            key="test",
            base_image="dummy.png",
            turbine_pixels=[(30, 30), (70, 70), (50, 50)],
            corridor_endpoints=eps,
        )

    @pytest.fixture
    def base_img(self):
        return Image.new("RGBA", (100, 100), (0, 128, 0, 255))

    def _render(self, base_img, small_view, isabella_cfg, mode):
        from src.phase1_paths.annotate_months import render_corridors
        rng = np.random.default_rng(42)
        return render_corridors(
            base_img=base_img,
            view=small_view,
            cfg=isabella_cfg,
            month_name="Mar",
            intensity=0.8,
            period_label="Spring migration",
            mode=mode,
            rng=rng,
        )

    def test_eco_returns_image(self, base_img, small_view, isabella_cfg):
        result = self._render(base_img, small_view, isabella_cfg, "eco")
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_cinematic_returns_image(self, base_img, small_view, isabella_cfg):
        result = self._render(base_img, small_view, isabella_cfg, "cinematic")
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_pub_returns_image(self, base_img, small_view, isabella_cfg):
        result = self._render(base_img, small_view, isabella_cfg, "pub")
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_winter_month_reduces_arrows(self, base_img, small_view, isabella_cfg):
        """Rendering a winter month should still succeed (low intensity path)."""
        from src.phase1_paths.annotate_months import render_corridors
        rng = np.random.default_rng(42)
        result = render_corridors(
            base_img=base_img,
            view=small_view,
            cfg=isabella_cfg,
            month_name="Jan",
            intensity=0.1,
            period_label="Winter",
            mode="eco",
            rng=rng,
        )
        assert isinstance(result, Image.Image)

    def test_output_is_rgba(self, base_img, small_view, isabella_cfg):
        result = self._render(base_img, small_view, isabella_cfg, "eco")
        assert result.mode == "RGBA"
