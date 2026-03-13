"""Tests for the map tile fetching module."""

from unittest.mock import patch

import pytest

from src.core.tiles import _check_contextily, fetch_basemap
from src.core.geo import BoundingBox


class TestCheckContextily:
    def test_raises_when_unavailable(self):
        with patch.dict("sys.modules", {"contextily": None}):
            with pytest.raises(ImportError, match="contextily"):
                _check_contextily()

    def test_returns_module_when_available(self):
        try:
            import contextily  # noqa: F401
        except ImportError:
            pytest.skip("contextily not installed")
        result = _check_contextily()
        assert result is not None


class TestFetchBasemapSignature:
    """Test that fetch_basemap accepts expected args without actually downloading."""

    def test_signature_accepts_required_args(self):
        import inspect
        sig = inspect.signature(fetch_basemap)
        params = list(sig.parameters.keys())
        assert "bbox" in params
        assert "out_path" in params
        assert "style" in params

    def test_default_style_is_satellite(self):
        import inspect
        sig = inspect.signature(fetch_basemap)
        assert sig.parameters["style"].default == "satellite"
