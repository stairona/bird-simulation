"""Tests for src.core.geo — geographic projection utilities."""

import numpy as np
import pytest

from src.core.geo import BoundingBox, bounding_box, latlon_to_normalized, normalized_to_pixels


class TestBoundingBox:
    def test_basic_construction(self):
        bb = BoundingBox(lat_min=40.0, lat_max=41.0, lon_min=-85.0, lon_max=-84.0)
        assert bb.center == (40.5, -84.5)
        assert bb.lat_span == pytest.approx(1.0)
        assert bb.lon_span == pytest.approx(1.0)

    def test_center(self):
        bb = BoundingBox(lat_min=10.0, lat_max=20.0, lon_min=30.0, lon_max=50.0)
        assert bb.center == (15.0, 40.0)


class TestBoundingBoxFromPoints:
    def test_padding(self):
        lats = np.array([43.0, 43.1])
        lons = np.array([-84.0, -83.9])
        bb = bounding_box(lats, lons, pad_fraction=0.10)
        assert bb.lat_min < 43.0
        assert bb.lat_max > 43.1
        assert bb.lon_min < -84.0
        assert bb.lon_max > -83.9

    def test_minimum_padding(self):
        lats = np.array([43.0, 43.0])
        lons = np.array([-84.0, -84.0])
        bb = bounding_box(lats, lons)
        assert bb.lat_span > 0
        assert bb.lon_span > 0

    def test_single_point(self):
        lats = np.array([43.0])
        lons = np.array([-84.0])
        bb = bounding_box(lats, lons)
        assert bb.lat_min < 43.0
        assert bb.lat_max > 43.0


class TestLatlonToNormalized:
    def test_output_shape(self):
        lats = np.array([43.0, 43.1, 43.2])
        lons = np.array([-84.0, -84.1, -84.2])
        bb = bounding_box(lats, lons)
        xy = latlon_to_normalized(lats, lons, bb)
        assert xy.shape == (3, 2)

    def test_values_in_unit_range(self):
        rng = np.random.default_rng(42)
        lats = rng.uniform(40, 45, size=50)
        lons = rng.uniform(-90, -85, size=50)
        bb = bounding_box(lats, lons)
        xy = latlon_to_normalized(lats, lons, bb)
        assert xy[:, 0].min() >= 0.0
        assert xy[:, 0].max() <= 1.0
        assert xy[:, 1].min() >= 0.0
        assert xy[:, 1].max() <= 1.0

    def test_north_is_top(self):
        lats = np.array([40.0, 45.0])
        lons = np.array([-84.0, -84.0])
        bb = bounding_box(lats, lons)
        xy = latlon_to_normalized(lats, lons, bb)
        # Higher latitude -> lower y (top of image)
        assert xy[1, 1] < xy[0, 1]

    def test_identical_points(self):
        lats = np.array([43.0, 43.0])
        lons = np.array([-84.0, -84.0])
        bb = bounding_box(lats, lons)
        xy = latlon_to_normalized(lats, lons, bb)
        assert xy.shape == (2, 2)
        # Both points should be at center
        assert xy[0, 0] == pytest.approx(0.5)
        assert xy[0, 1] == pytest.approx(0.5)


class TestNormalizedToPixels:
    def test_basic_conversion(self):
        xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        pixels = normalized_to_pixels(xy, 1920, 1080)
        assert pixels[0] == (0, 0)
        assert pixels[1] == (1919, 1079)
        assert pixels[2] == (959, 539)

    def test_returns_int_tuples(self):
        xy = np.array([[0.33, 0.67]])
        pixels = normalized_to_pixels(xy, 100, 100)
        assert len(pixels) == 1
        assert isinstance(pixels[0][0], int)
        assert isinstance(pixels[0][1], int)
