import numpy as np
import pytest

from demtools import BoolGrid, FloatGrid, IntGrid


class TestExisting:
    def test_min(self, dem):
        assert dem.min == pytest.approx(508.242681917153)

    def test_max(self, dem):
        assert dem.max == pytest.approx(516.7763943559141)

    def test_default_stretch(self, dem):
        assert not dem.stretch

    def test_digitize(self, dem):
        g = dem.digitize(bins=3)
        assert all(g.unique_values == [1, 2, 3])

    def test_normalization(self, dem):
        g = dem.normalized()
        assert (g.min == pytest.approx(0)) & (g.max == pytest.approx(1))

    def test_boolean_select(self, dem):
        assert dem[(dem >= 510) & (dem <= 515)].mean == pytest.approx(512.7716484527895)

    def test_over_aggregate(self, dem):
        assert dem.aggregate(max(dem.shape)).mean == pytest.approx(dem.mean)


class TestHillshade:
    def test_hillshade(self, dem):
        r = dem.hillshade()
        assert isinstance(r, FloatGrid)
        assert r.shape == dem.shape

    def test_shade(self, dem):
        r = dem.shade()
        from demtools.grids import RGBimage

        assert isinstance(r, RGBimage)


class TestTPI:
    def test_tpi(self, dem):
        r = dem.tpi(r=2)
        assert isinstance(r, FloatGrid)
        assert r.shape == dem.shape

    def test_tpi_fft(self, dem):
        r = dem.tpi_fft(r=2)
        assert isinstance(r, FloatGrid)
        assert r.shape == dem.shape


class TestPits:
    def test_pits(self, dem):
        r = dem.pits(size=3)
        assert isinstance(r, BoolGrid)
        assert r.shape == dem.shape

    def test_pits2(self, dem):
        r = dem.pits2(size=3)
        assert isinstance(r, BoolGrid)

    def test_flats(self, dem):
        r = dem.flats(size=3)
        assert isinstance(r, BoolGrid)

    def test_flats2(self, dem):
        r = dem.flats2(size=3)
        assert isinstance(r, BoolGrid)

    def test_fill_pits(self, dem):
        r = dem.fill_pits(size=3, eps=0)
        assert isinstance(r, FloatGrid)
        assert r.shape == dem.shape

    def test_sink_points(self, dem):
        points = dem.sink_points(size=3)
        assert isinstance(points, list)
