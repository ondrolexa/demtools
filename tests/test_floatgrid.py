import numpy as np
import pytest

from demtools import BoolGrid, FloatGrid, IntGrid
from demtools.grids import RGBimage


class TestProperties:
    def test_min(self, fgrid):
        assert fgrid.min == pytest.approx(1.0)

    def test_max(self, fgrid):
        assert fgrid.max == pytest.approx(9.0)

    def test_mean(self, fgrid):
        assert fgrid.mean == pytest.approx(5.0)

    def test_min_masked(self, fgrid_masked):
        assert fgrid_masked.min == pytest.approx(1.0)

    def test_max_masked(self, fgrid_masked):
        assert fgrid_masked.max == pytest.approx(9.0)

    def test_mean_masked(self, fgrid_masked):
        assert fgrid_masked.mean == pytest.approx(5.0)


class TestNormalized:
    def test_normalized(self, fgrid):
        r = fgrid.normalized()
        assert isinstance(r, FloatGrid)
        assert r.min == pytest.approx(0.0)
        assert r.max == pytest.approx(1.0)

    def test_inverted(self, fgrid):
        r = fgrid.inverted()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape


class TestMovingAverage:
    def test_moving_average(self, fgrid):
        r = fgrid.moving_average(r=1)
        assert r.shape == fgrid.shape
        assert isinstance(r, FloatGrid)


class TestDigitize:
    def test_digitize(self, fgrid):
        r = fgrid.digitize(bins=3)
        assert isinstance(r, IntGrid)
        assert r.shape == fgrid.shape

    def test_digitize_unique_values(self, fgrid):
        r = fgrid.digitize(bins=3)
        assert len(r.unique_values) <= 3


class TestResample:
    def test_resample_down(self, fgrid):
        r = fgrid.resample(scale=2, resampling="average")
        assert r.shape == (1, 1)

    def test_resample_nearest(self, fgrid):
        r = fgrid.resample(scale=2, resampling="nearest")
        assert r.shape == (1, 1)


class TestDerivatives:
    def test_dx(self, fgrid):
        r = fgrid.dx()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_dy(self, fgrid):
        r = fgrid.dy()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_dz(self, fgrid):
        r = fgrid.dz()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_thd(self, fgrid):
        r = fgrid.thd()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_tga(self, fgrid):
        r = fgrid.tga()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_theta(self, fgrid):
        r = fgrid.theta()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_nthd(self, fgrid):
        r = fgrid.nthd()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_tilt(self, fgrid):
        r = fgrid.tilt()
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_upcont_zero_height(self, fgrid):
        r = fgrid.upcont(0)
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape

    def test_upcont_nonzero_height(self, fgrid):
        r = fgrid.upcont(10)
        assert isinstance(r, FloatGrid)
        assert r.shape == fgrid.shape


class TestFilters:
    def test_gaussian_filter(self, fgrid):
        r = fgrid.gaussian_filter(sigma=1)
        assert r.shape == fgrid.shape
        assert isinstance(r, FloatGrid)

    def test_median_filter(self, fgrid):
        r = fgrid.median_filter(size=3)
        assert r.shape == fgrid.shape
        assert isinstance(r, FloatGrid)


class TestOverlay:
    @pytest.fixture
    def over(self):
        return FloatGrid(
            np.ma.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            height=3,
            width=3,
        )

    def test_overlay(self, fgrid, over):
        r = fgrid.overlay(over)
        assert isinstance(r, RGBimage)

    def test_overlay_invert(self, fgrid, over):
        r = fgrid.overlay(over, invert=True)
        assert isinstance(r, RGBimage)
