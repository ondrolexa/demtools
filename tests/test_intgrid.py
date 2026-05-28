import numpy as np
import pytest

from demtools import BoolGrid, DEMGrid, FloatGrid, IntGrid


class TestProperties:
    def test_unique_values(self, igrid):
        assert list(igrid.unique_values) == [1, 2, 3, 4, 5]

    def test_counts(self, igrid):
        values, counts = igrid.counts()
        assert list(values) == [1, 2, 3, 4, 5]
        assert list(counts) == [2, 2, 3, 1, 1]


class TestDivision:
    def test_truediv_with_intgrid(self, igrid):
        other = IntGrid(
            np.ma.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int), height=3, width=3
        )
        result = igrid // other
        assert isinstance(result, IntGrid)
        assert result.data[0, 0] == igrid.data[0, 0] // 1

    def test_truediv_with_scalar(self, igrid):
        result = igrid // 2
        assert result.data[0, 0] == 0
        assert result.data[2, 2] == 2

    def test_truediv_with_floatgrid(self, igrid):
        other = FloatGrid(
            np.ma.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), height=3, width=3
        )
        result = igrid // other
        assert isinstance(result, IntGrid)

    def test_rtruediv(self, igrid):
        result = 10 // igrid
        assert result.data[0, 0] == 10

    def test_itruediv(self, igrid):
        igrid //= 2
        assert igrid.data[2, 2] == 2


class TestFilters:
    def test_moving_average_int(self, igrid):
        result = igrid.moving_average(r=1)
        assert result.shape == igrid.shape
        assert isinstance(result, IntGrid)

    def test_majority_filter(self, igrid):
        result = igrid.majority_filter(size=3)
        assert result.shape == igrid.shape
        assert isinstance(result, IntGrid)


class TestNormalized:
    def test_normalized(self, igrid):
        r = igrid.normalized()
        assert isinstance(r, FloatGrid)
        assert r.min == pytest.approx(0.0)
        assert r.max == pytest.approx(1.0)

    def test_normalized_values(self, igrid):
        r = igrid.normalized()
        assert r.data[0, 0] == pytest.approx(0.0)
        assert r.data[2, 2] == pytest.approx(1.0)


class TestPolygonize:
    def test_polygonize_returns_list(self, igrid):
        geoms = igrid.polygonize()
        assert isinstance(geoms, list)
        assert len(geoms) > 0

    def test_polygonize_writes_file(self, igrid, tmp_path):
        path = tmp_path / "out.geojson"
        igrid.polygonize(file=str(path))
        assert path.exists()
