import numpy as np
import pytest
import rasterio

from demtools import BoolGrid, FloatGrid, IntGrid


class TestConstruction:
    def test_from_array(self, fgrid):
        assert fgrid.shape == (3, 3)
        assert fgrid.meta["height"] == 3
        assert fgrid.meta["width"] == 3
        assert fgrid.meta["crs"].to_epsg() == 3857
        assert fgrid.resolution == (1.0, 1.0)

    def test_from_array_with_mask_kwarg(self):
        data = np.ma.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        g = FloatGrid(data, mask=mask, height=2, width=2)
        assert g._mask[0, 1]

    def test_repr(self, fgrid):
        r = repr(fgrid)
        assert "FloatGrid" in r
        assert "(3, 3)" in r

    def test_properties(self, fgrid):
        assert isinstance(fgrid._array, np.ndarray)
        assert fgrid._mask.shape == (3, 3)
        assert fgrid.filled.shape == (3, 3)

    def test_filled_uses_nodata(self):
        data = np.ma.array(
            [[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]]
        )
        g = FloatGrid(data, height=2, width=2, nodata=-9999.0)
        assert g.filled[0, 1] == -9999.0

    def test_meta_nodata_correct_per_subclass(self):
        import numpy as np

        from demtools import BoolGrid, FloatGrid, IntGrid

        fg = FloatGrid(np.ma.array([[1.0]], dtype=float), height=1, width=1)
        assert np.isnan(fg.meta["nodata"])
        ig = IntGrid(np.ma.array([[1]], dtype=int), height=1, width=1)
        assert ig.meta["nodata"] == -9999
        bg = BoolGrid(np.ma.array([[True]]), height=1, width=1)
        assert bg.meta["nodata"] is False


class TestArithmetic:
    def test_add_grid(self, fgrid):
        result = fgrid + fgrid
        assert result.data[0, 0] == pytest.approx(2.0)

    def test_add_scalar(self, fgrid):
        result = fgrid + 10
        assert result.data[0, 0] == pytest.approx(11.0)

    def test_radd(self, fgrid):
        result = 10 + fgrid
        assert result.data[0, 0] == pytest.approx(11.0)

    def test_iadd(self, fgrid):
        fgrid += 10
        assert fgrid.data[0, 0] == pytest.approx(11.0)

    def test_sub_grid(self, fgrid):
        result = fgrid - fgrid
        assert result.data[1, 1] == pytest.approx(0.0)

    def test_sub_scalar(self, fgrid):
        result = fgrid - 1
        assert result.data[0, 0] == pytest.approx(0.0)

    def test_rsub(self, fgrid):
        result = 10 - fgrid
        assert result.data[0, 0] == pytest.approx(9.0)

    def test_isub(self, fgrid):
        fgrid -= 1
        assert fgrid.data[2, 2] == pytest.approx(8.0)

    def test_mul_grid(self, fgrid):
        result = fgrid * fgrid
        assert result.data[1, 1] == pytest.approx(25.0)

    def test_mul_scalar(self, fgrid):
        result = fgrid * 2
        assert result.data[0, 0] == pytest.approx(2.0)

    def test_rmul(self, fgrid):
        result = 2 * fgrid
        assert result.data[0, 0] == pytest.approx(2.0)

    def test_imul(self, fgrid):
        fgrid *= 2
        assert fgrid.data[1, 1] == pytest.approx(10.0)

    def test_truediv_grid(self, fgrid):
        result = fgrid / fgrid
        assert result.data[1, 1] == pytest.approx(1.0)

    def test_truediv_scalar(self, fgrid):
        result = fgrid / 2
        assert result.data[0, 0] == pytest.approx(0.5)

    def test_rtruediv(self, fgrid):
        result = 1 / fgrid
        assert result.data[1, 1] == pytest.approx(0.2)

    def test_itruediv(self, fgrid):
        fgrid /= 2
        assert fgrid.data[2, 2] == pytest.approx(4.5)

    def test_floordiv(self, fgrid):
        f2 = fgrid.clone(fgrid.data.astype(int), astype=IntGrid)
        result = f2 // 2
        assert result.data[0, 0] == 0

    def test_pow(self, fgrid):
        result = fgrid**2
        assert result.data[1, 1] == pytest.approx(25.0)

    def test_rpow(self, fgrid):
        result = 2**fgrid
        assert result.data[1, 1] == pytest.approx(32.0)


class TestComparison:
    def test_lt(self, fgrid):
        r = fgrid < 5
        assert isinstance(r, BoolGrid)
        assert r.data[0, 0]
        assert not r.data[2, 2]

    def test_le(self, fgrid):
        r = fgrid <= 5
        assert r.data[1, 1]

    def test_eq(self, fgrid):
        r = fgrid == 5
        assert r.data[1, 1]
        assert not r.data[0, 0]

    def test_ne(self, fgrid):
        r = fgrid != 5
        assert r.data[0, 0]
        assert not r.data[1, 1]

    def test_gt(self, fgrid):
        r = fgrid > 5
        assert r.data[2, 2]
        assert not r.data[0, 0]

    def test_ge(self, fgrid):
        r = fgrid >= 5
        assert r.data[1, 1]
        assert r.data[2, 2]

    def test_comparison_chained(self, fgrid):
        r = (fgrid >= 3) & (fgrid <= 7)
        assert isinstance(r, BoolGrid)
        assert not r.data[0, 0]
        assert r.data[0, 2]
        assert r.data[2, 0]
        assert not r.data[2, 1]
        assert not r.data[2, 2]


class TestCopyClone:
    def test_clone_same_type(self, fgrid):
        c = fgrid.clone(fgrid.data)
        assert type(c) is FloatGrid
        assert c.shape == fgrid.shape

    def test_clone_astype(self, fgrid):
        c = fgrid.clone(fgrid.data, astype=IntGrid)
        assert type(c) is IntGrid

    def test_clone_with_new_mask(self, fgrid):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        c = fgrid.clone(fgrid.data, mask=mask)
        assert c._mask[0, 0]

    def test_copy(self, fgrid):
        c = fgrid.copy()
        assert type(c) is FloatGrid
        assert c.shape == fgrid.shape
        assert c.data[0, 0] == fgrid.data[0, 0]

    def test_clone_preserves_meta(self, fgrid):
        fgrid.meta["compress"] = "lzw"
        c = fgrid.clone(fgrid.data)
        assert c.meta["compress"] == "lzw"


class TestTypeConversion:
    def test_asbool(self, fgrid):
        b = fgrid.asbool()
        assert isinstance(b, BoolGrid)

    def test_asint(self, fgrid):
        i = fgrid.asint()
        assert isinstance(i, IntGrid)

    def test_asfloat(self, fgrid):
        f = fgrid.asfloat()
        assert isinstance(f, FloatGrid)


class TestIndexing:
    def test_getitem_boolgrid(self, fgrid):
        mask = fgrid > 5
        selected = fgrid[mask]
        assert selected.data[2, 2] == pytest.approx(9.0)
        assert selected._mask[0, 0]

    def test_getitem_ndarray(self, fgrid):
        mask = np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        selected = fgrid[mask]
        assert not selected._mask[0, 0]
        assert selected._mask[2, 2]

    def test_setitem(self, fgrid):
        mask = np.array(
            [[False, False, False], [False, False, False], [True, True, True]]
        )
        fgrid[mask] = 99
        assert fgrid.data[2, 0] == pytest.approx(99.0)
        assert fgrid.data[0, 0] == pytest.approx(1.0)

    def test_setitem_with_boolgrid(self, fgrid):
        mask = BoolGrid(
            np.ma.array(
                [[False, False, False], [False, True, False], [False, False, True]]
            ),
            height=3,
            width=3,
        )
        fgrid[mask] = 99
        assert fgrid.data[1, 1] == pytest.approx(99.0)
        assert fgrid.data[2, 2] == pytest.approx(99.0)
        assert fgrid.data[0, 0] == pytest.approx(1.0)


class TestIO:
    def test_from_file(self, dem):
        assert dem.shape == (24, 26)

    def test_write_tif_roundtrip(self, fgrid, tmp_path):
        path = tmp_path / "test.tif"
        fgrid.write_tif(str(path))
        assert path.exists()
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.height == 3
            assert src.width == 3

    def test_write_tif_masked(self, fgrid_masked, tmp_path):
        path = tmp_path / "test_masked.tif"
        fgrid_masked.write_tif(str(path))
        assert path.exists()


class TestClip:
    def test_clip(self, fgrid):
        clipped = fgrid.clip(0, 2, 0, 2)
        assert clipped.shape == (2, 2)
        assert clipped.data[0, 0] == pytest.approx(1.0)
        assert clipped.data[1, 1] == pytest.approx(5.0)

    def test_clip_updates_transform(self, fgrid):
        clipped = fgrid.clip(0, 2, 2, 1)
        assert clipped.meta["transform"].c != fgrid.meta["transform"].c


class TestAggregate:
    def test_aggregate_mean(self, fgrid):
        agg = fgrid.aggregate(3)
        assert agg.shape == (1, 1)
        assert agg.data[0, 0] == pytest.approx(5.0)

    def test_aggregate_maximum(self, fgrid):
        agg = fgrid.aggregate(3, method="maximum")
        assert agg.data[0, 0] == pytest.approx(9.0)

    def test_aggregate_minimum(self, fgrid):
        agg = fgrid.aggregate(3, method="minimum")
        assert agg.data[0, 0] == pytest.approx(1.0)

    def test_aggregate_median(self, fgrid):
        agg = fgrid.aggregate(3, method="median")
        assert agg.data[0, 0] == pytest.approx(5.0)

    def test_aggregate_std(self, fgrid):
        agg = fgrid.aggregate(3, method="standard_deviation")
        assert agg.data[0, 0] == pytest.approx(np.std(np.arange(1, 10), ddof=0))

    def test_aggregate_variance(self, fgrid):
        agg = fgrid.aggregate(3, method="variance")
        assert agg.data[0, 0] == pytest.approx(np.var(np.arange(1, 10), ddof=0))

    def test_aggregate_invalid_method(self, fgrid):
        with pytest.raises(ValueError):
            fgrid.aggregate(3, method="invalid")

    def test_aggregate_std_vs_variance_differ(self, fgrid):
        std_result = fgrid.aggregate(3, method="standard_deviation")
        var_result = fgrid.aggregate(3, method="variance")
        assert std_result.data[0, 0] != pytest.approx(var_result.data[0, 0])


class TestFilter:
    def test_generic_filter(self, fgrid):
        result = fgrid.generic_filter(np.mean, size=3)
        assert result.shape == fgrid.shape

    def test_correlation(self, fgrid):
        filt = np.ones((3, 3))
        result = fgrid.correlation(filt)
        assert result.shape == fgrid.shape

    def test_similarity(self, fgrid):
        filt = np.ones((3, 3))
        result = fgrid.similarity(filt)
        assert result.shape == fgrid.shape

    def test_correlation_odd_filter_assert(self, fgrid):
        with pytest.raises(ValueError):
            fgrid.correlation(np.ones((2, 2)))


class TestSample:
    def test_sample(self, fgrid):
        pts = [(1.0, -1.0)]
        vals = fgrid.sample(pts)
        assert len(vals) == 1
        assert vals[0] == pytest.approx(5.0)

    def test_sample_line(self, fgrid):
        vals = fgrid.sample_line((0.5, -0.5), (1.5, -1.5), n=3)
        assert len(vals) == 3

    def test_index(self, fgrid):
        idx = fgrid.index((1.0, -1.0))
        assert idx is not None
        r, c = idx
        assert fgrid.data[r, c] == pytest.approx(5.0)


class TestAsDataset:
    def test_asdataset_context(self, fgrid):
        with fgrid.asdataset() as src:
            assert src.count == 1
            data = src.read([1])[0]
            assert data.shape == (3, 3)
            assert data[0, 0] == pytest.approx(1.0)
