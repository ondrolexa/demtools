import numpy as np

from demtools import BoolGrid, FloatGrid, IntGrid


class TestProperties:
    def test_count_true(self, bgrid):
        unmasked = (~bgrid._mask).sum()
        assert bgrid.count_true + bgrid.count_false == unmasked

    def test_count_true_values(self, bgrid):
        assert bgrid.count_true == 4

    def test_count_false_values(self, bgrid):
        assert bgrid.count_false == 5


class TestLogical:
    def test_and(self, bgrid):
        other = BoolGrid(
            np.ma.array(
                [[True, False, True], [False, True, False], [True, False, True]]
            ),
            height=3,
            width=3,
        )
        r = bgrid & other
        assert r.data[0, 0]
        assert not r.data[0, 1]
        assert r.data[2, 2]

    def test_and_scalar(self, bgrid):
        r = bgrid & True
        assert r.data[0, 0]
        assert not r.data[0, 2]

    def test_or(self, bgrid):
        other = BoolGrid(
            np.ma.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            height=3,
            width=3,
        )
        r = bgrid | other
        assert r.data[0, 0] == bgrid.data[0, 0]

    def test_or_scalar(self, bgrid):
        r = bgrid | False
        assert r.data[0, 0] == bgrid.data[0, 0]

    def test_neg(self, bgrid):
        r = -bgrid
        assert bool(r.data[0, 0]) == (not bool(bgrid.data[0, 0]))


class TestMorphology:
    def test_erosion(self, bgrid):
        r = bgrid.erosion(n=1)
        assert r.shape == bgrid.shape

    def test_dilation(self, bgrid):
        r = bgrid.dilation(n=1)
        assert r.shape == bgrid.shape

    def test_closing(self, bgrid):
        r = bgrid.closing(n=1)
        assert r.shape == bgrid.shape

    def test_opening(self, bgrid):
        r = bgrid.opening(n=1)
        assert r.shape == bgrid.shape

    def test_fill_holes(self, bgrid):
        r = bgrid.fill_holes()
        assert r.shape == bgrid.shape

    def test_remove_dots(self, bgrid):
        r = bgrid.remove_dots(size=1)
        assert r.shape == bgrid.shape

    def test_clear_boundary(self, bgrid):
        r = bgrid.clear_boundary()
        assert r.shape == bgrid.shape

    def test_clear_boundary_no_mask(self, bgrid):
        r = bgrid.clear_boundary(use_mask=False)
        assert r.shape == bgrid.shape

    def test_label(self, bgrid):
        r = bgrid.label()
        assert isinstance(r, IntGrid)
        assert r.shape == bgrid.shape

    def test_normalized(self, bgrid):
        r = bgrid.normalized()
        assert isinstance(r, FloatGrid)
        assert r.shape == bgrid.shape
