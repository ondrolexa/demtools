import pytest


def test_min(dem):
    assert dem.min == pytest.approx(508.242681917153)


def test_max(dem):
    assert dem.max == pytest.approx(516.7763943559141)


def test_default_stretch(dem):
    assert not dem.stretch


def test_digitize(dem):
    g = dem.digitize(bins=3)
    assert all(g.unique_values == [1, 2, 3])


def test_normalization(dem):
    g = dem.normalized()
    assert (g.min == pytest.approx(0)) & (g.max == pytest.approx(1))


def test_boolean_select(dem):
    assert dem[(dem >= 510) & (dem <= 515)].mean == pytest.approx(512.7716484527895)


def test_over_aggregate(dem):
    assert dem.aggregate(max(dem.shape)).mean == pytest.approx(dem.mean)
