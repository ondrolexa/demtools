import pytest


def test_min(dem):
    assert dem.min == pytest.approx(508.242681917153)


def test_max(dem):
    assert dem.max == pytest.approx(516.7763943559141)
