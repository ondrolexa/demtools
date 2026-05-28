import numpy as np
import pytest

from demtools import BoolGrid, DEMGrid, FloatGrid, IntGrid


@pytest.fixture
def dem():
    return DEMGrid.example(test=True)


@pytest.fixture
def fgrid():
    data = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return FloatGrid(data, height=3, width=3)


@pytest.fixture
def fgrid_masked():
    data = np.ma.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        mask=[[False, False, False], [False, True, False], [False, False, False]],
    )
    return FloatGrid(data, height=3, width=3)


@pytest.fixture
def bgrid():
    data = np.ma.array(
        [[True, True, False], [True, False, False], [False, False, True]]
    )
    return BoolGrid(data, height=3, width=3)


@pytest.fixture
def igrid():
    data = np.ma.array([[1, 1, 2], [2, 3, 3], [3, 4, 5]], dtype=int)
    return IntGrid(data, height=3, width=3)
