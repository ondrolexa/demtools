import pytest

from demtools import DEMGrid


@pytest.fixture
def dem():
    return DEMGrid.example(test=True)
