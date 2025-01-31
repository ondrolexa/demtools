import pytest

from demtools import DEMGrid


@pytest.fixture
def dem():
    return DEMGrid.from_examples("testdem")
