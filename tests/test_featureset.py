import numpy as np
import pytest

from demtools.grids import FeatureSet


@pytest.fixture
def small_feature_data():
    mask = np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )
    n_valid = mask.size - mask.sum()
    data = np.arange(n_valid * 3, dtype=float).reshape(n_valid, 3)
    meta = {
        "height": 3,
        "width": 3,
        "crs": None,
        "transform": None,
        "driver": "GTiff",
        "dtype": "float64",
        "nodata": -9999.0,
        "count": 1,
        "compress": "",
    }
    return data, mask, meta


class TestFeatureSet:
    def test_construct_no_cluster(self, small_feature_data):
        data, mask, meta = small_feature_data
        fs = FeatureSet(data, mask, meta, do_cluster=False)
        assert fs.clusters is None
        assert fs.centers is None
        assert fs.labels is None

    def test_cluster(self, small_feature_data):
        data, mask, meta = small_feature_data
        fs = FeatureSet(data, mask, meta, do_cluster=True, n_kmeans=2, n_clusters=2)
        assert fs.clusters is not None
        assert fs.centers is not None
        assert fs.labels is not None
        assert len(fs.clusters) == data.shape[0]

    def test_labels_average(self, small_feature_data):
        data, mask, meta = small_feature_data
        fs = FeatureSet(data, mask, meta, do_cluster=True, n_kmeans=2, n_clusters=2)
        avg = fs.labels_average()
        assert avg.shape[0] == len(np.unique(fs.labels))

    def test_as_grid(self, small_feature_data):
        data, mask, meta = small_feature_data
        fs = FeatureSet(data, mask, meta, do_cluster=True, n_kmeans=2, n_clusters=2)
        from demtools import IntGrid

        g = fs.as_grid()
        assert isinstance(g, IntGrid)
        assert g.shape == (3, 3)

    def test_as_grid_clusters(self, small_feature_data):
        data, mask, meta = small_feature_data
        fs = FeatureSet(data, mask, meta, do_cluster=True, n_kmeans=2, n_clusters=2)
        g = fs.as_grid(clusters=True)
        assert g.shape == (3, 3)
