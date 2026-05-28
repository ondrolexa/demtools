import numpy as np
import rasterio

from demtools.grids import RGBimage


class TestRGBimage:
    def test_construct_3band(self):
        rgb = np.ones((3, 10, 10), dtype=float)
        img = RGBimage(rgb, height=10, width=10)
        assert img.data.shape == (3, 10, 10)
        assert img.title == "RGB"

    def test_custom_title(self):
        rgb = np.ones((3, 5, 5), dtype=float)
        img = RGBimage(rgb, height=5, width=5, title="test")
        assert img.title == "test"

    def test_wrong_shape_raises(self):
        rgb = np.ones((2, 5, 5), dtype=float)
        with np.testing.assert_raises(AssertionError):
            RGBimage(rgb, height=5, width=5)

    def test_write_tif(self, tmp_path):
        rgb = np.ones((3, 10, 10), dtype=float)
        img = RGBimage(rgb, height=10, width=10)
        path = tmp_path / "test_rgb.tif"
        img.write_tif(str(path))
        assert path.exists()
        with rasterio.open(path) as src:
            assert src.count == 3
            assert src.width == 10
            assert src.height == 10
