"""HDF5 storage adapter for DEM grids."""

import h5py
import numpy as np

from demtools.grids import Affine, DEMGrid, riocrs


class H5Store:
    """Read-only adapter that exposes an HDF5 file as a DEMGrid source.

    The HDF5 file must contain ``x``, ``y``, ``Band1`` datasets and a
    ``grid_mapping`` attribute on ``Band1`` pointing to a group with
    ``crs_wkt`` and ``GeoTransform`` attributes.

    Args:
        file (str or pathlib.Path): Path to the HDF5 file.

    Attributes:
        data (h5py.Dataset): The ``Band1`` dataset handle.
    """

    def __init__(self, file):
        """Open an HDF5 file and bind the ``Band1`` dataset handle."""
        self.h5file = h5py.File(file, "r")
        self.data = self.h5file["Band1"]

    def __del__(self):
        """Close the HDF5 file handle when the object is garbage-collected."""
        self.h5file.close()

    def __enter__(self):
        """Return self to support use as a context manager.

        Returns:
            H5Store: This instance.
        """
        return self

    def __exit__(self, *_):
        """Close the HDF5 file handle on context-manager exit."""
        self.h5file.close()

    @property
    def x(self):
        """numpy.ndarray: 1-D array of x coordinates."""
        return np.array(self.h5file["x"])

    @property
    def y(self):
        """numpy.ndarray: 1-D array of y coordinates."""
        return np.array(self.h5file["y"])

    @property
    def mapping(self):
        """h5py.Group: The grid mapping group referenced by ``Band1``."""
        return self.h5file[self.data.attrs["grid_mapping"].decode()]

    def clip(self, r, h, c, w):
        """Extract a rectangular sub-region as a ``DEMGrid``.

        Args:
            r (int): Start row index (0-based, from the top).
            h (int): Height of the clip window in pixels.
            c (int): Start column index (0-based, from the left).
            w (int): Width of the clip window in pixels.

        Returns:
            DEMGrid: The clipped sub-region with correct geotransform.
        """
        ymax = len(self.y)
        fv = self.data.attrs["_FillValue"][0]
        res = np.flipud(
            np.ma.masked_values(self.data[ymax - r - h : ymax - r, c : c + w], fv)
        )
        wkt = self.mapping.attrs["crs_wkt"].decode()
        gt = list(map(float, self.mapping.attrs["GeoTransform"].decode().split()))
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": fv,
            "width": w,
            "height": h,
            "count": 1,
            "compress": "",
            "crs": riocrs.CRS.from_wkt(wkt),
            "transform": Affine(
                gt[1],
                0.0,
                gt[0] + c * gt[1],
                0.0,
                gt[5],
                gt[3] + (ymax - r - h) * gt[5],
            ),
        }
        return DEMGrid(res, **meta)
