import h5py
import numpy as np

from demtools.grids import Affine, DEMGrid, riocrs


class H5Store:
    def __init__(self, file):
        self.h5file = h5py.File(file, "r")
        self.data = self.h5file["Band1"]

    def __del__(self):
        self.h5file.close()

    @property
    def x(self):
        return np.array(self.h5file["x"])

    @property
    def y(self):
        return np.array(self.h5file["y"])

    @property
    def mapping(self):
        return self.h5file[self.data.attrs["grid_mapping"].decode()]

    def clip(self, r, h, c, w):
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
            "transform": Affine(gt[1], 0.0, gt[0], 0.0, gt[5], gt[3]),
        }
        return DEMGrid(res, **meta)
        # return res, meta
