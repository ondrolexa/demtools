import importlib.resources
from collections import Counter
from contextlib import contextmanager
from functools import cached_property

import geojson
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import rasterio as rio
import rasterio.crs as riocrs
import rasterio.features as riofeatures
import rasterio.plot as rioplot
import rasterio.sample as riosample
import shapely.geometry as shapelygeom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from scipy import ndimage, signal
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm import tqdm

from demtools.mathlib import derivx, derivy, derivz, upcontinue


class Grid:
    __dtype = None
    __fill_value = None

    def __init__(self, data, **kwargs):
        self.cmap = kwargs.get("cmap", None)
        self.title = kwargs.get("title", None)
        self.stretch = kwargs.get("stretch", False)
        self.figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        self.meta = {
            "driver": "GTiff",
            "dtype": np.dtype(self.__class__.__dtype),
            "nodata": self.__class__.__fill_value,
            "width": 0,
            "height": 0,
            "count": 1,
            "compress": "",
            "crs": riocrs.CRS.from_epsg(3857),
            "transform": Affine(1.0, 0.0, 0, 0.0, -1.0, 0),
        }
        self.meta.update((k, v) for k, v in kwargs.items() if k in self.meta)
        # ensure masked array
        if not isinstance(data, ma.MaskedArray):
            data = ma.array(
                data,
                dtype=self.meta["dtype"],
                fill_value=self.meta["nodata"],
            )
        # fix invalid entries
        data = ma.fix_invalid(data)
        # reshape if needed
        if len(data.shape) == 1:
            data = data.reshape((self.meta["height"], self.meta["width"]))
        # validate metadata
        assert data.shape == (
            self.meta["height"],
            self.meta["width"],
        ), "Wrong metadata !"
        # additional mask
        if (a_mask := kwargs.get("mask", None)) is not None:
            data[np.asarray(a_mask, dtype=bool)] = ma.masked
        # expand mask (needed)
        data[data.mask] = ma.masked
        self.data = data.copy()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.title} {self.shape} {self.resolution} "
            + f"masked:{self._mask.sum()} EPSG:{self.meta['crs'].to_epsg()}"
        )

    @property
    def _array(self):
        return ma.getdata(self.data)

    @property
    def _mask(self):
        return ma.getmask(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def resolution(self):
        return abs(self.meta["transform"].a), abs(self.meta["transform"].e)

    def __getitem__(self, mask):
        data = self.data.copy()
        if isinstance(mask, BoolGrid):
            data[~mask._array] = ma.masked
        else:
            data[~mask] = ma.masked
        return self.clone(data)

    def __setitem__(self, mask, value):
        value = np.array(value, np.dtype(self.__class__.__dtype))
        if value.ndim > 0:
            value = value.flatten()[0]
        if isinstance(mask, BoolGrid):
            mask2 = np.logical_and(~self._mask, mask._mask)
        else:
            mask2 = np.logical_and(~self._mask, mask)
        self.data[mask2] = value.item()

    def __add__(self, other):
        if issubclass(type(other), Grid):
            data = self.data + other.data
        else:
            data = self.data + other
        return self.clone(data)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if issubclass(type(other), Grid):
            self.data += other.data
        else:
            self.data += other
        return self

    def __sub__(self, other):
        if issubclass(type(other), Grid):
            data = self.data - other.data
        else:
            data = self.data - other
        return self.clone(data)

    def __rsub__(self, other):
        if issubclass(type(other), Grid):
            data = other.data - self.data
        else:
            data = other - self.data
        return self.clone(data)

    def __isub__(self, other):
        if issubclass(type(other), Grid):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def __mul__(self, other):
        if issubclass(type(other), Grid):
            data = self.data * other.data
        else:
            data = self.data * other
        return self.clone(data)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if issubclass(type(other), Grid):
            self.data *= other.data
        else:
            self.data *= other
        return self

    def __truediv__(self, other):
        if issubclass(type(other), Grid):
            data = self.data / other.data
        else:
            data = self.data / other
        return self.clone(data)

    def __rtruediv__(self, other):
        if issubclass(type(other), Grid):
            data = other.data / self.data
        else:
            data = other / self.data
        return self.clone(data)

    def __itruediv__(self, other):
        if issubclass(type(other), Grid):
            self.data /= other.data
        else:
            self.data /= other
        return self

    def __floordiv__(self, other):
        if issubclass(type(other), Grid):
            data = self.data // other.data
        else:
            data = self.data // other
        return self.clone(data)

    def __rfloordiv__(self, other):
        if issubclass(type(other), Grid):
            data = other.data // self.data
        else:
            data = other // self.data
        return self.clone(data)

    def __ifloordiv__(self, other):
        if issubclass(type(other), Grid):
            self.data //= other.data
        else:
            self.data //= other
        return self

    def __pow__(self, other):
        if issubclass(type(other), Grid):
            data = self.data**other.data
        else:
            data = self.data**other
        return self.clone(data)

    def __rpow__(self, other):
        if issubclass(type(other), Grid):
            data = other.data**self.data
        else:
            data = other**self.data
        return self.clone(data)

    def __lt__(self, other):
        if issubclass(type(other), Grid):
            data = self.data < other.data
        else:
            data = self.data < other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __le__(self, other):
        if issubclass(type(other), Grid):
            data = self.data <= other.data
        else:
            data = self.data <= other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __eq__(self, other):
        if issubclass(type(other), Grid):
            data = self.data == other.data
        else:
            data = self.data == other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __ne__(self, other):
        if issubclass(type(other), Grid):
            data = self.data != other.data
        else:
            data = self.data != other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __gt__(self, other):
        if issubclass(type(other), Grid):
            data = self.data > other.data
        else:
            data = self.data > other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __ge__(self, other):
        if issubclass(type(other), Grid):
            data = self.data >= other.data
        else:
            data = self.data >= other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def copy(self, **kwargs):
        dtype = kwargs.pop("dtype", self.meta["dtype"])
        return self.clone(self.data.astype(dtype).copy(), **kwargs)

    @property
    def filled(self):
        return self.data.filled(self.meta["nodata"])

    def clone(self, data, **kwargs):
        """Clone grid with new data

        Args:
            data (numpy.ma.MaskedArray): data as masked numpy array
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default keep original.
            figsize (tuple, optional): matplotlib figure size. Default keep original.

        """
        assert isinstance(data, np.ndarray), "Data must be the numpy.ndarray."
        typObj = kwargs.pop("astype", type(self))
        only_valid = kwargs.pop("only_valid", False)
        meta = kwargs.get("meta", self.meta).copy()
        meta["dtype"] = kwargs.get("dtype", self.meta["dtype"])
        meta["nodata"] = kwargs.get("nodata", self.meta["nodata"])
        if only_valid:
            full = meta["nodata"] * np.ones(self.shape, dtype=meta["dtype"])
            full[~self._mask] = np.array(data, dtype=meta["dtype"])
            return typObj(
                full,
                mask=self._mask,
                cmap=kwargs.get("cmap", self.cmap),
                # stretch=kwargs.get("stretch", self.stretch),
                title=kwargs.get("title", self.title),
                figsize=kwargs.get("figsize", self.figsize),
                **meta,
            )
        else:
            return typObj(
                data,
                mask=kwargs.get(
                    "mask", self._mask if data.shape == self._mask.shape else None
                ),
                cmap=kwargs.get("cmap", self.cmap),
                # stretch=kwargs.get("stretch", self.stretch),
                title=kwargs.get("title", self.title),
                figsize=kwargs.get("figsize", self.figsize),
                **meta,
            )

    def asfloat(self):
        """Return grid as FloatGrid"""
        return FloatGrid(
            ma.array(
                self.data,
                dtype=FloatGrid.__dtype,
                fill_value=FloatGrid.__fill_value,
            ),
            mask=self._mask if self.data.shape == self._mask.shape else None,
            cmap=self.cmap,
            title=self.title,
            figsize=self.figsize,
            **self.meta,
        )

    @contextmanager
    def asdataset(self):
        with MemoryFile() as memfile:
            # if np.any(self._mask):
            with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with memfile.open(**self.meta) as dst:
                    dst.write(self.data, 1)
            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Read dataset from georeferenced file

        Args:
            filename(str): Filename to read.
            band(int, optional): Band to read. Default 1.

        """
        band = kwargs.get("band", 1)
        with rio.open(filename) as src:
            data = src.read(band, masked=True)
            meta = src.meta
            # meta["dtype"] = cls.__dtype
            # meta["nodata"] = cls.__fill_value
        return cls(data, **kwargs, **meta)

    def write_tif(self, filename):
        """Write dataset to georeferenced file

        Args:
            filename (str): filename

        """
        if np.any(self._mask):
            with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rio.open(filename, "w", **self.meta) as dst:
                    dst.write(self.filled, 1)
                    dst.write_mask((~self._mask * 255).astype("uint8"))
        else:
            with rio.open(filename, "w", **self.meta) as dst:
                dst.write_band(1, self.filled)

    @property
    def _values(self):
        return self.data.compressed()

    def clip(self, r, h, c, w, **kwargs):
        res = self.data[r : r + h, c : c + w]
        # update metadata
        meta = self.meta.copy()
        meta["height"] = h
        meta["width"] = w
        t = self.meta["transform"]
        meta["transform"] = Affine(
            t.a, t.b, t.c + t.a * (c - 1), t.d, t.e, t.f + t.e * (r - 1)
        )
        return self.clone(res, meta=meta, **kwargs)

    def _kernel(self, **kwargs):
        win = kwargs.get("win", None)
        if win is None:
            r = kwargs.get("r", 5)
            L = np.arange(-r, r + 1)
            X, Y = np.meshgrid(L, L)
            win = np.array((X**2 + Y**2) <= r**2, dtype=int)
        if kwargs.get("exclude_centre", False):
            win[win.shape[0] // 2, win.shape[1] // 2] = 0
        n_sum = ndimage.convolve(self.data.filled(0), win, mode="constant")
        # do not count mask
        c_grid = np.ones(self.data.shape, dtype=int)
        c_grid[self._mask] = 0
        n_count = ndimage.convolve(c_grid, win, mode="constant")
        n_sum[self._mask] = np.nan
        return n_sum, n_count

    def generic_filter(self, func, size, **kwargs):
        d = self.filled
        res = ndimage.generic_filter(d, func, size=size)
        return self.clone(res, **kwargs)

    def correlation(self, filter, **kwargs):
        """Return the correlation between grid and filter"""
        assert (filter.shape[0] % 2, filter.shape[1] % 2) == (
            1,
            1,
        ), "Sizes of filter must be odd"
        d = self.filled
        pad = max(filter.shape) // 2
        dr, dc = d.shape
        padded = np.pad(d, pad, constant_values=np.nan)
        calc = np.nan_to_num(padded)
        res = np.zeros_like(d)
        counts = np.zeros_like(d, dtype=int)
        for (fr, fc), w in np.ndenumerate(filter):
            res += calc[fr : fr + dr, fc : fc + dc] * w
            counts += 1 - np.isnan(padded[fr : fr + dr, fc : fc + dc]).astype(int)

        # only full
        counts[counts < filter.size] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            res /= counts
        return self.clone(res, **kwargs)

    def similarity(self, filter, **kwargs):
        """Return the similarity between grid and filter"""
        assert (filter.shape[0] % 2, filter.shape[1] % 2) == (
            1,
            1,
        ), "Sizes of filter must be odd"
        d = self.filled
        pad = max(filter.shape) // 2
        dr, dc = d.shape
        padded = np.pad(d, pad, constant_values=np.nan)
        calc = np.nan_to_num(padded)
        res = np.zeros_like(d)
        counts = np.zeros_like(d, dtype=int)
        for (fr, fc), w in np.ndenumerate(filter):
            res += (w - calc[fr : fr + dr, fc : fc + dc]) ** 2
            counts += 1 - np.isnan(padded[fr : fr + dr, fc : fc + dc]).astype(int)

        # only full
        counts[counts < filter.size] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            res /= counts
        return self.clone(res, **kwargs)

    def aggregate(self, size, **kwargs):
        """Create aggregate dataset from squared blocks

        Args:
            size(int): Size of blocks.
            method (str): Aggregation function. One of `"maximum"`, `"mean"`,
                `"median"`, `"minimum"`, `"standard_deviation"`, `"variance"`.
                Default is `"mean"`.

        """
        # align blocks
        h, w = self.shape
        nh = max(h // size, 1)
        hpad = (max(h - nh * size, 0)) // 2
        nw = max(w // size, 1)
        wpad = (max(w - nw * size, 0)) // 2
        # create block
        blocks = []
        new_layout = []
        n = 0
        for r in range(nh):
            row = []
            new_row = []
            for c in range(nw):
                n += 1
                row.append(n * np.ones((size, size), dtype=int))
                new_row.append(n)
            blocks.append(row)
            new_layout.append(new_row)
        # create labels
        labels = np.zeros(self.shape, dtype=int)
        mxh, mxw = self.shape
        labels[hpad : hpad + nh * size, wpad : wpad + nw * size] = np.block(blocks)[
            hpad : min(hpad + nh * size, mxh), wpad : min(wpad + nw * size, mxw)
        ]
        labels[self._mask] = 0
        # remove mask and padding
        index = np.unique(labels)
        agg_index = np.array(new_layout)
        miss = np.setdiff1d(agg_index, index)
        agg_index[np.isin(agg_index, miss)] = 0
        # if only padding, skip it
        if 0 in np.setdiff1d(index, agg_index):
            index = np.delete(index, 0)
        # aggregate
        method = kwargs.get("method", "mean")
        match method:
            case "maximum":
                agg = ndimage.maximum(self._array, labels=labels, index=index)
            case "mean":
                agg = ndimage.mean(self._array, labels=labels, index=index)
            case "median":
                agg = ndimage.median(self._array, labels=labels, index=index)
            case "minimum":
                agg = ndimage.minimum(self._array, labels=labels, index=index)
            case "standard_deviation":
                agg = ndimage.standard_deviation(
                    self._array, labels=labels, index=index
                )
            case "variance":
                agg = ndimage.standard_deviation(
                    self._array, labels=labels, index=index
                )
            case _:
                raise ValueError(f"Method {method} is not available")
        # reconstruct
        _, bix = np.unique(agg_index, return_inverse=True)
        res = agg[bix]
        res[agg_index == 0] = np.nan
        # update metadata
        meta = self.meta.copy()
        meta["height"] = nh
        meta["width"] = nw
        t = self.meta["transform"]
        meta["transform"] = Affine(
            t.a * size, t.b, t.c + t.a * wpad, t.d, t.e * size, t.f + t.e * hpad
        )
        return self.clone(res, meta=meta, **kwargs)

    def sample(self, pts):
        """Returns array of values for sample points

        Args:
            pts(iterable): Pairs of x, y coordinates in the dataset’s reference system.

        """
        with self.asdataset() as src:
            res = np.array([*riosample.sample_gen(src, pts)]).flatten()
        return res

    def sample_line(self, p1, p2, n=10):
        """Returns array of values along line defined by endpoints p1 and p2

        Args:
            p1(tuple): Pair of x, y coordinates in the dataset’s reference system
            p2(tuple): Pair of x, y coordinates in the dataset’s reference system
            n(int, optional): NUmber of points sampled along line. Default `10`

        """
        pts = (
            (p1[0] + k * (p2[0] - p1[0]), p1[1] + k * (p2[1] - p1[1]))
            for k in np.linspace(0, 1, n)
        )
        return self.sample(pts)

    def show(self, **kwargs):
        """Show dataset

        Args:
            title (str, optional): Title of dataset. Default is `"DEM"`
            figsize (tuple, optional): matplotlib figure size.

        """
        ax = kwargs.pop("ax", None)
        title = kwargs.pop("title", self.title)
        transform = kwargs.pop("transform", self.meta["transform"])
        figsize = kwargs.pop("figsize", self.figsize)
        colorbar = kwargs.pop("colorbar", True)
        if "cmap" not in kwargs:
            kwargs["cmap"] = self.cmap
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:  # externally created fig, so no colorbar
            colorbar = False
        rioax = rioplot.show(
            self.data,
            ax=ax,
            title=title,
            transform=transform,
            **kwargs,
        )
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(rioax.images[0], cax=cax)  # type: ignore
        plt.show()


class BoolGrid(Grid):
    """A class to store boolean data.

    Args:
        data (numpy.ma.MaskedArray): data as masked numpy array
        cmap (str, optional): Colormap. Default is `"binary"`
        title (str, optional): Title of dataset. Default is `"Bool"`
        figsize (tuple, optional): matplotlib figure size.
        driver (str, optional): Default is `"GTiff"`
        dtype (str, optional): Default is `"float64"`
        compress (str, optional): Default is `"lzw"`
        crs (rasterio.crs.CRS, optional): Default is `CRS.from_epsg(3857)`
        transform (affine.Affine, optional): Default is `Affine(1.0, 0.0, 0, 0.0, -1.0, 0)`

    Attributes:
        unique_values (numpy.ndarray): array of unique values

    """

    __dtype = "bool"
    __fill_value = False

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.cmap = kwargs.get("cmap", "binary")
        self.title = kwargs.get("title", "Bool")

    def __and__(self, other):
        if isinstance(other, BoolGrid):
            data = np.logical_and(self.data, other.data)
        else:
            data = np.logical_and(self.data, other)
        return self.clone(data)

    def __or__(self, other):
        if isinstance(other, BoolGrid):
            data = np.logical_or(self.data, other.data)
        else:
            data = np.logical_or(self.data, other)
        return self.clone(data)

    def __neg__(self):
        return self.clone(np.logical_not(self.data))

    @property
    def count_true(self):
        return np.sum(self._values).item()

    @property
    def count_false(self):
        return np.sum(~self._values).item()

    def erosion(self, **kwargs):
        """Binary erosion of True regions

        Args:
            n(int, optional): Number of iterations. Default 1

        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_erosion(self.filled, iterations=n), **kwargs)

    def dilation(self, **kwargs):
        """Binary dilation of True regions

        Args:
            n(int, optional): Number of iterations. Default 1

        """
        n = kwargs.pop("n", 1)
        return self.clone(
            ndimage.binary_dilation(self.filled, iterations=n),
            **kwargs,
        )

    def closing(self, **kwargs):
        """Binary closing of True regions

        Args:
            n(int, optional): Number of iterations. Default 1

        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_closing(self.filled, iterations=n), **kwargs)

    def opening(self, **kwargs):
        """Binary opening of True regions

        Args:
            n(int, optional): Number of iterations. Default 1

        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_opening(self.filled, iterations=n), **kwargs)

    def fill_holes(self, **kwargs):
        """Binary fill of False holes in True regions"""
        return self.clone(ndimage.binary_fill_holes(self.filled), **kwargs)

    def remove_dots(self, **kwargs):
        """Remove True regions with size in pixels smaller or equal to given value

        Args:
            size(int, optional): Size in pixels of regions to remove. Default 1

        """
        labels, nb_labels = ndimage.label(self.filled)
        sizes = np.bincount(labels.ravel())
        mask_sizes = sizes > kwargs.pop("size", 1)
        mask_sizes[0] = 0
        return self.clone(mask_sizes[labels], **kwargs)

    def clear_boundary(self, **kwargs):
        """Remove True regions touching the boundary

        Args:
            use_mask(bool, optional): Also use mask boundary. Default True

        """
        if kwargs.pop("use_mask", True):
            # mask boundary
            bnd = ndimage.binary_dilation(self.data.mask) & ~self.data.mask
        else:
            bnd = np.zeros_like(self.filled)
        # data boundary
        bnd[:, [0, -1]] = True
        bnd[[0, -1], :] = True
        # apply mask
        bnd[self.data.mask] = False
        labels, nb_labels = ndimage.label(self.filled)
        touch = np.unique(labels[bnd])
        for lbl in touch:
            labels[labels == lbl] = 0
        return self.clone(labels > 1, **kwargs)

    def label(self, **kwargs):
        labeled_array, num_features = ndimage.label(self.data.filled(False))
        return self.clone(
            labeled_array, astype=IntGrid, dtype="int32", nodata=0, **kwargs
        )

    def normalized(self, **kwargs):
        """Returns normalized dataset to range (0,1)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            1 - self.data.astype(float),
            astype=FloatGrid,
            dtype=float,
            **kwargs,
        )


class IntGrid(Grid):
    """A class to store discrete data.

    Args:
        data (numpy.ma.MaskedArray): data as masked numpy array
        cmap (str, optional): Colormap. Default is `"viridis"`
        title (str, optional): Title of dataset. Default is `"IntGrid"`
        figsize (tuple, optional): matplotlib figure size.
        driver (str, optional): Default is `"GTiff"`
        dtype (str, optional): Default is `"float64"`
        nodata (float, optional): Default is `-9999.0`
        compress (str, optional): Default is `"lzw"`
        crs (rasterio.crs.CRS, optional): Default is `CRS.from_epsg(3857)`
        transform (affine.Affine, optional): Default is `Affine(1.0, 0.0, 0, 0.0, -1.0, 0)`

    Attributes:
        unique_values (numpy.ndarray): array of unique values

    """

    __dtype = "int"
    __fill_value = -9999

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.cmap = kwargs.get("cmap", "viridis")
        self.title = kwargs.get("title", "IntGrid")

    def __truediv__(self, other):
        if isinstance(other, DEMGrid):
            data = self.data // other.data.astype(int)
        else:
            data = self.data // int(other)
        return self.clone(data)

    def __rtruediv__(self, other):
        if isinstance(other, DEMGrid):
            data = other.data.astype(int) // self.data
        else:
            data = int(other) // self.data
        return self.clone(data)

    def __itruediv__(self, other):
        if isinstance(other, DEMGrid):
            self.data //= other.data.astype(int)
        else:
            self.data //= int(other)
        return self

    @classmethod
    def example(cls, test=False):
        """Get example data

        Returns:
            IntGrid: example int grid

        """
        datapath = importlib.resources.files("demtools.data")
        if test:
            fname = datapath.joinpath("int.tif")
        else:
            fname = datapath / "dem.tif"
        return cls.from_file(fname)

    @property
    def unique_values(self):
        return np.unique(self._values)

    def counts(self, **kwargs):
        """Count occurences of values in grid

        Args:
            plot(bool, optional): If True, show bar plot, otherwise returns (values, counts)

        """
        values, pos = np.unique(self._values, return_inverse=True)
        counts = np.bincount(pos)
        if kwargs.pop("plot", False):
            plt.bar(values, counts, **kwargs)
            plt.show()
        else:
            return values, counts

    def remove_dots(self, **kwargs):
        """Remove regions with size in pixels smaller or equal to given value

        Args:
            size(int, optional): Size in pixels of regions to remove. Default 1

        """
        values, counts = self.counts()
        counts[values[values == 0]] = 0
        mask_sizes = counts > kwargs.pop("size", 1)
        return self.clone(mask_sizes[self.data.filled()], **kwargs)

    def moving_average(self, **kwargs):
        """Returns moving window average

        Args:
            win(numpy.ndarray, optional): Numpy array of zeros and ones define window size and filter.
                Default is None
            r(int, optional): Define win halfsize if win is None. Shape of win is (2*r+1, 2*r+1)

        """
        n_sum, n_count = self._kernel(**kwargs)
        return self.clone(n_sum // n_count, **kwargs)

    def majority_filter(self, **kwargs):
        """Returns Majority filtered dataset

        Args:
            size(int): size gives the shape that is taken from the input array,
                at every element position, to define the input to the filter
                function. Default 3
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"M(...)"'.

        """

        def most_common(a):
            print(a, type(a))
            print("-------------------")
            return Counter(a).most_common(1)[0][0].item()

        size = kwargs.get("size", 3)
        filtered = ndimage.generic_filter(self.filled, most_common, size)
        kwargs["title"] = kwargs.get("title", f"M({self.title}, {size})")
        return self.clone(filtered, **kwargs)

    def normalized(self, **kwargs):
        """Returns normalized dataset to range (0,1)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            astype=FloatGrid,
            dtype=float,
            **kwargs,
        )

    def polygonize(self, file=None):
        source = self.clone(self.data.astype("int32"))
        if file is not None:
            features = []
            for shape, val in riofeatures.shapes(
                source.filled, transform=self.meta["transform"]
            ):
                if val != self.meta["nodata"]:
                    feat = geojson.Feature(
                        geometry=geojson.Polygon(shape["coordinates"]),
                        properties={"value": val},
                    )
                    features.append(feat)
            feature_collection = geojson.FeatureCollection(features)
            with open(file, "w") as f:
                geojson.dump(feature_collection, f)
        else:
            geoms = []
            for shape, val in riofeatures.shapes(
                source.filled, transform=self.meta["transform"]
            ):
                if val != self.meta["nodata"]:
                    geoms.append(shapelygeom.shape(shape))
            return geoms


class FloatGrid(Grid):
    """A class to store continuos data.

    Args:
        data (numpy.ma.MaskedArray): data as masked numpy array
        cmap (str, optional): Colormap. Default is `"viridis"`
        stretch (bool, optional): Stretch colormap to 2-98 percentil. Default is `True`
        title (str, optional): Title of dataset. Default is `"FloatGrid"`
        figsize (tuple, optional): matplotlib figure size.
        driver (str, optional): Default is `"GTiff"`
        dtype (str, optional): Default is `"float64"`
        nodata (float, optional): Default is `-9999.0`
        compress (str, optional): Default is `"lzw"`
        crs (rasterio.crs.CRS, optional): Default is `CRS.from_epsg(3857)`
        transform (affine.Affine, optional): Default is `Affine(1.0, 0.0, 0, 0.0, -1.0, 0)`

    Attributes:
        min (float): minimum of data
        max (float): maximum of data

    """

    __dtype = "float"
    __fill_value = np.nan

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.stretch = kwargs.get("stretch", True)
        self.cmap = kwargs.get("cmap", "viridis")
        self.title = kwargs.get("title", "FloatGrid")

    @property
    def min(self):
        return self._values.min().item()

    @property
    def max(self):
        return self._values.max().item()

    @property
    def mean(self):
        return self._values.mean().item()

    @cached_property
    def _dx(self):
        """Horizontal derivative dx as numpy array"""
        return derivx(self.filled, self.meta["transform"].a)

    @cached_property
    def _dy(self):
        """Horizontal derivative dy as numpy array"""
        return derivy(self.filled, self.meta["transform"].e)

    @cached_property
    def _dz(self):
        """Vertical derivative dz as numpy array"""
        dz = derivz(
            self.data.filled(
                self._values.mean()
            ),  # fix with implement fillna extrapolation
            self.meta["transform"].a,
            self.meta["transform"].e,
        )
        dz[np.isnan(self._dx) | np.isnan(self._dy)] = np.nan
        return dz

    @property
    def _tga(self):
        """Total gradient amplitude as numpy array"""
        return np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)

    def normalized(self, **kwargs):
        """Returns normalized dataset to range (0,1)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            astype=FloatGrid,
            **kwargs,
        )

    def inverted(self, **kwargs):
        """Returns inverted dataset

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        kwargs["title"] = kwargs.get("title", f"INV({self.title})")
        return self.clone(
            self.data.max() - self.data + self.data.min(), astype=FloatGrid, **kwargs
        )

    def moving_average(self, **kwargs):
        """Returns moving window average

        Args:
            win(numpy.ndarray, optional): Numpy array of zeros and ones define window size and filter.
                Default is None
            r(int, optional): Define win halfsize if win is None. Shape of win is (2*r+1, 2*r+1)

        """
        n_sum, n_count = self._kernel(**kwargs)
        return self.clone(n_sum / n_count, **kwargs)

    def digitize(self, **kwargs):
        """Return the IntGrid with indices of the bins to which each value belongs

        Args:
            bins (int or sequence of scalars or str, optional): If bins is an int,
                it defines the number of equal-width bins in the given range. If
                bins is a sequence, it defines the bin edges, including the
                rightmost edge, allowing for non-uniform bin widths. If string,
                see np.histogram_bin_edges. Default `auto`
            cmap (str, optional): Colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"DIG"'.
        """
        bins = np.histogram_bin_edges(
            self._values,
            bins=kwargs.get("bins", "auto"),
            range=kwargs.get("range", None),
        )
        data = np.digitize(self.data, bins)
        data[data == data.max()] = data.max() - 1
        kwargs["title"] = kwargs.get("title", f"DIG({self.title})")
        kwargs["cmap"] = kwargs.get("cmap", "viridis")
        return self.clone(data, mask=self._mask, astype=IntGrid, **kwargs)

    def resample(self, scale, resampling="average", **kwargs):
        """Returns resampled dataset

        Args:
            scale (float): Resampling scale. New resolution is scale multiple
                of original resolution
            resampling (str, optional): Resampling algorithm. One of 'nearest',
                'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average',
                'mode', and 'gauss'. Default 'average'.

        """
        algs = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
            "gauss": Resampling.gauss,
        }
        t = self.meta["transform"]
        transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
        height = int(self.meta["height"] / scale)
        width = int(self.meta["width"] / scale)
        meta = self.meta.copy()
        meta.update(transform=transform, height=height, width=width)
        with self.asdataset() as src:
            data = src.read(
                1,
                out_shape=(height, width),
                resampling=algs[resampling],
                masked=True,
            )
        return self.clone(data, meta=meta, **kwargs)

    def dx(self, **kwargs):
        """Returns horizontal derivative dx

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dx(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dx({self.title})")
        return self.clone(self._dx, astype=FloatGrid, **kwargs)

    def dy(self, **kwargs):
        """Returns horizontal derivative dy

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dy(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dy({self.title})")
        return self.clone(self._dy, astype=FloatGrid, **kwargs)

    def dz(self, **kwargs):
        """Returns vertical derivative dz

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dz(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dz({self.title})")
        return self.clone(self._dz, astype=FloatGrid, **kwargs)

    def upcont(self, h, **kwargs):
        """Returns upward continuation

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"up(...)"'.

        """
        up = upcontinue(
            self.data, self.meta["transform"].a, self.meta["transform"].e, h
        )
        up[np.isnan(self._dx) | np.isnan(self._dy)] = np.nan
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"up({self.title})")
        return self.clone(up, astype=FloatGrid, **kwargs)

    def thd(self, **kwargs):
        """Returns total horizontal derivative

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"THD(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"THD({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        return self.clone(thd, astype=FloatGrid, **kwargs)

    def tga(self, **kwargs):
        """Returns total gradient amplitude (also called the analytic signal)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"TGA(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"TGA({self.title})")
        tga = np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)
        return self.clone(tga, astype=FloatGrid, **kwargs)

    def theta(self, **kwargs):
        """Returns theta - total horizontal derivative divided by the analytical signal

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"Theta(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"Theta({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        tga = np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)
        return self.clone(thd / tga, astype=FloatGrid, **kwargs)

    def nthd(self, **kwargs):
        """Returns normalized total horizontal derivative

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NTHD(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"NTHD({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        return self.clone(
            np.real(np.arctan2(thd, np.absolute(self._dz))),
            astype=FloatGrid,
            **kwargs,
        )

    def tilt(self, **kwargs):
        """Returns tilt angle in radians

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"Tilt(...)"'.

        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"Tilt({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        return self.clone(np.arctan2(self._dz, thd), astype=FloatGrid, **kwargs)

    def gaussian_filter(self, **kwargs):
        """Returns Gaussian filtered dataset

        Args:
            sigma(float): Standard deviation for Gaussian kernel. Default 1
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"G(...)"'.

        """
        sigma = kwargs.get("sigma", 1)
        filtered = ndimage.gaussian_filter(self.data.filled(np.nan), sigma)
        kwargs["title"] = kwargs.get("title", f"G({self.title}, {sigma})")
        return self.clone(filtered, mask=np.isnan(filtered) | self._mask, **kwargs)

    def median_filter(self, **kwargs):
        """Returns median filtered dataset

        Args:
            size(int): A scalar or a list of length 2, giving the size
                of the median filter window. Default 3
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"M(...)"'.

        """
        size = kwargs.get("size", 3)
        filtered = ndimage.median_filter(self.data.filled(np.nan), size)
        kwargs["title"] = kwargs.get("title", f"M({self.title}, {size})")
        return self.clone(filtered, mask=np.isnan(filtered) | self._mask, **kwargs)

    def overlay(self, over, invert=False):
        """Create RGBimage of overlied datasets in HSV space

        Args:
            invert(bool): Default False

        """
        img_array = plt.get_cmap(self.cmap)(self.normalized().data)
        hsv = clr.rgb_to_hsv(img_array[:, :, :3])
        if invert:
            hsv[:, :, 2] = 1 - over.normalized().data
        else:
            hsv[:, :, 2] = over.normalized().data
        rgb = clr.hsv_to_rgb(hsv)
        return RGBimage(
            rioplot.reshape_as_raster(rgb),
            figsize=self.figsize,
            title=f"{over.title}/{self.title}",
            **self.meta,
        )

    def show(self, **kwargs):
        """Show dataset

        Args:
            contour(bool): Show data as contours. Default `False`
            contour_label_kws(dict): Default None
            cmap (str, optional): Colormap. Default is `"terrain"`
            stretch (bool, optional): Stretch colormap. Default is `False`
            title (str, optional): Title of dataset. Default is `"DEM"`
            figsize (tuple, optional): matplotlib figure size.

        """
        contour = kwargs.pop("contour", False)
        contour_label_kws = kwargs.pop("contour_label_kws", None)
        ax = kwargs.pop("ax", None)
        title = kwargs.pop("title", self.title)
        transform = kwargs.pop("transform", self.meta["transform"])
        stretch = kwargs.pop("stretch", self.stretch)
        figsize = kwargs.pop("figsize", self.figsize)
        colorbar = kwargs.pop("colorbar", True)
        if "cmap" not in kwargs:
            kwargs["cmap"] = self.cmap
        if stretch:
            kwargs["vmin"], kwargs["vmax"] = np.percentile(self._values, [2, 98])
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:  # externally created fig, so no colorbar
            colorbar = False
        rioax = rioplot.show(
            self.data,
            contour=contour,
            contour_label_kws=contour_label_kws,
            ax=ax,
            title=title,
            transform=transform,
            **kwargs,
        )
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if contour:
                fig.colorbar(rioax.collections[0], cax=cax)  # type: ignore
            else:
                fig.colorbar(rioax.images[0], cax=cax)  # type: ignore
        plt.show()


class DEMGrid(FloatGrid):
    """A class to store digital elevation model.

    Args:
        data (numpy.ma.MaskedArray): data as masked numpy array
        cmap (str, optional): Colormap. Default is `"terrain"`
        stretch (bool, optional): Stretch colormap to 2-98 percentil. Default is `False`
        title (str, optional): Title of dataset. Default is `"DEM"`
        figsize (tuple, optional): matplotlib figure size.
        driver (str, optional): Default is `"GTiff"`
        dtype (str, optional): Default is `"float64"`
        nodata (float, optional): Default is `-9999.0`
        compress (str, optional): Default is `"lzw"`
        crs (rasterio.crs.CRS, optional): Default is `CRS.from_epsg(3857)`
        transform (affine.Affine, optional): Default is `Affine(1.0, 0.0, 0, 0.0, -1.0, 0)`

    Attributes:
        min (float): minimum of data
        max (float): maximum of data

    Example:
        >>> d = DEMGrid.from_examples('smalldem')
        >>> d.show()

    """

    __dtype = "float"
    __fill_value = np.nan

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.stretch = kwargs.get("stretch", False)
        self.cmap = kwargs.get("cmap", "terrain")
        self.title = kwargs.get("title", "DEM")

    @classmethod
    def from_pysheds(cls, dem):
        data = ma.array(
            dem.data,
            mask=np.array(dem.data) == dem.nodata,
            fill_value=dem.nodata,
        )
        return cls(
            data,
            transform=dem.affine,
            height=dem.shape[0],
            width=dem.shape[1],
            crs=dem.crs,
            nodata=dem.nodata,
            dtype=dem.dtype,
        )

    @classmethod
    def example(cls, test=False):
        """Get example dem data

        Returns:
            DEMGrid: digital elevation model

        """
        datapath = importlib.resources.files("demtools") / "data"
        if test:
            fname = datapath / "testdem.tif"
        else:
            fname = datapath / "dem.tif"
        return cls.from_file(fname)

    def hillshade(self, **kwargs):
        """Returns hillshade

        Args:
            azdeg(float): The azimuth (0-360, degrees clockwise from North) of
                the light source. Default 150
            altdeg(float): The altitude (0-90, degrees up from horizontal) of
                the light source. Default 40
            vert_ezag(float): The amount to exaggerate the elevation values by
                when calculating illumination
        """
        vert_exag = kwargs.get("vert_exag", 1)
        azdeg = kwargs.get("azdeg", 150)
        altdeg = kwargs.get("altdeg", 40)
        ls = clr.LightSource(azdeg=azdeg, altdeg=altdeg)
        hs = ls.hillshade(
            self.data,
            vert_exag=vert_exag,
            dx=self.meta["transform"].a,
            dy=self.meta["transform"].e,
        )
        kwargs["cmap"] = kwargs.get("cmap", "gray")
        kwargs["title"] = kwargs.get(
            "title", f"Hillshade({self.title}, {vert_exag}, {azdeg}, {altdeg})"
        )
        return self.clone(hs, astype=FloatGrid, **kwargs)

    def shade(self, **kwargs):
        """Show combined colormapped data values with an illumination intensity map

        Args:
            azdeg(float): The azimuth (0-360, degrees clockwise from North) of
                the light source. Default 150
            altdeg(float): The altitude (0-90, degrees up from horizontal) of
                the light source. Default 40
            vert_ezag(float): The amount to exaggerate the elevation values by
                when calculating illumination
            blend_mode(str): The type of blending. Default `"overlay"`

        """
        azdeg = kwargs.get("azdeg", 150)
        altdeg = kwargs.get("altdeg", 40)
        ls = clr.LightSource(azdeg=azdeg, altdeg=altdeg)
        cmap = kwargs.get("cmap", self.cmap)
        rgb = ls.shade(
            self.data,
            cmap=plt.get_cmap(cmap),
            blend_mode=kwargs.get("blend_mode", "overlay"),
            vert_exag=kwargs.get("vert_exag", 1),
            dx=self.meta["transform"].a,
            dy=self.meta["transform"].e,
            fraction=kwargs.get("fraction", 1),
        )
        return RGBimage(
            rioplot.reshape_as_raster(rgb[:, :, :3]), figsize=self.figsize, **self.meta
        )

    def tpi(self, **kwargs):
        """Calculate Topographic Position Index (TPI)

        Args:
            win(numpy.array, optional): structural element. Default disk
            r(int, optional): radius of disk structural element. Default `5`
            cmap (str, optional): Colormap. Default `"seismic"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"TPI(...)"'.

        """
        n_sum, n_count = self._kernel(exclude_centre=True, **kwargs)
        # this is TPI (spot height – average neighbourhood height)
        tpi = self.filled - n_sum / n_count
        kwargs["cmap"] = kwargs.get("cmap", "seismic")
        kwargs["title"] = kwargs.get("title", f"TPI({self.title})")
        return self.clone(
            tpi, mask=np.isnan(tpi) | self._mask, astype=FloatGrid, **kwargs
        )

    def tpi_fft(self, **kwargs):
        """Calculate Topographic Position Index (TPI) using FFT

        Args:
            win(numpy.array, optional): operation window. Default disk
            r(int, optional): radius of disk structural element. Default `5`
            cmap (str, optional): Colormap. Default `"seismic"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"TPI(...)"'.

        """
        win = kwargs.get("win", None)
        if win is None:
            r = kwargs.get("r", 5)
            L = np.arange(-r, r + 1)
            X, Y = np.meshgrid(L, L)
            win = np.array((X**2 + Y**2) <= r**2, dtype=int)
        # ensure zero in centre
        win[win.shape[0] // 2, win.shape[1] // 2] = 0
        # count grid
        c_grid = np.ones(self.data.shape)
        c_grid[self._mask] = 0
        # do fast convolution
        n_sum = signal.convolve(self.data.filled(0), win, mode="same")
        n_count = signal.convolve(c_grid, win, mode="same")
        n_sum[self._mask] = np.nan
        # this is TPI (spot height – average neighbourhood height)
        tpi = self.filled - n_sum / n_count
        kwargs["cmap"] = kwargs.get("cmap", "seismic")
        kwargs["title"] = kwargs.get("title", f"TPI({self.title})")
        return self.clone(
            tpi, mask=np.isnan(tpi) | self._mask, astype=FloatGrid, **kwargs
        )

    def tpi_cube(self, **kwargs):
        n = kwargs.pop("n", 20)
        scale = kwargs.pop("scale", 3.5)
        steps = np.arange(0, (n - 1) * 2**scale + 1, 2**scale, dtype=int)
        for r in tqdm(steps, desc="Calculating TPI"):
            if r == 0:
                cube = [np.zeros_like(self.filled, dtype=float)[~self._mask]]
            else:
                with np.errstate(divide="ignore"):
                    L = np.arange(-r, r + 1)
                    X, Y = np.meshgrid(L, L)
                    win = np.array(1 / np.sqrt(X**2 + Y**2))
                    win[(X**2 + Y**2) > r**2] = 0
                tpi = self.tpi_fft(win=win).filled
                cube.append(tpi[~self._mask])
        if kwargs.get("use_diff", False):
            steps = steps[1:]
        return FeatureSet(
            np.array(cube).T, self._mask, self.meta, feature_coords=steps, **kwargs
        )

    def pits(self, size=3, **kwargs):
        win = np.ones((size, size))
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") > d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def pits2(self, size=3, **kwargs):
        win = np.ones((size, size))
        win[size // 2, size // 2] = 0
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") > d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def flats(self, size=3, **kwargs):
        win = np.ones((size, size))
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") == d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def flats2(self, size=3, **kwargs):
        win = np.ones((size, size))
        win[size // 2, size // 2] = 0
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") == d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def fill_pits(self, size=3, eps=0, **kwargs):
        win = np.ones((size, size))
        # win[size // 2, size // 2] = 0
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled.copy()
        fill = ndimage.minimum_filter(d, footprint=win, mode="mirror")
        d[fill > d] = fill[fill > d] + eps
        return self.clone(d, **kwargs)

    def sink_points(self, size):
        # locs = self.fill_pits(3).fill_pits(5).pits(7).fill_holes().opening().closing()
        locs = self.pits2(size)  # .fill_holes().opening().closing()
        labeled_array, num_features = ndimage.label(locs.filled)
        return ndimage.minimum_position(
            self.filled, labels=labeled_array, index=range(1, num_features + 1)
        )


class RGBimage:
    """A class to store RGB dataset.

    Args:
        data (numpy.array): data as 3d numpy array
        title (str, optional): Title of dataset. Default is `"RGB"`
        figsize (tuple, optional): matplotlib figure size.
        driver (str, optional): Default is `"GTiff"`
        dtype (str, optional): Default is `"float64"`
        nodata (float, optional): Default is `-9999.0`
        compress (str, optional): Default is `"lzw"`
        crs (rasterio.crs.CRS, optional): Default is `CRS.from_epsg(3857)`
        transform (affine.Affine, optional): Default is `Affine(1.0, 0.0, 0, 0.0, -1.0, 0)`

    """

    def __init__(self, rgb, **kwargs):
        self.data = rgb
        self.title = kwargs.pop("title", "RGB")
        self.figsize = kwargs.pop("figsize", plt.rcParams["figure.figsize"])
        self.meta = {
            "driver": "GTiff",
            "dtype": "float64",
            "nodata": -9999.0,
            "width": 0,
            "height": 0,
            "compress": "lzw",
            "crs": riocrs.CRS.from_epsg(3857),
            "transform": Affine(1.0, 0.0, 0, 0.0, -1.0, 0),
        }
        self.meta.update(kwargs)
        self.meta["count"] = 3
        assert rgb.shape == (
            3,
            self.meta["height"],
            self.meta["width"],
        ), "Wrong metadata !"

    def write_tif(self, filename):
        """Write dataset to file

        Args:
            filename (str): filename

        """
        with rio.open(filename, "w", **self.meta) as dst:
            dst.write(self.data)

    def show(self, **kwargs):
        """Show RGB dataset

        Args:
            title (str, optional): Title of dataset. Default is `"DEM"`
            figsize (tuple, optional): matplotlib figure size.

        """
        ax = kwargs.pop("ax", None)
        title = kwargs.pop("title", self.title)
        transform = kwargs.pop("transform", self.meta["transform"])
        figsize = kwargs.pop("figsize", self.figsize)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        rioplot.show(
            self.data,
            ax=ax,
            title=title,
            transform=transform,
            **kwargs,
        )
        plt.show()


class FeatureSet:
    """A class to store ML features.

    Args:
        data (numpy.array): data as 2d numpy array. Number of rows
            corresponds to valid raster cells.
        mask (numpy.array): boolean array mask
        meta (dict): rasterio metadata
    """

    def __init__(self, data, mask, meta, **kwargs):
        if kwargs.get("use_diff", False):
            print("Calculating differences... ")
            data = np.array([np.diff(r) for r in data])
        self.data = data
        self._mask = mask
        self.meta = meta.copy()
        self.meta["dtype"] = "int32"
        self.meta["nodata"] = -9999
        self.clusters = None
        self.centers = None
        self.labels = None
        self.feature_coords = kwargs.get("feature_coords", np.arange(data.shape[1]))
        if kwargs.get("do_cluster", True):
            self.cluster(**kwargs)

    def cluster(self, **kwargs):
        print("Calculating initial clusters... ")
        kmeans = KMeans(
            n_clusters=kwargs.get("n_kmeans", 128),
            random_state=kwargs.get("random_state", 42),
            init=kwargs.get("init", "k-means++"),
        )
        self.clusters = kmeans.fit_predict(self.data)
        self.centers = kmeans.cluster_centers_
        self.aggclusters(**kwargs)
        print("Done.")

    def dendrogram(self, **kwargs):
        Z = hierarchy.linkage(
            self.centers,
            metric=kwargs.get("metric", "euclidean"),
            method=kwargs.get("linkage", "ward"),
        )
        hierarchy.dendrogram(Z)
        plt.show()

    def aggclusters(self, **kwargs):
        hierarchical_cluster = AgglomerativeClustering(
            n_clusters=kwargs.get("n_clusters", 7),
            metric=kwargs.get("metric", "euclidean"),
            linkage=kwargs.get("linkage", "ward"),
        )
        agg = hierarchical_cluster.fit_predict(self.centers)
        self.labels = self.clusters.copy()
        for ix in range(len(agg)):
            self.labels[self.clusters == ix] = agg[ix]

    def labels_average(self):
        cids = np.unique(self.labels)
        return np.array([self.data[self.labels == c, :].mean(axis=0) for c in cids])

    def sort_label_integrals(self):
        cids = np.unique(self.labels)
        labels = self.labels.copy()
        six = np.argsort(np.trapezoid(self.labels_average()))
        for a, b in zip(cids, six):
            labels[self.labels == b] = a
        self.labels = labels

    def plot_averages(self, clusters=False):
        fig, ax = plt.subplots()
        if clusters:
            ax.plot(self.feature_coords, self.centers.T)
            for c, y in enumerate(self.centers[:, -1]):
                ax.text(self.feature_coords[-1], y, f"{c}", verticalalignment="center")
        else:
            cids = np.unique(self.labels)
            cave = self.labels_average()
            ax.plot(self.feature_coords, cave.T)
            for c, y in zip(cids, cave[:, -1]):
                ax.text(self.feature_coords[-1], y, f"{c}", verticalalignment="center")
        plt.show()

    def as_grid(self, clusters=False):
        res = IntGrid(
            self.meta["nodata"] * np.ones((self.meta["height"], self.meta["width"])),
            mask=self._mask,
            **self.meta,
        )
        if clusters:
            return res.clone(self.clusters, only_valid=True)
        else:
            return res.clone(self.labels, only_valid=True)
