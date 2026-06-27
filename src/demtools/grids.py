"""Raster grid classes built on ``numpy.ma.MaskedArray`` and ``rasterio``."""

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
import requests
import shapely.geometry as shapelygeom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from scipy import ndimage, signal
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from tqdm import tqdm

from demtools.mathlib import derivx, derivy, derivz, upcontinue


class Grid:
    """Base raster grid class.

    Wraps a :class:`numpy.ma.MaskedArray` with :mod:`rasterio` CRS and
    geotransform metadata. Provides arithmetic, comparison, I/O, and
    spatial operations.

    Subclasses MUST set ``_dtype`` and ``_fill_value`` class attributes.

    Args:
        data (numpy.ma.MaskedArray or numpy.ndarray): Input data.
        cmap (str, optional): Matplotlib colormap name. Default ``None``.
        title (str, optional): Dataset title. Default ``None``.
        stretch (bool, optional): Stretch colormap to 2–98 percentile.
            Default ``False``.
        figsize (tuple, optional): Figure size in inches.
            Default ``plt.rcParams["figure.figsize"]``.
        mask (numpy.ndarray, optional): Boolean mask; ``True`` entries are
            masked. Default ``None``.
        **meta: Additional rasterio metadata keys (``driver``, ``crs``,
            ``transform``, ``width``, ``height``, ``nodata``, ``compress``).
    """

    _dtype = None
    _fill_value = None

    def __init__(self, data, _copy=True, **kwargs):
        """Construct a grid from array data and rasterio metadata keywords."""
        self.cmap = kwargs.get("cmap", None)
        self.title = kwargs.get("title", None)
        self.stretch = kwargs.get("stretch", False)
        self.figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        self.meta = {
            "driver": "GTiff",
            "dtype": np.dtype(self.__class__._dtype),
            "nodata": self.__class__._fill_value,
            "width": None,
            "height": None,
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
        # set width an heigth if not set
        if (self.meta["height"] is None) and (self.meta["width"] is None):
            self.meta["height"], self.meta["width"] = data.shape
        # validate metadata
        if data.shape != (self.meta["height"], self.meta["width"]):
            raise ValueError(
                f"Data shape {data.shape} does not match metadata "
                f"({self.meta['height']}, {self.meta['width']})"
            )
        # additional mask
        if (a_mask := kwargs.get("mask", None)) is not None:
            data[np.asarray(a_mask, dtype=bool)] = ma.masked
        # expand mask (needed)
        data[data.mask] = ma.masked
        self.data = data.copy() if _copy else data

    def __repr__(self):
        """Return a printable summary of the grid."""
        return (
            f"{self.__class__.__name__} {self.title} {self.shape} {self.resolution} "
            + f"masked:{self._mask.sum()} EPSG:{self.meta['crs'].to_epsg()}"
        )

    @property
    def _array(self):
        """numpy.ndarray: Underlying data array (without mask)."""
        return ma.getdata(self.data)

    @property
    def _mask(self):
        """numpy.ndarray: Boolean mask array (``True`` = masked)."""
        m = ma.getmask(self.data)
        if m is ma.nomask:
            return np.zeros(self.shape, dtype=bool)
        return m

    @property
    def shape(self):
        """tuple: ``(height, width)`` of the grid in pixels."""
        return self.data.shape

    @property
    def resolution(self):
        """tuple: ``(x_resolution, y_resolution)`` in CRS units."""
        return abs(self.meta["transform"].a), abs(self.meta["transform"].e)

    def __getitem__(self, mask):
        """Mask grid with a boolean mask and return as a new grid.

        Args:
            mask (BoolGrid or numpy.ndarray): Boolean mask. Pixels
                corresponding to ``False`` entries are masked in the result.

        Returns:
            Grid: New grid with additional masking applied.
        """
        data = self.data.copy()
        if isinstance(mask, BoolGrid):
            data[~mask._array] = ma.masked
        else:
            data[~mask] = ma.masked
        return self.clone(data)

    def __setitem__(self, mask, value):
        """Set values at unmasked positions selected by a boolean mask.

        Args:
            mask (BoolGrid or numpy.ndarray): Boolean mask of positions to set.
            value (int or float): Value to assign at selected positions.
        """
        value = np.array(value, np.dtype(self.__class__._dtype))
        if value.ndim > 0:
            value = value.flatten()[0]
        if isinstance(mask, BoolGrid):
            mask2 = np.logical_and(~self._mask, mask._array)
        else:
            mask2 = np.logical_and(~self._mask, mask)
        self.data[mask2] = value.item()
        for attr in ("_dx", "_dy", "_dz"):
            self.__dict__.pop(attr, None)

    def __add__(self, other):
        """Element-wise addition.

        Args:
            other (Grid or scalar): Value to add.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data + other.data
        else:
            data = self.data + other
        return self.clone(data)

    def __radd__(self, other):
        """Element-wise reverse addition (``other + self``).

        Args:
            other (Grid or scalar): Left operand.

        Returns:
            Grid: Result as a new grid.
        """
        return self.__add__(other)

    def __iadd__(self, other):
        """In-place element-wise addition.

        Args:
            other (Grid or scalar): Value to add.

        Returns:
            Grid: This grid, modified in place.
        """
        if issubclass(type(other), Grid):
            self.data += other.data
        else:
            self.data += other
        return self

    def __sub__(self, other):
        """Element-wise subtraction.

        Args:
            other (Grid or scalar): Value to subtract.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data - other.data
        else:
            data = self.data - other
        return self.clone(data)

    def __rsub__(self, other):
        """Element-wise reverse subtraction (``other - self``).

        Args:
            other (Grid or scalar): Left operand.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = other.data - self.data
        else:
            data = other - self.data
        return self.clone(data)

    def __isub__(self, other):
        """In-place element-wise subtraction.

        Args:
            other (Grid or scalar): Value to subtract.

        Returns:
            Grid: This grid, modified in place.
        """
        if issubclass(type(other), Grid):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def __mul__(self, other):
        """Element-wise multiplication.

        Args:
            other (Grid or scalar): Value to multiply by.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data * other.data
        else:
            data = self.data * other
        return self.clone(data)

    def __rmul__(self, other):
        """Element-wise reverse multiplication (``other * self``).

        Args:
            other (Grid or scalar): Left operand.

        Returns:
            Grid: Result as a new grid.
        """
        return self.__mul__(other)

    def __imul__(self, other):
        """In-place element-wise multiplication.

        Args:
            other (Grid or scalar): Value to multiply by.

        Returns:
            Grid: This grid, modified in place.
        """
        if issubclass(type(other), Grid):
            self.data *= other.data
        else:
            self.data *= other
        return self

    def __truediv__(self, other):
        """Element-wise true division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data / other.data
        else:
            data = self.data / other
        return self.clone(data)

    def __rtruediv__(self, other):
        """Element-wise reverse true division (``other / self``).

        Args:
            other (Grid or scalar): Left operand (dividend).

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = other.data / self.data
        else:
            data = other / self.data
        return self.clone(data)

    def __itruediv__(self, other):
        """In-place element-wise true division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            Grid: This grid, modified in place.
        """
        if issubclass(type(other), Grid):
            self.data /= other.data
        else:
            self.data /= other
        return self

    def __floordiv__(self, other):
        """Element-wise floor division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data // other.data
        else:
            data = self.data // other
        return self.clone(data)

    def __rfloordiv__(self, other):
        """Element-wise reverse floor division (``other // self``).

        Args:
            other (Grid or scalar): Left operand (dividend).

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = other.data // self.data
        else:
            data = other // self.data
        return self.clone(data)

    def __ifloordiv__(self, other):
        """In-place element-wise floor division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            Grid: This grid, modified in place.
        """
        if issubclass(type(other), Grid):
            self.data //= other.data
        else:
            self.data //= other
        return self

    def __pow__(self, other):
        """Element-wise exponentiation.

        Args:
            other (Grid or scalar): Exponent.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = self.data**other.data
        else:
            data = self.data**other
        return self.clone(data)

    def __rpow__(self, other):
        """Element-wise reverse exponentiation (``other ** self``).

        Args:
            other (Grid or scalar): Base value.

        Returns:
            Grid: Result as a new grid.
        """
        if issubclass(type(other), Grid):
            data = other.data**self.data
        else:
            data = other**self.data
        return self.clone(data)

    def __lt__(self, other):
        """Element-wise less-than comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data < other.data
        else:
            data = self.data < other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __le__(self, other):
        """Element-wise less-than-or-equal comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data <= other.data
        else:
            data = self.data <= other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __eq__(self, other):
        """Element-wise equality comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data == other.data
        else:
            data = self.data == other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __ne__(self, other):
        """Element-wise not-equal comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data != other.data
        else:
            data = self.data != other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __gt__(self, other):
        """Element-wise greater-than comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data > other.data
        else:
            data = self.data > other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def __ge__(self, other):
        """Element-wise greater-than-or-equal comparison.

        Args:
            other (Grid or scalar): Value to compare against.

        Returns:
            BoolGrid: Boolean grid of comparison results.
        """
        if issubclass(type(other), Grid):
            data = self.data >= other.data
        else:
            data = self.data >= other
        return self.clone(data, cmap="binary", astype=BoolGrid, nodata=False)

    def copy(self, **kwargs):
        """Return a deep copy with optional dtype conversion.

        Args:
            dtype (numpy.dtype, optional): Override output dtype.
            **kwargs: Passed through to :meth:`clone`.

        Returns:
            Grid: Copy of the grid.
        """
        dtype = kwargs.pop("dtype", self.meta["dtype"])
        return self.clone(self.data.astype(dtype).copy(), **kwargs)

    @property
    def filled(self):
        """numpy.ndarray: Data with masked values replaced by the nodata value."""
        return self.data.filled(self.meta["nodata"])

    def clone(self, data, **kwargs):
        """Create a new grid of the same (or specified) type with new data.

        Args:
            data (numpy.ndarray or numpy.ma.MaskedArray): New data.
            astype (type, optional): Subclass to instantiate. Defaults to
                ``type(self)``.
            cmap (str, optional): Colormap. Defaults to original.
            stretch (bool, optional): Stretch colormap. Defaults to original.
            title (str, optional): Dataset title. Defaults to original.
            figsize (tuple, optional): Figure size. Defaults to original.
            mask (numpy.ndarray, optional): Boolean mask. Defaults to the
                current mask if shapes match.
            only_valid (bool, optional): If ``True``, rebuild the full 2-D
                array by placing *data* at unmasked positions.
                Default ``False``.
            dtype (numpy.dtype, optional): Override output dtype.
            nodata (scalar, optional): Override nodata value.
            meta (dict, optional): Full rasterio metadata dict override.

        Returns:
            Grid: New grid with *data*.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(data).__name__}")
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
                _copy=False,
                mask=self._mask,
                cmap=kwargs.get("cmap", self.cmap),
                stretch=kwargs.get("stretch", self.stretch),
                title=kwargs.get("title", self.title),
                figsize=kwargs.get("figsize", self.figsize),
                **meta,
            )
        else:
            return typObj(
                data,
                _copy=False,
                mask=kwargs.get(
                    "mask", self._mask if data.shape == self._mask.shape else None
                ),
                cmap=kwargs.get("cmap", self.cmap),
                stretch=kwargs.get("stretch", self.stretch),
                title=kwargs.get("title", self.title),
                figsize=kwargs.get("figsize", self.figsize),
                **meta,
            )

    def asbool(self):
        """Convert to :class:`BoolGrid`.

        Returns:
            BoolGrid: Boolean grid.
        """
        return BoolGrid(
            ma.array(
                self.data,
                dtype=BoolGrid._dtype,
                fill_value=BoolGrid._fill_value,
            ),
            mask=self._mask if self.data.shape == self._mask.shape else None,
            cmap=self.cmap,
            title=self.title,
            figsize=self.figsize,
            **self.meta,
        )

    def asint(self):
        """Convert to :class:`IntGrid`.

        Returns:
            IntGrid: Integer grid.
        """
        return IntGrid(
            ma.array(
                self.data,
                dtype=IntGrid._dtype,
                fill_value=IntGrid._fill_value,
            ),
            mask=self._mask if self.data.shape == self._mask.shape else None,
            cmap=self.cmap,
            title=self.title,
            figsize=self.figsize,
            **self.meta,
        )

    def asfloat(self):
        """Convert to :class:`FloatGrid`.

        Returns:
            FloatGrid: Float grid.
        """
        return FloatGrid(
            ma.array(
                self.data,
                dtype=FloatGrid._dtype,
                fill_value=FloatGrid._fill_value,
            ),
            mask=self._mask if self.data.shape == self._mask.shape else None,
            cmap=self.cmap,
            title=self.title,
            figsize=self.figsize,
            **self.meta,
        )

    @contextmanager
    def asdataset(self):
        """Context manager exposing the grid as a :class:`rasterio.DatasetReader`.

        Yields:
            rasterio.DatasetReader: In-memory read-only dataset.
        """
        with MemoryFile() as memfile:
            # if np.any(self._mask):
            with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with memfile.open(**self.meta) as dst:
                    dst.write(self.data, 1)
            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return

    def index(self, point, linear=False):
        """Return the grid row/col for a geographic coordinate.

        Args:
            point (tuple): ``(x, y)`` coordinate in the dataset's CRS.
            linear (bool, optional): If ``True``, return a linear (flattened)
                index. Default ``False``.

        Returns:
            tuple or int or None: ``(row, col)`` tuple (or linear index if
            ``linear=True``), or ``None`` if the point is masked or out
            of bounds.
        """
        with self.asdataset() as src:
            idx = src.index(*point)
        try:
            if not ma.is_masked(self.data[idx]):
                if linear:
                    full = np.zeros(self.shape, dtype=int)
                    full[~self._mask] = np.arange((~self._mask).sum())
                    return full[idx]
                else:
                    return idx
            else:
                return None
        except IndexError:
            return None

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Read a grid from a georeferenced raster file.

        Args:
            filename (str): Path to the raster file.
            band (int, optional): Band to read. Default ``1``.

        Returns:
            Grid: New grid instance with data and metadata from the file.
        """
        band = kwargs.get("band", 1)
        with rio.open(filename) as src:
            # Read as 3D (1, h, w) and slice to 2D to avoid the deprecated
            # MaskedArray shape-setter path triggered by rasterio on NumPy 2.5+.
            data = src.read([band], masked=True)[0]
            meta = src.meta
        return cls(data, **kwargs, **meta)

    def write_tif(self, filename):
        """Write grid to a GeoTIFF file.

        Writes an internal mask when the grid has masked pixels.

        Args:
            filename (str): Output file path.
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
        """numpy.ndarray: Flattened array of unmasked values only."""
        return self.data.compressed()

    def clip(self, r, h, c, w, **kwargs):
        """Extract a rectangular sub-region.

        Args:
            r (int): Start row (0-based).
            h (int): Number of rows.
            c (int): Start column (0-based).
            w (int): Number of columns.

        Returns:
            Grid: Clipped sub-region with updated transform.
        """
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
        """Compute the sum and count of pixels in a moving window.

        The window is a binary disk (or a custom array). Masked pixels are
        excluded from both the sum and the count.

        Args:
            win (numpy.ndarray, optional): Binary window array. Default
                ``None`` (creates a disk with radius *r*).
            r (int, optional): Disk radius in pixels. Default ``5``.
            exclude_centre (bool, optional): Zero out the centre pixel of
                the window. Default ``False``.

        Returns:
            tuple: ``(n_sum, n_count)`` where *n_sum* is the sum of pixel
            values in the window (float), NaN where masked, and *n_count* is
            the number of unmasked pixels in the window.
        """
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
        n_sum = n_sum.astype(float)
        n_sum[self._mask] = np.nan
        return n_sum, n_count

    def generic_filter(self, func, size, **kwargs):
        """Apply an arbitrary function over moving windows.

        Args:
            func (callable): Function applied to each window.
            size (int or tuple): Window size.

        Returns:
            Grid: Filtered grid.
        """
        d = self.filled
        res = ndimage.generic_filter(d, func, size=size)
        return self.clone(res, **kwargs)

    def correlation(self, filter, **kwargs):
        """Return the correlation between the grid and a filter kernel.

        Args:
            filter (numpy.ndarray): 2-D kernel of odd dimensions.

        Returns:
            Grid: Correlation map.
        """
        if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
            raise ValueError(f"Filter dimensions must be odd, got {filter.shape}")
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
        """Return the similarity between the grid and a filter kernel.

        Computed as the mean squared difference between kernel weights and
        grid values in each window.

        Args:
            filter (numpy.ndarray): 2-D kernel of odd dimensions.

        Returns:
            Grid: Similarity map.
        """
        if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
            raise ValueError(f"Filter dimensions must be odd, got {filter.shape}")
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
        """Create an aggregated (coarser) grid from squared blocks.

        Args:
            size (int): Side length of each square block in pixels.
            method (str, optional): Aggregation function. One of
                ``"maximum"``, ``"mean"``, ``"median"``, ``"minimum"``,
                ``"standard_deviation"``, ``"variance"``.
                Default ``"mean"``.

        Returns:
            Grid: Aggregated grid.
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
        with np.errstate(invalid="ignore"):
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
                    agg = ndimage.variance(self._array, labels=labels, index=index)
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
        """Return values at specified coordinates.

        Args:
            pts (iterable): Pairs of ``(x, y)`` coordinates in the
                dataset's CRS.

        Returns:
            numpy.ndarray: 1-D array of sampled values.
        """
        with self.asdataset() as src:
            res = np.array([*riosample.sample_gen(src, pts)]).flatten()
        return res

    def sample_line(self, p1, p2, n=10):
        """Return values along a line between two points.

        Args:
            p1 (tuple): ``(x, y)`` start coordinate.
            p2 (tuple): ``(x, y)`` end coordinate.
            n (int, optional): Number of sample points. Default ``10``.

        Returns:
            numpy.ndarray: 1-D array of sampled values along the line.
        """
        pts = (
            (p1[0] + k * (p2[0] - p1[0]), p1[1] + k * (p2[1] - p1[1]))
            for k in np.linspace(0, 1, n)
        )
        return self.sample(pts)

    def ginput(self, **kwargs):
        """Interactively select points on the grid plot.

        Args:
            n (int, optional): Number of points to select. Default ``1``.
            **kwargs: Passed through to :meth:`show` and
                :func:`matplotlib.pyplot.ginput`.

        Returns:
            list: Selected ``(x, y)`` coordinates.
        """
        n = kwargs.pop("n", 1)
        show_clicks = kwargs.pop("show_clicks", 1)
        figsize = kwargs.get("figsize", self.figsize)
        fig, ax = plt.subplots(figsize=figsize)
        kwargs["ax"] = ax
        kwargs["show"] = False
        self.show(**kwargs)
        return plt.ginput(n, show_clicks=show_clicks)

    def show(self, **kwargs):
        """Display the grid using :func:`rasterio.plot.show`.

        Args:
            title (str, optional): Plot title. Defaults to ``self.title``.
            figsize (tuple, optional): Figure size in inches.
            transform (affine.Affine, optional): Geotransform for axis labels.
            colorbar (bool, optional): Show colour bar. Default ``True``.
            adjust (bool, optional): Automatically adjust layout.
                Default ``False``.
            ax (matplotlib.axes.Axes, optional): Existing axes. When provided,
                the colour bar is suppressed.
            show (bool, optional): Call ``plt.show()``. Default ``True``.
            cmap (str, optional): Matplotlib colormap.

        Returns:
            None
        """
        ax = kwargs.pop("ax", None)
        show = kwargs.pop("show", True)
        title = kwargs.pop("title", self.title)
        transform = kwargs.pop("transform", self.meta["transform"])
        figsize = kwargs.pop("figsize", self.figsize)
        colorbar = kwargs.pop("colorbar", True)
        adjust = kwargs.pop("adjust", False)
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
            adjust=adjust,
            **kwargs,
        )
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(rioax.images[0], cax=cax)  # type: ignore
        if show:
            plt.show()


class BoolGrid(Grid):
    """Grid class specialised for boolean (binary) data.

    Supports logical operations (``&``, ``|``, ``~``), morphological
    operators (erosion, dilation, closing, opening), and connected-component
    labelling.

    Args:
        data (numpy.ma.MaskedArray or numpy.ndarray): Input boolean data.
        cmap (str, optional): Colormap. Default ``"binary"``.
        title (str, optional): Dataset title. Default ``"Bool"``.
        **kwargs: Additional arguments passed to :class:`Grid`.

    """

    _dtype = "bool"
    _fill_value = False

    def __init__(self, data, **kwargs):
        """Construct a BoolGrid with default binary colormap and title."""
        super().__init__(data, **kwargs)
        self.cmap = kwargs.get("cmap", "binary")
        self.title = kwargs.get("title", "Bool")

    def __and__(self, other):
        """Element-wise logical AND.

        Args:
            other (BoolGrid or scalar): Operand.

        Returns:
            BoolGrid: Result of logical AND.
        """
        if isinstance(other, BoolGrid):
            data = np.logical_and(self.data, other.data)
        else:
            data = np.logical_and(self.data, other)
        return self.clone(data)

    def __or__(self, other):
        """Element-wise logical OR.

        Args:
            other (BoolGrid or scalar): Operand.

        Returns:
            BoolGrid: Result of logical OR.
        """
        if isinstance(other, BoolGrid):
            data = np.logical_or(self.data, other.data)
        else:
            data = np.logical_or(self.data, other)
        return self.clone(data)

    def __neg__(self):
        """Element-wise logical NOT (inversion).

        Returns:
            BoolGrid: Inverted boolean grid.
        """
        return self.clone(np.logical_not(self.data))

    @property
    def count_true(self):
        """int: Number of unmasked ``True`` values."""
        return np.sum(self._values).item()

    @property
    def count_false(self):
        """int: Number of unmasked ``False`` values."""
        return len(self._values) - self.count_true

    def erosion(self, **kwargs):
        """Binary erosion of ``True`` regions.

        Args:
            n (int, optional): Number of iterations. Default ``1``.

        Returns:
            BoolGrid: Eroded grid.
        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_erosion(self.filled, iterations=n), **kwargs)

    def dilation(self, **kwargs):
        """Binary dilation of ``True`` regions.

        Args:
            n (int, optional): Number of iterations. Default ``1``.

        Returns:
            BoolGrid: Dilated grid.
        """
        n = kwargs.pop("n", 1)
        return self.clone(
            ndimage.binary_dilation(self.filled, iterations=n),
            **kwargs,
        )

    def closing(self, **kwargs):
        """Binary closing (dilation then erosion) of ``True`` regions.

        Args:
            n (int, optional): Number of iterations. Default ``1``.

        Returns:
            BoolGrid: Closed grid.
        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_closing(self.filled, iterations=n), **kwargs)

    def opening(self, **kwargs):
        """Binary opening (erosion then dilation) of ``True`` regions.

        Args:
            n (int, optional): Number of iterations. Default ``1``.

        Returns:
            BoolGrid: Opened grid.
        """
        n = kwargs.pop("n", 1)
        return self.clone(ndimage.binary_opening(self.filled, iterations=n), **kwargs)

    def fill_holes(self, **kwargs):
        """Fill holes (``False`` regions completely surrounded by ``True``).

        Args:
            **kwargs: Display options passed to :meth:`clone`.

        Returns:
            BoolGrid: Grid with holes filled.
        """
        return self.clone(ndimage.binary_fill_holes(self.filled), **kwargs)

    def remove_dots(self, **kwargs):
        """Remove ``True`` regions smaller than or equal to a given size.

        Args:
            size (int, optional): Maximum pixel count of regions to remove.
                Default ``1``.

        Returns:
            BoolGrid: Grid with small regions removed.
        """
        labels, nb_labels = ndimage.label(self.filled)
        sizes = np.bincount(labels.ravel())
        mask_sizes = sizes > kwargs.pop("size", 1)
        mask_sizes[0] = 0
        return self.clone(mask_sizes[labels], **kwargs)

    def clear_boundary(self, **kwargs):
        """Remove ``True`` regions that touch the grid boundary.

        Args:
            use_mask (bool, optional): Also consider the data mask as a
                boundary. Default ``True``.

        Returns:
            BoolGrid: Grid with boundary-touching regions removed.
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
        """Label connected components of ``True`` regions.

        Args:
            **kwargs: Display options passed to :meth:`clone`.

        Returns:
            IntGrid: Labelled grid with component indices (1-based).
        """
        labeled_array, num_features = ndimage.label(self.data.filled(False))
        return self.clone(
            labeled_array, astype=IntGrid, dtype="int32", nodata=0, **kwargs
        )

    def normalized(self, **kwargs):
        """Return inverted-normalized grid to range ``[0, 1]``.

        For boolean grids this produces ``1 - data`` as a :class:`FloatGrid`.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"NORM(…)"``.

        Returns:
            FloatGrid: Normalised float grid.
        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            1 - self.data.astype(float),
            astype=FloatGrid,
            dtype=float,
            **kwargs,
        )


class IntGrid(Grid):
    """Grid class specialised for discrete (integer) data.

    Supports integer division semantics (``//``) and majority / moving-average
    filters.

    Args:
        data (numpy.ma.MaskedArray or numpy.ndarray): Input integer data.
        cmap (str, optional): Colormap. Default ``"viridis"``.
        title (str, optional): Dataset title. Default ``"IntGrid"``.
        **kwargs: Additional arguments passed to :class:`Grid`.

    """

    _dtype = "int"
    _fill_value = -9999

    def __init__(self, data, **kwargs):
        """Construct an IntGrid with default viridis colormap and title."""
        super().__init__(data, **kwargs)
        self.cmap = kwargs.get("cmap", "viridis")
        self.title = kwargs.get("title", "IntGrid")

    def __truediv__(self, other):
        """Element-wise integer (floor) division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            IntGrid: Result as integer grid.
        """
        if isinstance(other, Grid):
            data = self.data // other.data.astype(int)
        else:
            data = self.data // int(other)
        return self.clone(data)

    def __rtruediv__(self, other):
        """Element-wise reverse integer division (``other // self``).

        Args:
            other (Grid or scalar): Left operand (dividend).

        Returns:
            IntGrid: Result as integer grid.
        """
        if isinstance(other, Grid):
            data = other.data.astype(int) // self.data
        else:
            data = int(other) // self.data
        return self.clone(data)

    def __itruediv__(self, other):
        """In-place element-wise integer (floor) division.

        Args:
            other (Grid or scalar): Divisor.

        Returns:
            IntGrid: This grid, modified in place.
        """
        if isinstance(other, Grid):
            self.data //= other.data.astype(int)
        else:
            self.data //= int(other)
        return self

    @classmethod
    def example(cls, test=False):
        """Load an example :class:`IntGrid` from bundled test data.

        Args:
            test (bool, optional): Use ``int.tif`` (small) when ``True``,
                ``dem.tif`` otherwise. Default ``False``.

        Returns:
            IntGrid: Example integer grid.
        """
        datapath = importlib.resources.files("demtools.data")
        if test:
            fname = datapath.joinpath("int.tif")
        else:
            fname = datapath / "dem.tif"
        return cls.from_file(fname)

    @property
    def unique_values(self):
        """numpy.ndarray: Sorted unique unmasked values."""
        return np.unique(self._values)

    def counts(self, **kwargs):
        """Count occurrences of each unique value in the grid.

        Args:
            plot (bool, optional): If ``True``, show a bar plot; otherwise
                return ``(values, counts)``. Default ``False``.
            **kwargs: Passed through to :func:`matplotlib.pyplot.bar` when
                *plot* is ``True``.

        Returns:
            tuple or None: ``(values, counts)`` when *plot* is ``False``.
        """
        values, pos = np.unique(self._values, return_inverse=True)
        counts = np.bincount(pos)
        if kwargs.pop("plot", False):
            plt.bar(values, counts, **kwargs)
            plt.show()
        else:
            return values, counts

    def remove_dots(self, **kwargs):
        """Remove regions of a given value smaller than a size threshold.

        For each distinct value, connected components smaller than *size*
        pixels are replaced with the nodata value.

        Args:
            size (int, optional): Maximum pixel count of regions to remove.
                Default ``1``.

        Returns:
            IntGrid: Grid with small regions removed.
        """
        size = kwargs.pop("size", 1)
        filled = self.filled
        nodata = self.meta["nodata"]
        result = filled.copy()
        for val in np.unique(filled):
            if val == nodata:
                continue
            val_mask = filled == val
            labels, _ = ndimage.label(val_mask)
            region_sizes = np.bincount(labels.ravel())
            small = np.where(region_sizes <= size)[0]
            small = small[small != 0]
            for rid in small:
                result[labels == rid] = nodata
        return self.clone(result, **kwargs)

    def moving_average(self, **kwargs):
        """Return moving-window average using integer division.

        Args:
            win (numpy.ndarray, optional): Binary window array. Default
                ``None`` (creates a disk with radius *r*).
            r (int, optional): Disk radius in pixels. Default ``5``.

        Returns:
            IntGrid: Moving-average grid (integer).
        """
        n_sum, n_count = self._kernel(**kwargs)
        return self.clone(n_sum // n_count, **kwargs)

    def majority_filter(self, **kwargs):
        """Apply a majority (most-common-value) filter.

        Args:
            size (int, optional): Window size. Default ``3``.
            cmap (str, optional): Colormap.
            stretch (bool, optional): Stretch colormap.
            title (str, optional): Dataset title. Default ``"M(…)"``.

        Returns:
            IntGrid: Majority-filtered grid.
        """

        def most_common(a):
            return Counter(a).most_common(1)[0][0].item()

        size = kwargs.get("size", 3)
        filtered = ndimage.generic_filter(self.filled, most_common, size)
        kwargs["title"] = kwargs.get("title", f"M({self.title}, {size})")
        return self.clone(filtered, **kwargs)

    def normalized(self, **kwargs):
        """Return min-max normalised grid to range ``[0, 1]``.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"NORM(…)"``.

        Returns:
            FloatGrid: Normalised float grid.
        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            astype=FloatGrid,
            dtype=float,
            **kwargs,
        )

    def polygonize(self, file=None):
        """Vectorise connected regions of equal value.

        Args:
            file (str, optional): Path to write GeoJSON output. If ``None``,
                returns a list of :class:`shapely.geometry.shape` objects.

        Returns:
            list or None: List of Shapely geometries when *file* is ``None``.
        """
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
    """Grid class specialised for continuous (floating-point) data.

    Provides potential-field derivatives (:meth:`dx`, :meth:`dy`, :meth:`dz`,
    :meth:`thd`, :meth:`tga`, :meth:`theta`, :meth:`nthd`, :meth:`tilt`),
    upward continuation, resampling, and various filters.

    Args:
        data (numpy.ma.MaskedArray or numpy.ndarray): Input float data.
        cmap (str, optional): Colormap. Default ``"viridis"``.
        stretch (bool, optional): Stretch colormap to 2–98 percentile.
            Default ``True``.
        title (str, optional): Dataset title. Default ``"FloatGrid"``.
        **kwargs: Additional arguments passed to :class:`Grid`.

    """

    _dtype = "float"
    _fill_value = np.nan

    def __init__(self, data, **kwargs):
        """Construct a FloatGrid with stretch enabled and viridis colormap."""
        super().__init__(data, **kwargs)
        self.stretch = kwargs.get("stretch", True)
        self.cmap = kwargs.get("cmap", "viridis")
        self.title = kwargs.get("title", "FloatGrid")

    @property
    def min(self):
        """float: Minimum of unmasked values."""
        return self._values.min().item()

    @property
    def max(self):
        """float: Maximum of unmasked values."""
        return self._values.max().item()

    @property
    def mean(self):
        """float: Mean of unmasked values."""
        return self._values.mean().item()

    @cached_property
    def _dx(self):
        """numpy.ndarray: Horizontal derivative in the x-direction."""
        return derivx(self.filled, self.meta["transform"].a)

    @cached_property
    def _dy(self):
        """numpy.ndarray: Horizontal derivative in the y-direction."""
        return derivy(self.filled, self.meta["transform"].e)

    @cached_property
    def _dz(self):
        """numpy.ndarray: Vertical derivative computed via FFT."""
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
        """numpy.ndarray: Total gradient amplitude
        :math:`\\sqrt{dx^2 + dy^2 + dz^2}`."""
        return np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)

    def normalized(self, **kwargs):
        """Return min-max normalised grid to range ``[0, 1]``.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"NORM(…)"``.

        Returns:
            FloatGrid: Normalised float grid.
        """
        kwargs["title"] = kwargs.get("title", f"NORM({self.title})")
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            astype=FloatGrid,
            **kwargs,
        )

    def inverted(self, **kwargs):
        """Return inverted dataset (``max - data + min``).

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"INV(…)"``.

        Returns:
            FloatGrid: Inverted float grid.
        """
        kwargs["title"] = kwargs.get("title", f"INV({self.title})")
        return self.clone(
            self.data.max() - self.data + self.data.min(), astype=FloatGrid, **kwargs
        )

    def moving_average(self, **kwargs):
        """Return moving-window average.

        Args:
            win (numpy.ndarray, optional): Binary window array. Default
                ``None`` (creates a disk with radius *r*).
            r (int, optional): Disk radius in pixels. Default ``5``.

        Returns:
            FloatGrid: Moving-average grid.
        """
        n_sum, n_count = self._kernel(**kwargs)
        return self.clone(n_sum / n_count, **kwargs)

    def digitize(self, **kwargs):
        """Bin values into equally spaced bins and return bin indices.

        Args:
            bins (int or sequence or str, optional): Bin specification.
                An integer defines the number of equal-width bins; a sequence
                defines bin edges; a string selects a
                :func:`numpy.histogram_bin_edges` method.
                Default ``"auto"``.
            cmap (str, optional): Colormap.
            title (str, optional): Dataset title. Default ``"DIG(…)"``.

        Returns:
            IntGrid: Grid of bin indices.
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
        """Resample the grid to a coarser or finer resolution.

        Args:
            scale (float): Rescaling factor. Values > 1 coarsen the grid;
                values < 1 refine it.
            resampling (str, optional): Resampling algorithm. One of
                ``"nearest"``, ``"bilinear"``, ``"cubic"``,
                ``"cubic_spline"``, ``"lanczos"``, ``"average"``,
                ``"mode"``, ``"gauss"``. Default ``"average"``.

        Returns:
            FloatGrid: Resampled grid.
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
                [1],
                out_shape=(1, height, width),
                resampling=algs[resampling],
                masked=True,
            )[0]
        return self.clone(data, meta=meta, **kwargs)

    def dx(self, **kwargs):
        """Return the first horizontal derivative in the x-direction.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"dx(…)"``.

        Returns:
            FloatGrid: Derivative grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dx({self.title})")
        return self.clone(self._dx, astype=FloatGrid, **kwargs)

    def dy(self, **kwargs):
        """Return the first horizontal derivative in the y-direction.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"dy(…)"``.

        Returns:
            FloatGrid: Derivative grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dy({self.title})")
        return self.clone(self._dy, astype=FloatGrid, **kwargs)

    def dz(self, **kwargs):
        """Return the first vertical derivative computed via FFT.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"dz(…)"``.

        Returns:
            FloatGrid: Derivative grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"dz({self.title})")
        return self.clone(self._dz, astype=FloatGrid, **kwargs)

    def upcont(self, h, **kwargs):
        """Upward-continue the potential-field grid to a given height.

        Args:
            h (float): Height (in CRS units) to continue the field to.
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"up(…)"``.

        Returns:
            FloatGrid: Upward-continued grid.
        """
        up = upcontinue(
            self.filled, self.meta["transform"].a, self.meta["transform"].e, h
        )
        up[np.isnan(self._dx) | np.isnan(self._dy)] = np.nan
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"up({self.title})")
        return self.clone(up, astype=FloatGrid, **kwargs)

    def thd(self, **kwargs):
        """Return the total horizontal derivative
        :math:`\\sqrt{dx^2 + dy^2}`.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"THD(…)"``.

        Returns:
            FloatGrid: THD grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"THD({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        return self.clone(thd, astype=FloatGrid, **kwargs)

    def tga(self, **kwargs):
        """Return the total gradient amplitude (analytic signal)
        :math:`\\sqrt{dx^2 + dy^2 + dz^2}`.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"TGA(…)"``.

        Returns:
            FloatGrid: TGA grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"TGA({self.title})")
        tga = np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)
        return self.clone(tga, astype=FloatGrid, **kwargs)

    def theta(self, **kwargs):
        r"""Return :math:`\\theta = \\frac{THD}{\\text{TGA}}`.

        The ratio of the total horizontal derivative to the total gradient
        amplitude.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"Theta(…)"``.

        Returns:
            FloatGrid: Theta grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"Theta({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        tga = np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)
        return self.clone(thd / tga, astype=FloatGrid, **kwargs)

    def nthd(self, **kwargs):
        """Return the normalised total horizontal derivative (NTHD).

        :math:`\\text{NTHD} = \\arctan\\left(
        \\frac{THD}{|dz|}\\right)`.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"NTHD(…)"``.

        Returns:
            FloatGrid: NTHD grid.
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
        """Return the tilt angle :math:`\\arctan(dz / THD)` in radians.

        Args:
            cmap (str, optional): Colormap. Default ``"bone_r"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"Tilt(…)"``.

        Returns:
            FloatGrid: Tilt-angle grid.
        """
        kwargs["cmap"] = kwargs.get("cmap", "bone_r")
        kwargs["title"] = kwargs.get("title", f"Tilt({self.title})")
        thd = np.sqrt(self._dx**2 + self._dy**2)
        return self.clone(np.arctan2(self._dz, thd), astype=FloatGrid, **kwargs)

    def gaussian_filter(self, **kwargs):
        """Apply a Gaussian filter.

        Args:
            sigma (float, optional): Standard deviation in pixels.
                Default ``1``.
            cmap (str, optional): Colormap.
            stretch (bool, optional): Stretch colormap.
            title (str, optional): Dataset title. Default ``"G(…)"``.

        Returns:
            FloatGrid: Filtered grid.
        """
        sigma = kwargs.get("sigma", 1)
        filtered = ndimage.gaussian_filter(self.data.filled(np.nan), sigma)
        kwargs["title"] = kwargs.get("title", f"G({self.title}, {sigma})")
        return self.clone(filtered, mask=np.isnan(filtered) | self._mask, **kwargs)

    def median_filter(self, **kwargs):
        """Apply a median filter.

        Args:
            size (int or list, optional): Window size (scalar or ``[h, w]``).
                Default ``3``.
            cmap (str, optional): Colormap.
            stretch (bool, optional): Stretch colormap.
            title (str, optional): Dataset title. Default ``"M(…)"``.

        Returns:
            FloatGrid: Filtered grid.
        """
        size = kwargs.get("size", 3)
        filtered = ndimage.median_filter(self.data.filled(np.nan), size)
        kwargs["title"] = kwargs.get("title", f"M({self.title}, {size})")
        return self.clone(filtered, mask=np.isnan(filtered) | self._mask, **kwargs)

    def overlay(self, over, invert=False):
        """Create an :class:`RGBimage` by overlaying two datasets in HSV space.

        The current grid is used for hue/saturation and *over* for value
        (brightness).

        Args:
            over (Grid): Grid providing the value (brightness) channel.
            invert (bool, optional): Invert the brightness channel.
                Default ``False``.

        Returns:
            RGBimage: Overlaid RGB image.
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
        """Display the grid with optional contour overlay and stretch.

        Args:
            contour (bool, optional): Overlay contour lines.
                Default ``False``.
            contour_label_kws (dict, optional): Keyword args for contour
                labels.
            stretch (bool, optional): Stretch displayed range to 2–98
                percentile. Defaults to ``self.stretch``.
            title (str, optional): Plot title. Defaults to ``self.title``.
            figsize (tuple, optional): Figure size in inches.
            transform (affine.Affine, optional): Geotransform.
            colorbar (bool, optional): Show colour bar. Default ``True``.
            adjust (bool, optional): Auto-adjust layout. Default ``False``.
            ax (matplotlib.axes.Axes, optional): Existing axes.
            show (bool, optional): Call ``plt.show()``. Default ``True``.
            cmap (str, optional): Colormap.
        """
        contour = kwargs.pop("contour", False)
        contour_label_kws = kwargs.pop("contour_label_kws", None)
        ax = kwargs.pop("ax", None)
        show = kwargs.pop("show", True)
        title = kwargs.pop("title", self.title)
        transform = kwargs.pop("transform", self.meta["transform"])
        stretch = kwargs.pop("stretch", self.stretch)
        figsize = kwargs.pop("figsize", self.figsize)
        colorbar = kwargs.pop("colorbar", True)
        adjust = kwargs.pop("adjust", False)
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
            adjust=adjust,
            **kwargs,
        )
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if contour:
                fig.colorbar(rioax.collections[0], cax=cax)  # type: ignore
            else:
                fig.colorbar(rioax.images[0], cax=cax)  # type: ignore
        if show:
            plt.show()


class DEMGrid(FloatGrid):
    """Grid class specialised for digital elevation models.

    Inherits all :class:`FloatGrid` functionality and adds
    terrain-specific methods: hillshade, shaded relief, Topographic
    Position Index (TPI), pit detection, and sink filling.

    Args:
        data (numpy.ma.MaskedArray or numpy.ndarray): Input elevation data.
        cmap (str, optional): Colormap. Default ``"terrain"``.
        stretch (bool, optional): Stretch colormap to 2–98 percentile.
            Default ``False``.
        title (str, optional): Dataset title. Default ``"DEM"``.
        **kwargs: Additional arguments passed to :class:`FloatGrid`.

    Example:
        >>> dem = DEMGrid.example(test=True)
        >>> dem.show()
    """

    _dtype = "float"
    _fill_value = np.nan

    def __init__(self, data, **kwargs):
        """Construct a DEMGrid with stretch disabled and terrain colormap."""
        super().__init__(data, **kwargs)
        self.stretch = kwargs.get("stretch", False)
        self.cmap = kwargs.get("cmap", "terrain")
        self.title = kwargs.get("title", "DEM")

    @classmethod
    def from_serve5g(
        cls,
        xmin,
        ymin,
        width=2048,
        buffer=500,
        centre=False,
        api_url="http://localhost:8088",
    ):
        """Read a :class:`DEMGrid` from a serve5g REST API.

        Args:
            xmin (int): x coordinate of lower left corner
            ymin (int): y coordinate of lower left corner
            width (int, optional): Size of clipped region in meters
            buffer (int, optional): size of padding in meters
            centre (bool, optional): If True, xmin and ymin is centre of DEM. Default False
            api_url (str, optional): URL of REST API. Default is ``http://localhost:8088``

        Returns:
            DEMGrid: New DEM grid.
        """
        if centre:
            xmin -= width / 2
            ymin -= width / 2
        # round to 2.5 resolution grid
        xmin = 2.5 * round(xmin / 2.5)
        ymin = 2.5 * round(ymin / 2.5)
        width = 2.5 * round(width / 2.5)
        buffer = 2.5 * round(buffer / 2.5)
        url = f"{api_url}/{xmin}/{ymin}/{width}/{buffer}"
        with requests.Session() as s:
            response = s.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with MemoryFile(response.raw) as memfile:
                return DEMGrid.from_file(memfile)

    @classmethod
    def from_pysheds(cls, dem):
        """Create a :class:`DEMGrid` from a ``pysheds`` ``Grid`` object.

        Args:
            dem (pysheds.Grid): A pysheds grid instance with ``data``,
                ``nodata``, ``affine``, ``shape``, ``crs``, and ``dtype``
                attributes.

        Returns:
            DEMGrid: New DEM grid.
        """
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
        """Load an example :class:`DEMGrid` from bundled data.

        Args:
            test (bool, optional): Use ``testdem.tif`` (32×32) when
                ``True``, ``dem.tif`` otherwise. Default ``False``.

        Returns:
            DEMGrid: Example DEM.
        """
        datapath = importlib.resources.files("demtools") / "data"
        if test:
            fname = datapath / "testdem.tif"
        else:
            fname = datapath / "dem.tif"
        return cls.from_file(fname)

    def hillshade(self, **kwargs):
        """Return a hillshade (shaded relief) grid.

        Args:
            azdeg (float, optional): Light azimuth in degrees clockwise
                from north. Default ``150``.
            altdeg (float, optional): Light altitude in degrees above the
                horizon. Default ``40``.
            vert_exag (float, optional): Vertical exaggeration factor.
                Default ``1``.

        Returns:
            FloatGrid: Hillshade grid.
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
        """Return an RGB shaded-relief image.

        Combines the colormapped elevation with hillshading in HSV space.

        Args:
            azdeg (float, optional): Light azimuth. Default ``150``.
            altdeg (float, optional): Light altitude. Default ``40``.
            vert_exag (float, optional): Vertical exaggeration.
                Default ``1``.
            blend_mode (str, optional): Blending mode. Default ``"overlay"``.
            fraction (float, optional): Fraction of hillshade to apply.
                Default ``1``.

        Returns:
            RGBimage: Shaded-relief RGB image.
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
        """Calculate the Topographic Position Index (TPI).

        TPI = spot height – mean neighbourhood height (with a disc-shaped
        neighbourhood). The cell at the centre of the window is excluded.

        Args:
            win (numpy.ndarray, optional): Custom binary structural element.
                Default ``None`` (creates a disk with radius *r*).
            r (int, optional): Disk radius in pixels. Default ``5``.
            cmap (str, optional): Colormap. Default ``"seismic"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"TPI(…)"``.

        Returns:
            FloatGrid: TPI grid.
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
        """Calculate the Topographic Position Index using FFT convolution.

        Faster than :meth:`tpi` for large windows. The centre cell is
        excluded from the neighbourhood sum.

        Args:
            win (numpy.ndarray, optional): Custom binary operation window.
                Default ``None`` (creates a disk with radius *r*).
            r (int, optional): Disk radius in pixels. Default ``5``.
            cmap (str, optional): Colormap. Default ``"seismic"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"TPI(…)"``.

        Returns:
            FloatGrid: TPI grid.
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
        n_sum = n_sum.astype(float)
        n_sum[self._mask] = np.nan
        # this is TPI (spot height – average neighbourhood height)
        tpi = self.filled - n_sum / n_count
        kwargs["cmap"] = kwargs.get("cmap", "seismic")
        kwargs["title"] = kwargs.get("title", f"TPI({self.title})")
        return self.clone(
            tpi, mask=np.isnan(tpi) | self._mask, astype=FloatGrid, **kwargs
        )

    def tpi_fft2(self, **kwargs):
        """Calculate the Topographic Position Index using FFT convolution.

        This version using two radiuses and calculate TPI bbetween inner
        circle and donut circle.

        Args:
            win (numpy.ndarray, optional): Custom binary operation window.
                Default ``None`` (creates a disk with radius *r*).
            r1 (int, optional): Disk radius in pixels. Default ``5``.
            r2 (int, optional): Disk radius in pixels. Default ``10``.
            cmap (str, optional): Colormap. Default ``"seismic"``.
            stretch (bool, optional): Stretch colormap. Default ``True``.
            title (str, optional): Dataset title. Default ``"TPI(…)"``.

        Returns:
            FloatGrid: TPI grid.
        """
        r1 = kwargs.get("r1", 5)
        r2 = kwargs.get("r2", 10)
        assert r2 > r1, "r2 must be greater than r1"
        L = np.arange(-r2, r2 + 1)
        X, Y = np.meshgrid(L, L)
        win_in = np.array((X**2 + Y**2) < r1**2, dtype=int)
        win_out = np.array(
            ((X**2 + Y**2) <= r2**2) & ((X**2 + Y**2) >= r1**2), dtype=int
        )
        # count grid
        c_grid = np.ones(self.data.shape)
        c_grid[self._mask] = 0
        # do fast convolution
        n_sum_in = signal.convolve(self.data.filled(0), win_in, mode="same")
        n_count_in = signal.convolve(c_grid, win_in, mode="same")
        n_sum_in = n_sum_in.astype(float)
        n_sum_in[self._mask] = np.nan

        n_sum_out = signal.convolve(self.data.filled(0), win_out, mode="same")
        n_count_out = signal.convolve(c_grid, win_out, mode="same")
        n_sum_out = n_sum_out.astype(float)
        n_sum_out[self._mask] = np.nan
        # this is TPI (spot height – average neighbourhood height)
        tpi = n_sum_in / n_count_in - n_sum_out / n_count_out
        kwargs["cmap"] = kwargs.get("cmap", "seismic")
        kwargs["title"] = kwargs.get("title", f"TPI2({self.title})")
        return self.clone(
            tpi, mask=np.isnan(tpi) | self._mask, astype=FloatGrid, **kwargs
        )

    def tpi_cube(self, **kwargs):
        """Compute TPI at multiple scales and return as a :class:`FeatureSet`.

        The resulting feature set has one column per scale, allowing
        multi-scale terrain analysis.

        Args:
            n (int, optional): Number of scales. Default ``20``.
            scale (float, optional): Base-2 logarithmic scaling factor.
                Default ``3.322``.

        Returns:
            FeatureSet: Multi-scale TPI features.
        """
        n = kwargs.pop("n", 20)
        scale = kwargs.pop("scale", 3.322)
        progress = kwargs.pop("progress", True)
        steps = np.arange(0, (n - 1) * 2**scale + 1, 2**scale, dtype=int)
        if progress:
            looper = tqdm(steps, desc="Calculating TPI")
        else:
            looper = steps
        for r in looper:
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
        return FeatureSet(
            np.array(cube).T, self._mask, self.meta, feature_coords=steps, **kwargs
        )

    def tpi_cube2(self, **kwargs):
        """Compute TPI2 at multiple scales and return as a :class:`FeatureSet`.

        The resulting feature set has one column per scale, allowing
        multi-scale terrain analysis.

        Args:
            n (int, optional): Number of scales. Default ``20``.
            step (int, optional): Window size step. Default ``10``.

        Returns:
            FeatureSet: Multi-scale TPI features.
        """
        n = kwargs.pop("n", 20)
        step = kwargs.pop("step", 10)
        progress = kwargs.pop("progress", True)
        steps = np.arange(0, n * step, step, dtype=int)
        if progress:
            looper = tqdm(steps, desc="Calculating TPI")
        else:
            looper = steps
        for r in looper:
            if r == 0:
                cube = [np.zeros_like(self.filled, dtype=float)[~self._mask]]
            else:
                tpi = self.tpi_fft2(r1=r - step // 2, r2=r + step // 2).filled
                cube.append(tpi[~self._mask])
        return FeatureSet(
            np.array(cube).T, self._mask, self.meta, feature_coords=steps, **kwargs
        )

    def pits(self, size=3, **kwargs):
        """Detect pits (cells lower than all neighbours in a square ring).

        Uses a ``(size, size)`` square ring kernel (centre excluded).

        Args:
            size (int, optional): Kernel size. Default ``3``.

        Returns:
            BoolGrid: Boolean grid where ``True`` indicates pits.
        """
        win = np.ones((size, size))
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") > d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def pits2(self, size=3, **kwargs):
        """Detect pits using a kernel that excludes only the centre cell.

        Args:
            size (int, optional): Kernel size. Default ``3``.

        Returns:
            BoolGrid: Boolean grid where ``True`` indicates pits.
        """
        win = np.ones((size, size))
        win[size // 2, size // 2] = 0
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") > d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def flats(self, size=3, **kwargs):
        """Detect flat cells (equal to the minimum in a square ring).

        Uses a ``(size, size)`` square ring kernel.

        Args:
            size (int, optional): Kernel size. Default ``3``.

        Returns:
            BoolGrid: Boolean grid where ``True`` indicates flat cells.
        """
        win = np.ones((size, size))
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") == d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def flats2(self, size=3, **kwargs):
        """Detect flat cells using a kernel that excludes only the centre.

        Args:
            size (int, optional): Kernel size. Default ``3``.

        Returns:
            BoolGrid: Boolean grid where ``True`` indicates flat cells.
        """
        win = np.ones((size, size))
        win[size // 2, size // 2] = 0
        d = self.filled
        res = ndimage.minimum_filter(d, footprint=win, mode="mirror") == d
        return self.clone(res, astype=BoolGrid, **kwargs)

    def fill_pits(self, size=3, eps=0, **kwargs):
        """Fill pits by raising cells to the minimum of their neighbourhood.

        Iteratively raises pit cells to the ring-minimum value plus an
        optional epsilon.

        Args:
            size (int, optional): Kernel size. Default ``3``.
            eps (float, optional): Small constant added to the fill value.
                Default ``0``.

        Returns:
            FloatGrid: Pit-filled DEM.
        """
        win = np.ones((size, size))
        # win[size // 2, size // 2] = 0
        win[1:-1, 1:-1] = np.zeros((size - 2, size - 2))
        d = self.filled.copy()
        while True:
            fill = ndimage.minimum_filter(d, footprint=win, mode="mirror")
            changed = fill > d
            if not changed.any():
                break
            d[changed] = fill[changed] + eps
        return self.clone(d, **kwargs)

    def sink_points(self, size):
        """Return the coordinates of sink (pit) centres.

        Args:
            size (int): Kernel size for pit detection.

        Returns:
            list: ``(row, col)`` tuples of sink centre positions.
        """
        # locs = self.fill_pits(3).fill_pits(5).pits(7).fill_holes().opening().closing()
        locs = self.pits2(size)  # .fill_holes().opening().closing()
        labeled_array, num_features = ndimage.label(locs.filled)
        return ndimage.minimum_position(
            self.filled, labels=labeled_array, index=range(1, num_features + 1)
        )


class RGBimage:
    """A class to store and display 3-band RGB imagery.

    Args:
        rgb (numpy.ndarray): 3-D array of shape ``(3, height, width)``.
        title (str, optional): Dataset title. Default ``"RGB"``.
        figsize (tuple, optional): Figure size in inches.
        **meta: Rasterio metadata keys (``crs``, ``transform``, etc.).

    Attributes:
        data (numpy.ndarray): The 3-band raster data.
        title (str): Dataset title.
        figsize (tuple): Figure size.
        meta (dict): Rasterio-compatible metadata.
    """

    def __init__(self, rgb, **kwargs):
        """Construct an RGBimage and validate array shape against metadata."""
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
        if rgb.shape != (3, self.meta["height"], self.meta["width"]):
            raise ValueError(
                f"RGB shape {rgb.shape} does not match metadata "
                f"(3, {self.meta['height']}, {self.meta['width']})"
            )

    def write_tif(self, filename):
        """Write RGB data to a GeoTIFF file.

        Args:
            filename (str): Output file path.
        """
        with rio.open(filename, "w", **self.meta) as dst:
            dst.write(self.data)

    def show(self, **kwargs):
        """Display the RGB image using :func:`rasterio.plot.show`.

        Args:
            title (str, optional): Plot title. Defaults to ``self.title``.
            figsize (tuple, optional): Figure size in inches.
            transform (affine.Affine, optional): Geotransform for axis labels.
            ax (matplotlib.axes.Axes, optional): Existing axes.
            show (bool, optional): Call ``plt.show()``. Default ``True``.
        """
        ax = kwargs.pop("ax", None)
        show = kwargs.pop("show", True)
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
        if show:
            plt.show()


class FeatureSet:
    """A class to store and cluster multi-dimensional feature data.

    Designed for multi-scale TPI feature vectors (rows = unmasked raster
    cells, columns = scales). Supports K-means clustering, hierarchical
    aggregation, and grid reconstruction.

    Args:
        data (numpy.ndarray): 2-D array of shape ``(n_cells, n_features)``.
        mask (numpy.ndarray): Boolean mask of the original raster.
        meta (dict): Rasterio metadata for grid reconstruction.
        feature_coords (numpy.ndarray, optional): 1-D array of coordinates
            for each feature column. Default ``np.arange(n_features)``.
        use_grad (bool, optional): Replace features with their gradient
            along the feature axis. Default ``False``.
        do_cluster (bool, optional): Run clustering immediately.
            Default ``True``.
        **kwargs: Passed to :meth:`cluster`.

    Attributes:
        data (numpy.ndarray): Feature array.
        clusters (numpy.ndarray): Cluster assignments from K-means.
        centers (numpy.ndarray): K-means cluster centres.
        labels (numpy.ndarray): Final labels after hierarchical aggregation.
    """

    def __init__(self, data, mask, meta, **kwargs):
        """Construct a FeatureSet, optionally computing gradient and clustering."""
        self.feature_coords = kwargs.get("feature_coords", np.arange(data.shape[1]))
        if kwargs.get("use_grad", False):
            data = np.gradient(data, self.feature_coords, axis=1)
        self.data = data
        self._mask = mask
        self.meta = meta.copy()
        self.meta["dtype"] = "int32"
        self.meta["nodata"] = -9999
        self.clusters = None
        self.centers = None
        self.labels = None
        if kwargs.get("do_cluster", True):
            self.cluster(**kwargs)

    def cluster(self, **kwargs):
        """Run K-means followed by hierarchical aggregation.

        Sets ``self.clusters``, ``self.centers``, and ``self.labels``.

        Args:
            model (object, optional): Pre-fitted clustering model with
                ``predict`` and ``cluster_centers_`` attributes.
                Default ``None`` (creates a :class:`KMeans` instance).
            n_kmeans (int, optional): Number of K-means clusters.
                Default ``256``.
            random_state (int, optional): Random seed. Default ``42``.
            init (str, optional): Initialisation method. Default
                ``"k-means++"``.

        Returns:
            None
        """
        model = kwargs.get("model", None)
        if model is None:
            n_kmeans = kwargs.get("n_kmeans", 256)
            if len(self.data) > 100_000:
                model = MiniBatchKMeans(
                    n_clusters=n_kmeans,
                    random_state=kwargs.get("random_state", 42),
                    n_init=kwargs.get("n_init", "auto"),
                )
            else:
                model = KMeans(
                    n_clusters=n_kmeans,
                    random_state=kwargs.get("random_state", 42),
                    init=kwargs.get("init", "k-means++"),
                )
            model.fit(self.data)
        self.clusters = model.predict(self.data)
        self.centers = model.cluster_centers_
        self.aggclusters(**kwargs)

    def batch_cluster(self, **kwargs):
        """Run mini-batch K-means for large datasets.

        Sets ``self.clusters``, ``self.centers``, and ``self.labels``.

        Args:
            n_kmeans (int, optional): Number of K-means clusters.
                Default ``256``.
            batch_size (int, optional): Mini-batch size. Default ``1024``.
            n_init (str or int, optional): Number of initialisations.
                Default ``"auto"``.
            bs (int, optional): Chunk size for partial fit.
                Default ``1000000``.

        Returns:
            None
        """
        kmeans = MiniBatchKMeans(
            n_clusters=kwargs.get("n_kmeans", 256),
            random_state=kwargs.get("random_state", 42),
            init=kwargs.get("init", "k-means++"),
            batch_size=kwargs.get("batch_size", 1024),
            n_init=kwargs.get("n_init", "auto"),
        )
        n = self.data.shape[0]
        bs = kwargs.get("bs", 1000000)
        for x in range(0, n, bs):
            kmeans = kmeans.partial_fit(self.data[x : min(x + bs, n), :])
        self.clusters = kmeans.predict(self.data)
        self.centers = kmeans.cluster_centers_
        self.aggclusters(**kwargs)

    def dendrogram(self, **kwargs):
        """Display a dendrogram of the cluster centres.

        Args:
            metric (str, optional): Distance metric. Default ``"euclidean"``.
            linkage (str, optional): Linkage method. Default ``"ward"``.

        Returns:
            None
        """
        Z = hierarchy.linkage(
            self.centers,
            metric=kwargs.get("metric", "euclidean"),
            method=kwargs.get("linkage", "ward"),
        )
        hierarchy.dendrogram(Z)
        plt.show()

    def aggclusters(self, **kwargs):
        """Aggregate K-means clusters via hierarchical clustering.

        Sets ``self.labels`` by mapping each K-means cluster to an
        agglomerative cluster assignment.

        Args:
            n_clusters (int, optional): Target number of aggregated clusters.
                Default ``7``.
            metric (str, optional): Distance metric. Default ``"euclidean"``.
            linkage (str, optional): Linkage method. Default ``"ward"``.

        Returns:
            None
        """
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
        """Compute the mean feature vector per aggregated label.

        Returns:
            numpy.ndarray: Array of shape ``(n_labels, n_features)``.
        """
        cids = np.unique(self.labels)
        return np.array([self.data[self.labels == c, :].mean(axis=0) for c in cids])

    def sort_label_integrals(self):
        """Sort aggregated labels by the integral (area) of their mean curve.

        Reassigns ``self.labels`` so that label 0 has the smallest integral
        and the highest label has the largest, enabling consistent colourmap
        ordering across different runs.

        Returns:
            None
        """
        cids = np.unique(self.labels)
        labels = self.labels.copy()
        six = np.argsort(np.trapezoid(self.labels_average()))
        for a, b in zip(cids, six):
            labels[self.labels == b] = a
        self.labels = labels

    def plot_averages(self, clusters=False, cmap=None):
        """Plot the mean feature vectors for each cluster or label.

        Args:
            clusters (bool, optional): Plot K-means cluster centres instead
                of aggregated labels. Default ``False``.
            cmap (matplotlib.colors.Colormap, optional): Colour map for
                the lines.
        """
        fig, ax = plt.subplots()
        if clusters:
            if cmap is not None:
                ax.set_prop_cycle(
                    plt.cycler("color", cmap(np.linspace(0, 1, self.centers.shape[1])))
                )
            ax.plot(self.feature_coords, self.centers.T)
            for c, y in enumerate(self.centers[:, -1]):
                ax.text(self.feature_coords[-1], y, f"{c}", verticalalignment="center")
        else:
            cids = np.unique(self.labels)
            cave = self.labels_average()
            if cmap is not None:
                ax.set_prop_cycle(
                    plt.cycler("color", cmap(np.linspace(0, 1, cave.shape[0])))
                )
            ax.plot(self.feature_coords, cave.T)
            for c, y in zip(cids, cave[:, -1]):
                ax.text(self.feature_coords[-1], y, f"{c}", verticalalignment="center")
        plt.show()

    def as_grid(self, clusters=False):
        """Reconstruct a raster grid from the cluster or label assignments.

        Args:
            clusters (bool, optional): If ``True``, use K-means cluster
                assignments; otherwise use aggregated labels.
                Default ``False``.

        Returns:
            IntGrid: Grid of cluster/label values.
        """
        res = IntGrid(
            self.meta["nodata"] * np.ones((self.meta["height"], self.meta["width"])),
            mask=self._mask,
            **self.meta,
        )
        if clusters:
            return res.clone(self.clusters, only_valid=True)
        else:
            return res.clone(self.labels, only_valid=True)

    def difference(self, dst, threshold=None):
        """Return sum of squared differences of features from a reference vector.

        Args:
            dst (array-like): Reference feature vector of length ``n_features``.
            threshold (float, optional): If given, return a :class:`BoolGrid`
                where ``True`` means the squared difference is below this
                value. Default ``None`` (return :class:`FloatGrid`).

        Returns:
            FloatGrid or BoolGrid: Grid of summed squared differences, or a
            boolean grid if ``threshold`` is provided.

        Raises:
            ValueError: If ``dst`` length does not match ``n_features``.
        """
        if len(dst) != self.data.shape[1]:
            raise ValueError(
                f"Wrong length of dst vector. Must be {self.data.shape[1]}"
            )
        res = FloatGrid(
            self.meta["nodata"] * np.ones((self.meta["height"], self.meta["width"])),
            mask=self._mask,
            **self.meta,
        )
        vals = np.sum((self.data - np.asarray(dst)) ** 2, axis=1)
        if threshold is None:
            return res.clone(vals, only_valid=True)
        else:
            return res.clone(vals, only_valid=True) < threshold
