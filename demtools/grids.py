import importlib.resources
from contextlib import contextmanager
from functools import cached_property

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.crs as riocrs
import rasterio.plot as rioplot
import rasterio.sample as riosample
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d, ss2tf

from demtools.mathlib import derivx, derivy, derivz, upcontinue


class DEMGrid:
    """A class to store digital elevation model.

    Args:
        data (numpy.ma.MaskedArray): data as masked numpy array
        meta (dict): rasterio metadata
        cmap (str, optional): Colormap. Default is `"terrain"`
        stretch (bool, optional): Stretch colormap. Default is `False`
        title (str, optional): title of dataset. Default is `"DEM"`
        figsize (tuple, optional): matplotlib figure size.

    Attributes:
        min (float): minimum of data
        max (float): maximum of data

    Example:
        >>> d = DEMGrid.from_exmaples('smalldem')
        >>> d.show()

    """

    def __init__(self, data, **kwargs):
        self.cmap = kwargs.pop("cmap", "terrain")
        self.stretch = kwargs.pop("stretch", False)
        self.title = kwargs.pop("title", "DEM")
        self.figsize = kwargs.pop("figsize", plt.rcParams["figure.figsize"])
        self.meta = {
            "driver": "GTiff",
            "dtype": "float64",
            "nodata": -9999.0,
            "width": 0,
            "height": 0,
            "count": 1,
            "compress": "lzw",
            "crs": riocrs.CRS.from_epsg(3857),
            "transform": Affine(1.0, 0.0, 0, 0.0, -1.0, 0),
        }
        self.meta.update(kwargs)
        assert data.shape == (
            self.meta["height"],
            self.meta["width"],
        ), "Wrong metadata !"
        self.data = data

    @classmethod
    def from_file(cls, filename, **kwargs):
        with rio.open(filename) as src:
            data = src.read(1, masked=True)
            meta = src.meta
        return cls(data, **kwargs, **meta)

    @classmethod
    def from_examples(cls, example=None):
        """Get example dem data

        Args:
            example (str, optional): Name of example. When None, available examples
                are printed. Default is `None`

        Returns:
            DEMGrid: digital elevation model

        """
        datapath = importlib.resources.files("demtools") / "data"
        if example is None:
            print(f"Available examples: {[f.stem for f in datapath.glob('*.tif')]}")
        else:
            fname = (datapath / example).with_suffix(".tif")
            assert fname.exists(), "Example {example} do not exists."
            return cls.from_file(fname)

    def __repr__(self):
        return (
            f"{self.title} [{self.meta['height']},{self.meta['width']}] "
            + f"EPSG:{self.meta['crs'].to_epsg()} masked:{self.data.mask.sum()}"
        )

    def write_tif(self, filename):
        """Write dataset to file

        Args:
            filename (str): filename

        """
        if np.any(self.data.mask):
            with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rio.open(filename, "w", **self.meta) as dst:
                    dst.write(self.data.filled(np.nan), 1)
                    dst.write_mask((~self.data.mask * 255).astype("uint8"))
        else:
            with rio.open(filename, "w", **self.meta) as dst:
                dst.write_band(1, self.data.filled())

    def clone(self, data, **kwargs):
        """Clone dataset with modified data

        Args:
            data (numpy.ma.MaskedArray): data as masked numpy array
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): title of dataset. Default keep original.
            figsize (tuple, optional): matplotlib figure size. Default keep original.

        """
        return DEMGrid(
            data,
            cmap=kwargs.get("cmap", self.cmap),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get("title", self.title),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    @property
    def _values(self):
        return np.asarray(self.data[~self.data.mask])

    @cached_property
    def _dx(self):
        """Horizontal derivative dx as numpy array"""
        return derivx(self.data, self.meta["transform"].a)

    @cached_property
    def _dy(self):
        """Horizontal derivative dy as numpy array"""
        return derivy(self.data, self.meta["transform"].e)

    @cached_property
    def _dz(self):
        """Vertical derivative dz as numpy array"""
        return derivz(
            self.data.filled(self._values.mean()),
            self.meta["transform"].a,
            self.meta["transform"].e,
        )

    @property
    def _thd(self):
        """Total horizontal derivative as numpy array"""
        return np.sqrt(self._dx**2 + self._dy**2)

    @property
    def _tga(self):
        """Total gradient amplitude as numpy array"""
        return np.sqrt(self._dx**2 + self._dy**2 + self._dz**2)

    @property
    def min(self):
        return self._values.min().item()

    @property
    def max(self):
        return self._values.max().item()

    def normalized(self, **kwargs):
        """Returns normalized dataset to range (0,1)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): title of dataset. Default '"NORM(...)"'.

        """
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"NORM({self.title})"),
        )

    def inverted(self, **kwargs):
        return self.clone(
            self.data.max() - self.data + self.data.min(),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"INV({self.title})"),
        )

    @contextmanager
    def asdataset(self):
        with MemoryFile() as memfile:
            # if np.any(self.data.mask):
            with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with memfile.open(**self.meta) as dst:
                    dst.write(self.data, 1)
            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return

    def resample(self, scale=1):
        """Resample grid"""
        t = self.meta["transform"]
        transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
        height = int(self.meta["height"] * scale)
        width = int(self.meta["width"] * scale)
        meta = self.meta.copy()
        meta.update(transform=transform, height=height, width=width)
        with self.asdataset() as src:
            data = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
                masked=True,
            )
        return DEMGrid(data, **meta)

    def sample(self, pts):
        """Return array of values for sample points"""
        with self.asdataset() as src:
            res = np.array([*riosample.sample_gen(src, pts)]).flatten()
        return res

    def sample_line(self, p1, p2, n=10):
        """Return array of values along line defined by endpoints p1 and p2"""
        pts = (
            (p1[0] + k * (p2[0] - p1[0]), p1[1] + k * (p2[1] - p1[1]))
            for k in np.linspace(0, 1, n)
        )
        return self.sample(pts)

    def dx(self, **kwargs):
        """Horizontal derivative dx"""
        return self.clone(
            self._dx,
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"dx({self.title})"),
        )

    def dy(self, **kwargs):
        """Horizontal derivative dy"""
        return self.clone(
            self._dy,
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"dy({self.title})"),
        )

    def dz(self, **kwargs):
        """Vertical derivative dz"""
        return self.clone(
            np.ma.masked_array(self._dz, mask=self._dx.mask | self._dy.mask),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"dz({self.title})"),
        )

    def upcont(self, h, **kwargs):
        """Upward continuation"""
        up = upcontinue(
            self.data, self.meta["transform"].a, self.meta["transform"].e, h
        )
        return self.clone(
            np.ma.masked_array(up, mask=self._dx.mask | self._dy.mask),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"up({self.title})"),
        )

    def thd(self, **kwargs):
        """Total horizontal derivative"""
        return self.clone(
            self._thd,
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"THD({self.title})"),
        )

    def tga(self, **kwargs):
        """Total gradient amplitude (also called the analytic signal)"""
        return self.clone(
            self._tga,
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"TGA({self.title})"),
        )

    def theta(self, **kwargs):
        """Theta - total horizontal derivative divided by the analytical signal"""
        return self.clone(
            self._thd / self._tga,
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"Theta({self.title})"),
        )

    def nthd(self, **kwargs):
        """Normalized total horizontal derivative"""
        return self.clone(
            np.real(np.arctan2(self._thd, np.absolute(self._dz))),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"NTHD({self.title})"),
        )

    def tilt(self, **kwargs):
        """Calculate the tilt angle in radians"""
        return self.clone(
            np.arctan2(self._dz, self._thd),
            cmap=kwargs.get("cmap", "bone_r"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"Tilt({self.title})"),
        )

    def gaussian_filter(self, **kwargs):
        """Gaussian filter"""
        sigma = kwargs.get("sigma", 1)
        filtered = gaussian_filter(self.data.filled(np.nan), sigma)
        return self.clone(
            np.ma.masked_array(
                filtered,
                mask=np.isnan(filtered) | self.data.mask,
                fill_value=self.data.fill_value,
            ),
            cmap=kwargs.get("cmap", self.cmap),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get("title", f"G({self.title}, {sigma})"),
        )

    def median(self, **kwargs):
        """Median filter"""
        kernel_size = kwargs.get("kernel_size", 3)
        mask_zero = kwargs.get("mask_zero", True)
        filtered = medfilt2d(self.data.filled(np.nan), kernel_size)
        return self.clone(
            np.ma.masked_array(
                filtered,
                mask=np.isnan(filtered)
                | self.data.mask
                | ((filtered == 0.0) & mask_zero),
                fill_value=self.data.fill_value,
            ),
            cmap=kwargs.get("cmap", self.cmap),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get("title", f"M({self.title}, {kernel_size})"),
        )

    def hillshade(self, **kwargs):
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
        return self.clone(
            hs,
            cmap=kwargs.get("cmap", "gray"),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get(
                "title", f"Hillshade({self.title}, {vert_exag}, {azdeg}, {altdeg})"
            ),
        )

    def shade(self, **kwargs):
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
        def view(offset_y, offset_x, shape):
            size_y, size_x = shape
            x, y = abs(offset_x), abs(offset_y)
            x_in = slice(x, size_x)
            x_out = slice(0, size_x - x)
            y_in = slice(y, size_y)
            y_out = slice(0, size_y - y)
            # the swapping trick
            if offset_x < 0:
                x_in, x_out = x_out, x_in
            if offset_y < 0:
                y_in, y_out = y_out, y_in
            # return window view (in) and main view (out)
            return np.s_[y_in, x_in], np.s_[y_out, x_out]

        win = kwargs.get("win", None)
        if win is None:
            r = kwargs.get("r", 5)
            win = np.ones((2 * r + 1, 2 * r + 1))

        r_y, r_x = win.shape[0] // 2, win.shape[1] // 2
        win[r_y, r_x] = 0  # ensure the central cell is zero
        # matrices for temporary data
        n_sum = np.zeros(self.data.shape)
        n_count = np.zeros(self.data.shape)

        for (y, x), weight in np.ndenumerate(win):
            if weight == 0:
                continue  # skip zero values !
            # determine views to extract data
            view_in, view_out = view(y - r_y, x - r_x, self.data.shape)
            # using window weights (eg. for a Gaussian function)
            n_sum[view_out] += self.data.filled(np.nan)[view_in] * weight
            # track the number of neighbours
            # (this is used for weighted mean : Σ weights*val / Σ weights)
            n_count[view_out] += weight

        # this is TPI (spot height – average neighbourhood height)
        tpi = self.data.filled(np.nan) - n_sum / n_count
        return self.clone(
            np.ma.masked_array(
                tpi,
                mask=np.isnan(tpi) | self.data.mask,
                fill_value=self.data.fill_value,
            ),
            cmap=kwargs.get("cmap", "seismic"),
            stretch=kwargs.get("stretch", True),
            title=kwargs.get("title", f"TPI({self.title})"),
        )

    def overlay(self, over, invert=False):
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


class RGBimage:
    def __init__(self, rgb, **kwargs):
        self.data = rgb
        self.title = kwargs.pop("title", "RGB image")
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

    def write_tif(self, filename):
        with rio.open(filename, "w", **self.meta) as dst:
            dst.write(self.data)

    def show(self, **kwargs):
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
