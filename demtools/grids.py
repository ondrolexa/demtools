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
from scipy.signal import medfilt2d

from demtools.mathlib import derivx, derivy, derivz, upcontinue


class Grid:
    KIND = ""

    def __init__(self, data, **kwargs):
        self.cmap = kwargs.get("cmap", "viridis")
        self.stretch = kwargs.get("stretch", True)
        self.title = kwargs.get("title", "Grid")
        self.figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        self.meta = {
            "driver": "",
            "dtype": "",
            "nodata": 0,
            "width": 0,
            "height": 0,
            "count": 1,
            "compress": "",
            "crs": riocrs.CRS.from_epsg(3857),
            "transform": Affine(1.0, 0.0, 0, 0.0, -1.0, 0),
        }
        self.meta.update((k, v) for k, v in kwargs.items() if k in self.meta)
        assert data.shape == (
            self.meta["height"],
            self.meta["width"],
        ), "Wrong metadata !"
        # check mask shape
        if data.mask.ndim < 2:
            data.mask = np.ones(data.shape, dtype="bool") * data.mask
        self.data = data
        if not data.dtype.name.startswith(self.__class__.KIND):
            raise ValueError(
                f"Data type {data.dtype.name} is not valid for {self.__class__.__name__}. "
                + f"Must be {self.__class__.KIND}."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.title} "
            + f"[{self.meta['height']},{self.meta['width']}] "
            + f"masked:{self.data.mask.sum()} EPSG:{self.meta['crs'].to_epsg()}"
        )

    def __getitem__(self, mask):
        data = self.data.copy()
        if isinstance(mask, BoolGrid):
            data.mask = np.logical_or(self.data.mask, ~mask.data.data)
        else:
            data.mask = np.logical_or(self.data.mask, ~mask)
        return self.clone(data)

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

    def __lt__(self, other):
        if issubclass(type(other), Grid):
            data = self.data < other.data
        else:
            data = self.data < other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def __le__(self, other):
        if issubclass(type(other), Grid):
            data = self.data <= other.data
        else:
            data = self.data <= other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def __eq__(self, other):
        if issubclass(type(other), Grid):
            data = self.data == other.data
        else:
            data = self.data == other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def __ne__(self, other):
        if issubclass(type(other), Grid):
            data = self.data != other.data
        else:
            data = self.data != other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def __gt__(self, other):
        if issubclass(type(other), Grid):
            data = self.data > other.data
        else:
            data = self.data > other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def __ge__(self, other):
        if issubclass(type(other), Grid):
            data = self.data >= other.data
        else:
            data = self.data >= other
        return self.clone(data, cmap="binary", astype=BoolGrid)

    def clone(self, data, **kwargs):
        """Clone grid with new data

        Args:
            data (numpy.ma.MaskedArray): data as masked numpy array
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default keep original.
            figsize (tuple, optional): matplotlib figure size. Default keep original.

        """
        typObj = kwargs.pop("astype", type(self))
        return typObj(
            data,
            cmap=kwargs.get("cmap", self.cmap),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get("title", self.title),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
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

    @classmethod
    def from_file(cls, filename, **kwargs):
        band = kwargs.get("band", 1)
        with rio.open(filename) as src:
            data = src.read(band, masked=True)
            meta = src.meta
        return cls(data, **kwargs, **meta)

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

    @property
    def _values(self):
        return np.asarray(self.data[~self.data.mask])

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
    KIND = "bool"

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.cmap = kwargs.get("cmap", "binary")

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


class IntGrid(Grid):
    KIND = "int"

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

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
        datapath = importlib.resources.files("demtools") / "data"
        if test:
            fname = datapath / "int.tif"
        else:
            fname = datapath / "int.tif"
        return cls.from_file(fname)


class FloatGrid(Grid):
    KIND = "float"

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

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
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        return self.clone(
            (self.data - self.data.min()) / (self.data.max() - self.data.min()),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"NORM({self.title})"),
        )

    def inverted(self, **kwargs):
        """Returns inverted dataset

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NORM(...)"'.

        """
        return self.clone(
            self.data.max() - self.data + self.data.min(),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"INV({self.title})"),
        )

    def digitize(self, bins, **kwargs):
        """Return the IntGrid with indices of the bins to which each value belongs

        Args:
            bins (array_like): Array of bins. It has to be 1-dimensional and monotonic.
            right(bool, optional): Indicating whether the intervals include the right
                or the left bin edge. See numpy.digitize for more details. Default `False`
            cmap (str, optional): Colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"DIG"'.
        """
        right = kwargs.get("right", False)
        data = np.digitize(self.data, bins, right=right)
        return IntGrid(
            np.ma.masked_array(
                data,
                mask=self.data.mask,
                fill_value=self.data.fill_value,
            ),
            cmap=kwargs.get("cmap", self.cmap),
            title=kwargs.get("title", "DIG"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def resample(self, scale, **kwargs):
        """Returns resampled dataset

        Args:
            scale (float): Resampling scale.

        """
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
        return type(self)(data, **kwargs, **meta)

    def dx(self, **kwargs):
        """Returns horizontal derivative dx

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dx(...)"'.

        """
        return FloatGrid(
            self._dx,
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"dx({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def dy(self, **kwargs):
        """Returns horizontal derivative dy

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dy(...)"'.

        """
        return FloatGrid(
            self._dy,
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"dy({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def dz(self, **kwargs):
        """Returns vertical derivative dz

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"dz(...)"'.

        """
        return FloatGrid(
            np.ma.masked_array(self._dz, mask=self._dx.mask | self._dy.mask),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"dz({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

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
        return FloatGrid(
            np.ma.masked_array(up, mask=self._dx.mask | self._dy.mask),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"up({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def thd(self, **kwargs):
        """Returns total horizontal derivative

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"THD(...)"'.

        """
        return FloatGrid(
            self._thd,
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"THD({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def tga(self, **kwargs):
        """Returns total gradient amplitude (also called the analytic signal)

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"TGA(...)"'.

        """
        return FloatGrid(
            self._tga,
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"TGA({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def theta(self, **kwargs):
        """Returns theta - total horizontal derivative divided by the analytical signal

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"Theta(...)"'.

        """
        return FloatGrid(
            self._thd / self._tga,
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"Theta({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def nthd(self, **kwargs):
        """Returns normalized total horizontal derivative

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"NTHD(...)"'.

        """
        return FloatGrid(
            np.real(np.arctan2(self._thd, np.absolute(self._dz))),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"NTHD({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def tilt(self, **kwargs):
        """Returns tilt angle in radians

        Args:
            cmap (str, optional): Colormap. Default `"bone_r"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"Tilt(...)"'.

        """
        return FloatGrid(
            np.arctan2(self._dz, self._thd),
            cmap=kwargs.get("cmap", "bone_r"),
            title=kwargs.get("title", f"Tilt({self.title})"),
            figsize=kwargs.get("figsize", self.figsize),
            **self.meta,
        )

    def gaussian_filter(self, **kwargs):
        """Returns Gaussian filtered dataset

        Args:
            sigma(float): Standard deviation for Gaussian kernel. Default 1
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"G(...)"'.

        """
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
        """Returns median filtered dataset

        Args:
            kernel_size(int): A scalar or a list of length 2, giving the size
                of the median filter window. Default 3
            cmap (str, optional): Colormap. Default keep original.
            stretch (bool, optional): Stretch colormap. Default keep original.
            title (str, optional): Title of dataset. Default '"M(...)"'.

        """
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
        >>> d = DEMGrid.from_exmaples('smalldem')
        >>> d.show()

    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.stretch = kwargs.get("stretch", False)
        self.cmap = kwargs.get("cmap", "terrain")
        self.title = kwargs.get("title", "DEM")

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
        return FloatGrid(
            hs,
            cmap=kwargs.get("cmap", "gray"),
            stretch=kwargs.get("stretch", self.stretch),
            title=kwargs.get(
                "title", f"Hillshade({self.title}, {vert_exag}, {azdeg}, {altdeg})"
            ),
            **self.meta,
        )

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
            win(numpy.array, optional): operation window. Default rectangle 2*r + 1
            r(int, optional): radius of window. Default `5`
            cmap (str, optional): Colormap. Default `"seismic"`.
            stretch (bool, optional): Stretch colormap. Default `True`.
            title (str, optional): Title of dataset. Default '"TPI(...)"'.

        """

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
        return FloatGrid(
            np.ma.masked_array(
                tpi,
                mask=np.isnan(tpi) | self.data.mask,
                fill_value=self.data.fill_value,
            ),
            cmap=kwargs.get("cmap", "seismic"),
            title=kwargs.get("title", f"TPI({self.title})"),
            **self.meta,
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
