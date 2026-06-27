# demtools

![PyPI - Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Python package to manipulate and analyse Digital Elevation Model (DEM) grids.
Built on `numpy`, `rasterio`, `scipy`, and `matplotlib`.

## Installation

Install from PyPI:

```bash
pip install demtools
```

Optional extras:

| Extra | Installs | Use when |
|---|---|---|
| `lab` | JupyterLab | interactive notebook work |
| `tests` | pytest, nbval | running the test suite |
| `docs` | Sphinx and extensions | building the documentation |

```bash
pip install demtools[lab]      # + JupyterLab
pip install demtools[tests]    # + test suite tools
```

For development, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ondrejlexa/demtools.git
cd demtools
uv sync   # creates .venv and installs all dev dependencies
```

## Quick start

```python
from demtools import DEMGrid

# Load the bundled test DEM
dem = DEMGrid.example(test=True)
print(dem.shape, dem.resolution)  # (24, 26), (...)

# Terrain derivatives
slope = dem.tga()      # total gradient amplitude
aspect = dem.theta()   # azimuth angle

# Topographic Position Index
tpi = dem.tpi(r=5)

# Hillshade
hs = dem.hillshade()
hs.show()

# Boolean indexing
high = dem > 512       # BoolGrid
mean_high = dem[high].mean  # float — mean of cells above 512 m
```

## Grid class hierarchy

```
Grid  (base — masked array + rasterio metadata)
├── BoolGrid   (dtype=bool,    nodata=False)
├── IntGrid    (dtype=int32,   nodata=-9999)
└── FloatGrid  (dtype=float64, nodata=nan)
    └── DEMGrid
```

| Class | Purpose |
|---|---|
| `BoolGrid` | Binary masks; morphological operations |
| `IntGrid` | Discrete / categorical data; polygonize |
| `FloatGrid` | Continuous fields; FFT derivatives |
| `DEMGrid` | Digital elevation models; terrain analysis |
| `RGBimage` | 3-band raster for display and export |
| `FeatureSet` | Multi-scale feature vectors for ML clustering |
| `H5Store` | Read-only HDF5 → `DEMGrid` adapter |

## Data loading

```python
from demtools import DEMGrid, H5Store

# From a GeoTIFF (or any rasterio-supported format)
dem = DEMGrid.from_file("path/to/dem.tif")

# Bundled example DEM (no file needed)
dem = DEMGrid.example(test=True)

# From a REST service (5G DEM API)
dem = DEMGrid.from_serve5g(lon=14.4, lat=50.0, size=1000)

# From an HDF5 file (context manager ensures file is closed)
with H5Store("path/to/data.h5") as store:
    dem = store.clip(r=0, h=200, c=500, w=200)

# From a pysheds DEM object
from pysheds.grid import Grid as PyshedsGrid
pg = PyshedsGrid.from_raster("path/to/dem.tif")
dem = DEMGrid.from_pysheds(pg.dem)
```

## Grid operations

All grid types support:

- **Arithmetic** — `+`, `-`, `*`, `/`, `//`, `**` with a scalar or another grid
- **Comparison** — `<`, `<=`, `==`, `!=`, `>`, `>=` returning a `BoolGrid`
- **Boolean indexing** — `grid[bool_mask]` to select or assign cells
- **I/O** — `from_file()`, `write_tif()`
- **Clone** — `clone(data, astype=OtherGrid)` preserving metadata
- **Filters** — `generic_filter()`, `correlation()`, `similarity()`
- **Sampling** — `sample(pts)`, `sample_line(p1, p2, n)`, `index(point)`
- **Aggregation** — `aggregate(size, method=...)` (mean, median, std, variance, min, max)
- **Clip** — `clip(r, h, c, w)` with automatic transform update

### `BoolGrid` methods

| Method | Description |
|---|---|
| `erosion(n)` / `dilation(n)` | Morphological erosion / dilation |
| `opening(n)` / `closing(n)` | Morphological opening / closing |
| `fill_holes()` | Fill `False` holes enclosed by `True` |
| `remove_dots(size)` | Remove small `True` regions |
| `clear_boundary()` | Remove regions touching the raster edge |
| `label()` | Label connected components → `IntGrid` |
| `count_true` / `count_false` | Count `True` / `False` cells |

### `IntGrid` methods

| Method | Description |
|---|---|
| `unique_values` | Array of unique values |
| `counts(plot=False)` | Value frequency table |
| `majority_filter(size)` | Replace each cell with neighbourhood majority |
| `moving_average(r)` | Integer moving average |
| `normalized()` | Rescale to [0, 1] → `FloatGrid` |
| `remove_dots(size)` | Remove small connected regions per label |
| `polygonize(file=None)` | Vectorise → GeoJSON features or Shapely geometries |

### `FloatGrid` methods

| Method | Description |
|---|---|
| `min` / `max` / `mean` | Statistics (mask-aware) |
| `normalized()` / `inverted()` | Rescale to [0, 1] or invert |
| `moving_average(r)` | Moving window average |
| `digitize(bins)` | Bin values → `IntGrid` |
| `resample(scale, resampling=...)` | Resample via rasterio |
| `dx()` / `dy()` / `dz()` | Horizontal / vertical FFT derivatives |
| `thd()` / `tga()` | Total horizontal derivative / gradient amplitude |
| `theta()` / `nthd()` / `tilt()` | Azimuth, normalised THD, tilt angle |
| `upcont(h)` | Upward continuation (FFT-based) |
| `gaussian_filter()` / `median_filter()` | Smoothing filters |
| `overlay(other)` | Combine two grids in HSV space → `RGBimage` |

### `DEMGrid` methods

Inherits all `FloatGrid` methods, and adds:

| Method | Description |
|---|---|
| `hillshade()` | Illumination map → `RGBimage` |
| `shade()` | Colour hillshade → `RGBimage` |
| `tpi(r)` / `tpi_fft(r)` | Topographic Position Index (spatial / FFT) |
| `tpi_cube(n, scale)` | Multi-scale TPI stack → `FeatureSet` |
| `pits(size)` / `pits2(size)` | Local minimum detection |
| `flats(size)` / `flats2(size)` | Flat area detection |
| `fill_pits(size, eps)` | Fill local depressions iteratively |
| `sink_points(size)` | Coordinates of sink centres |

## Example: terrain classification workflow

```python
from demtools import DEMGrid, FeatureSet

dem = DEMGrid.example(test=True)

# 1. Pit-fill the DEM
filled = dem.fill_pits(size=5, eps=0.01)

# 2. Build a multi-scale TPI feature cube (10 radii)
cube = filled.tpi_cube(n=10)

# 3. Cluster into terrain landform classes
features = FeatureSet(cube.data, cube._mask, cube.meta,
                      n_kmeans=32, n_clusters=5)

# 4. Reconstruct a labelled raster
landforms = features.as_grid()
landforms.show()

# 5. Inspect mean TPI profiles per class
features.plot_averages()
```

## Development

```bash
uv sync                       # install package + all dev/docs dependencies

uv run pytest tests/          # run unit tests only
uv run pytest --nbval-lax notebooks/  # run notebook smoke tests
uv run pytest                 # run everything

uv run black src/ tests/      # format
uv run isort src/ tests/      # sort imports
uv run flake8 src/ tests/     # lint

cd docs && uv run make html   # build docs
```

## License

MIT © Ondrej Lexa
