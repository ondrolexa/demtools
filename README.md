# demtools

[![PyPI - Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

Python package to manipulate and analyze Digital Elevation Model (DEM) grids. Built on `numpy`, `rasterio`, `scipy`, and `matplotlib`.

## Installation

```bash
pip install demtools
```

For development:

```bash
pip install -e ".[dev]"      # editable install with all dev deps
pre-commit install            # enable pre-commit hooks
```

## Quick start

```python
from demtools import DEMGrid, FloatGrid, BoolGrid

# Load a DEM
dem = DEMGrid.example(test=True)
print(dem.shape, dem.resolution)

# Compute derivatives
dx = dem.dx()
dy = dem.dy()
dz = dem.dz()

# Topographic Position Index
tpi = dem.tpi(r=5)

# Hillshade
hs = dem.hillshade()

# Pits and sinks
pits = dem.pits(size=3)
points = dem.sink_points(size=3)

# Boolean masking
high = dem > 512
mean_high = dem[high].mean
```

## Features

### Grid types

| Class | Data type | Nodata | Description |
|---|---|---|---|
| `BoolGrid` | `bool` | `False` | Boolean / binary masks |
| `IntGrid` | `int32` | `-9999` | Discrete / categorical data |
| `FloatGrid` | `float64` | `nan` | Continuous data |
| `DEMGrid` | `float64` | `nan` | Digital Elevation Models (inherits `FloatGrid`) |

### Grid operations

All grid types support:

- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `**` (scalar or grid)
- **Comparison**: `<`, `<=`, `==`, `!=`, `>`, `>=` (returns `BoolGrid`)
- **Indexing**: `grid[bool_mask]` selects/sets cells via boolean array or `BoolGrid`
- **I/O**: `read_tif()`, `write_tif()`, `from_file()`
- **Clone**: `clone(data, astype=OtherGrid, **kwargs)` for type conversion
- **Filters**: `generic_filter()`, `correlation()`, `similarity()`
- **Sampling**: `sample(pts)`, `sample_line(p1, p2, n)`, `index(point)`
- **Aggregation**: `aggregate(size, method=...)` — mean, median, std, variance, min, max
- **Clip**: `clip(r, h, c, w)` with automatic transform update

### `BoolGrid` methods

| Method | Description |
|---|---|
| `count_true` / `count_false` | Count boolean values |
| `erosion(n)` / `dilation(n)` | Morphological operations |
| `opening(n)` / `closing(n)` | Morphological opening / closing |
| `fill_holes()` | Fill false holes in true regions |
| `remove_dots(size)` | Remove small true regions |
| `clear_boundary()` | Remove regions touching edges |
| `label()` | Label connected components → `IntGrid` |

### `IntGrid` methods

| Method | Description |
|---|---|
| `unique_values` | Array of unique values |
| `counts(plot=False)` | Value frequencies |
| `moving_average(r)` / `majority_filter(size)` | Neighborhood filters |
| `normalized()` | Normalize to [0, 1] → `FloatGrid` |
| `remove_dots(size)` | Remove small connected regions per value |
| `polygonize(file=None)` | Vectorize → GeoJSON features or Shapely geometries |

### `FloatGrid` methods

| Method | Description |
|---|---|
| `min` / `max` / `mean` | Statistics (mask-aware) |
| `normalized()` / `inverted()` | Scale to [0, 1] or invert |
| `moving_average(r)` | Moving window average |
| `digitize(bins)` | Bin values → `IntGrid` |
| `resample(scale, resampling=...)` | Resample via rasterio |
| `dx()` / `dy()` / `dz()` | Horizontal/vertical derivatives |
| `thd()` / `tga()` | Total horizontal derivative / gradient amplitude |
| `theta()` / `nthd()` / `tilt()` | Theta map, normalized THD, tilt angle |
| `upcont(height)` | Upward continuation (FFT-based) |
| `gaussian_filter()` / `median_filter()` | Smoothing filters |
| `overlay(other)` | Combine in HSV space → `RGBimage` |

### `DEMGrid` methods (inherits `FloatGrid`)

| Method | Description |
|---|---|
| `hillshade()` | Illumination map |
| `shade()` | Color hillshade → `RGBimage` |
| `tpi(r)` / `tpi_fft(r)` | Topographic Position Index |
| `tpi_cube(n, scale)` | Multi-scale TPI → `FeatureSet` |
| `pits(size)` / `pits2(size)` | Local minimum detection |
| `flats(size)` / `flats2(size)` | Flat area detection |
| `fill_pits(size, eps)` | Fill local depressions |
| `sink_points(size)` | Sink locations |
| `from_pysheds(dem)` | Import from pysheds DEM |

### Additional classes

- **`RGBimage`** — 3-band raster with `show()` and `write_tif()`
- **`FeatureSet`** — ML feature storage with KMeans clustering, hierarchical aggregation, and `as_grid()` reconstruction
- **`H5Store`** — Read-only HDF5 → `DEMGrid` adapter

## Example: full workflow

```python
from demtools import DEMGrid, BoolGrid

dem = DEMGrid.example(test=True)

# Terrain analysis
pits_mask = dem.pits(size=5)
filled = dem.fill_pits(size=5, eps=0.01)

# Slope-related derivatives
slope = dem.tga()
aspect = dem.theta()

# Cluster terrain types
from demtools.grids import FeatureSet
cube = dem.tpi_cube(n=10)
features = FeatureSet(cube.data, cube._mask, cube.meta)
features.cluster(n_kmeans=32, n_clusters=5)
landforms = features.as_grid()
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

See `AGENTS.md` for detailed architecture notes and session logs.

## License

MIT
