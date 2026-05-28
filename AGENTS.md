# demtools

Python package to manipulate and analyze DEM (Digital Elevation Model) grids.

## Quick start

```bash
.venv/bin/pip install -e ".[tests]"     # editable install with test deps
.venv/bin/pre-commit install             # enable pre-commit hooks
```

## Commands

Prefix all commands with `.venv/bin/` or activate the venv first.

| Action | Command |
|---|---|
| Run all tests (incl. notebooks) | `.venv/bin/python -m pytest --nbval-lax` |
| Run unit tests only | `.venv/bin/python -m pytest tests/` |
| Format code | `.venv/bin/black src/ tests/` |
| Lint | `.venv/bin/flake8 src/ tests/ --max-line-length=88 --ignore=F401,E501,W503,E731,E743,E741,E203` |
| Sort imports | `.venv/bin/isort src/ tests/ --profile black` |

CI runs `python -m pytest --nbval-lax` on ubuntu-latest for Python 3.10–3.12.

Pre-commit runs isort → black → flake8 → pytest.

## Architecture

Single-package project in `src/` layout. Package name `demtools`.

```
src/demtools/
  __init__.py   → exports DEMGrid, FloatGrid, IntGrid, BoolGrid, H5Store
  grids.py      → Grid base + BoolGrid, IntGrid, FloatGrid, DEMGrid, RGBimage, FeatureSet
  mathlib.py    → FFT gradients, upward continuation (adapted from Fatiando a Terra)
  storage.py    → H5Store reader for HDF5 DEM files
  data/         → bundled test TIFFs: testdem.tif, dem.tif, int.tif
```

### Class hierarchy

`Grid` → `BoolGrid`, `IntGrid`, `FloatGrid` → `DEMGrid`

All grid classes wrap `numpy.ma.MaskedArray` with `rasterio` CRS/transform metadata. `clone()` is the common copy-with-new-data pattern.

### Non-obvious conventions

- `DEMGrid.stretch` defaults to `False`; `FloatGrid.stretch` defaults to `True`
- `clone(data, astype=SomeGrid, **kwargs)` converts between grid types
- `asdataset()` context manager exposes a rasterio `MemoryFile` for rasterio API calls
- Default EPSG is 3857 (Web Mercator), resolution 1.0
- `H5Store` is read-only HDF5 → `DEMGrid` adapter, requires `x`, `y`, `Band1`, and `grid_mapping` attributes

## Testing quirks

- Tests use `DEMGrid.example(test=True)` → loads `src/demtools/data/testdem.tif`
- `--nbval-lax` validates notebook outputs; set `NBVAL_IGNORE_OUTPUT=1` env var to skip during active dev
- `conftest.py` provides a single `dem` fixture
- All tests are deterministic (no mocks, no external services)

## Notable dependencies

numpy, matplotlib, scipy, rasterio, affine, h5py, geojson, scikit-learn, colorcet, shapely, tqdm

## Session log — 2026-05-28 bug fix round

### Bug fixes applied to `src/demtools/grids.py`

| # | File | What | Detail |
|---|---|---|---|
| 1 | grids.py:317 | Fix `asbool` docstring | "Return grid as **BoolGrid**" (was FloatGrid) |
| 2 | grids.py:332 | Fix `asint` docstring | "Return grid as **IntGrid**" (was FloatGrid) |
| 3 | grids.py:347 | Fix `asfloat` docstring | "Return grid as **FloatGrid**" (was FloatGrid) |
| 4 | grids.py:704–705 | Fix `BoolGrid.count_false` | Use `len(self._values) - self.count_true` instead of `np.sum(~self._values)` |
| 5 | grids.py:897–907 | Rewrite `IntGrid.remove_dots` | Use `ndimage.label` + `bincount` (was broken: used `counts()` which returns pixel counts not region sizes) |
| 6 | grids.py:450 | Fix `_kernel` NaN on int arrays | Cast `n_sum` to float before setting NaN (was crashing IntGrid.moving_average) |
| 7 | grids.py:563–566 | Fix `aggregate("variance")` | Call `ndimage.variance` instead of `ndimage.standard_deviation` |
| 8 | grids.py:845–873 | Fix `IntGrid` division | Change `isinstance(other, DEMGrid)` → `isinstance(other, Grid)` in `__truediv__`, `__rtruediv__`, `__itruediv__` |
| 9 | grids.py:935–936 | Remove debug prints | Remove `print(a, type(a))` and `print("-------------------")` from `IntGrid.majority_filter` |
| 10 | grids.py:298,310 | Uncomment `stretch` in `clone()` | Restore `stretch=kwargs.get("stretch", self.stretch)` |
| — | mathlib.py:23 | Remove unused `shape` unpack | Remove `nx, ny = shape` from `_fftfreqs` |

### Bug #10 reclassified
`contour_label_kws` in `FloatGrid.show()` — the parameter IS already correctly passed to `rioplot.show` (lines 1355, 1375). Not a bug.

### Tests updated
- `test_aggregate_variance`: expect `np.var` instead of `np.std`
- `test_aggregate_std_vs_variance_same` → `test_aggregate_std_vs_variance_differ`: now asserts they differ
- `test_moving_average_raises_on_int` → `test_moving_average_int`: now asserts success (IntGrid.moving_average works)

## Session log — 2026-05-28 follow-up fixes

### Name mangling fix (`_dtype` / `_fill_value`)
**Root cause**: Python's `__` prefix mangles identifiers to `_ClassName__attr` at the *definition site*. Since `Grid.__init__` references `self.__class__.__fill_value` and `self.__class__.__dtype`, they resolve to `Grid._Grid__fill_value = None` / `Grid._Grid__dtype = None` in all subclasses.

**Fix**: Renamed all `__dtype` → `_dtype` and `__fill_value` → `_fill_value` (16 sites in `grids.py`). Subclass values (`IntGrid` → `-9999`/`int32`, `FloatGrid` → `nan`/`float64`, etc.) now propagate correctly to `meta["nodata"]` and `meta["dtype"]`.

### `__setitem__` BoolGrid fix
**Bug**: `Grid.__setitem__` used `mask._mask` when masking with BoolGrid. `_mask` returns the MaskedArray's mask (which entries are *unknown*), not the boolean data values. Changed to `mask._array` to match `__getitem__` behavior.

### New tests
- `test_setitem_with_boolgrid` — verifies BoolGrid mask works in `__setitem__`
- `test_meta_nodata_correct_per_subclass` — verifies each subclass gets correct nodata in meta

### Remaining issues fixed

| # | File | What | Detail |
|---|---|---|---|
| 11 | grids.py:84–86 | Fix `_mask` nomask | Return `np.zeros(...)` when `mask is ma.nomask` (was returning scalar `False`) |
| 12 | grids.py:316–359 | Fix `asbool`/`asint`/`asfloat` constants | Use target class's `_dtype`/`_fill_value` instead of `FloatGrid`'s |
| 13 | grids.py:898–909 | Rewrite `IntGrid.remove_dots` | Label per-value connected components (was labeling `> 0` as single class) |
| 14 | grids.py:1562 | Cast `tpi_fft` `n_sum` to float | Add `n_sum.astype(float)` before setting NaN (same pattern as `_kernel` fix) |
| 15 | mathlib.py:19 | Remove unused `shape` param | Remove `shape` from `_fftfreqs` signature (was unused after `nx, ny` unpack removed) |
| 16 | pyproject.toml:49 | Add `flake8`, `isort` to dev deps | Pre-commit runs these but they were missing from `[project.optional-dependencies]` |
