"""Remap gridded lat-lon data onto admin-level-3 (adm3) polygon boundaries.

Ported from the onset_blending-adm3 project (utils/remap_weights.py,
utils/remap_weights_ngcm.py, utils/remap_nc.py) with two changes:

- grid cell sizes are derived dynamically from the coordinate spacing
  (per axis), so the same code serves fine and coarse grids;
- the output spatial dimension is named ``adm3`` with the district names as
  its (string) coordinate values.

Two stages:

1. ``build_grid_to_adm3_weights`` computes area-overlap weights between grid
   cells and adm3 polygons with geopandas (slow, done once and cached on disk
   as CSV via ``get_adm3_weights``).
2. ``aggregate_da_to_adm3`` applies the weights as a sparse-matrix product
   (fast, NaN-aware area-weighted mean per district).
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

# in-memory caches: weights DataFrames by cache-file path, and assembled
# sparse matrices by (grid, weights-file) key
_WEIGHTS_DF_CACHE = {}
_SPARSE_CACHE = {}


def _coord_key(lat, lon, precision=10):
    return (round(float(lat), precision), round(float(lon), precision))


def _half_deltas(values):
    """Half the spacing around each coordinate (handles irregular spacing)."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        raise ValueError("cannot infer cell size from a single coordinate value")
    deltas = np.abs(np.diff(values))
    half = np.empty_like(values)
    half[0] = deltas[0] / 2.0
    half[-1] = deltas[-1] / 2.0
    half[1:-1] = (deltas[:-1] + deltas[1:]) / 4.0
    return half


def build_grid_to_adm3_weights(shapefile, lats, lons, name_col="adm3_name",
                               min_weight=1e-5):
    """Area-overlap weights between grid cells and adm3 polygons.

    Returns a DataFrame with columns [lat, lon, adm3_name, weight] where
    weight is the fraction of the grid cell covered by the district.
    """
    import geopandas as gpd
    import shapely

    regions = gpd.read_file(shapefile)
    if name_col not in regions.columns:
        raise ValueError(
            f"Column '{name_col}' not found in {shapefile}; "
            f"available columns: {[c for c in regions.columns if c != 'geometry']}"
        )
    regions = regions[[name_col, "geometry"]].rename(columns={name_col: "adm3_name"})
    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")
    else:
        regions = regions.to_crs("EPSG:4326")

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lat_half = _half_deltas(lats)
    lon_half = _half_deltas(lons)

    lon2, lat2 = np.meshgrid(lons, lats)
    lonh2, lath2 = np.meshgrid(lon_half, lat_half)
    cell_boxes = shapely.box(
        (lon2 - lonh2).ravel(), (lat2 - lath2).ravel(),
        (lon2 + lonh2).ravel(), (lat2 + lath2).ravel(),
    )
    grid_gdf = gpd.GeoDataFrame(
        {"lat": lat2.ravel(), "lon": lon2.ravel()},
        geometry=cell_boxes, crs="EPSG:4326",
    )

    print(f"Computing grid-to-adm3 overlap for {len(grid_gdf)} cells x "
          f"{len(regions)} districts (this may take a while)...")
    import warnings
    with warnings.catch_warnings():
        # areas are used only as same-cell ratios, so the geographic-CRS area
        # bias cancels (same convention as the onset_blending-adm3 weights)
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        grid_gdf["cell_area"] = grid_gdf.geometry.area
        overlaid = gpd.overlay(grid_gdf, regions, how="intersection")
        overlaid["weight"] = overlaid.geometry.area / overlaid["cell_area"]
    overlaid = overlaid[overlaid["weight"] > min_weight]

    mapping = overlaid[["lat", "lon", "adm3_name", "weight"]].reset_index(drop=True)
    print(f"Built {len(mapping)} grid-cell/district weights "
          f"covering {mapping['adm3_name'].nunique()} districts")
    return mapping


def _weights_cache_path(lats, lons, shapefile, name_col, work_dir):
    tag = hashlib.md5()
    tag.update(np.round(np.asarray(lats, float), 6).tobytes())
    tag.update(np.round(np.asarray(lons, float), 6).tobytes())
    tag.update(str(shapefile).encode())
    tag.update(str(name_col).encode())
    try:
        stat = os.stat(shapefile)
        tag.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    except OSError:
        pass
    stem = Path(str(shapefile)).stem
    base = Path(str(work_dir)).expanduser() if work_dir else Path(".")
    return base / "adm3_weights" / f"{stem}_{tag.hexdigest()[:10]}.csv"


def get_adm3_weights(lats, lons, *, adm3_shapefile, adm3_name_col="adm3_name",
                     adm3_weights_cache=None, work_dir=None, **kwargs):
    """Load (or build and cache) the grid-to-adm3 weights for a grid."""
    if adm3_weights_cache:
        cache_path = Path(str(adm3_weights_cache)).expanduser()
    else:
        if not adm3_shapefile:
            raise ValueError(
                "benchmark_space='adm3' requires 'adm3_shapefile' (or an "
                "existing 'adm3_weights_cache' CSV) in the config"
            )
        cache_path = _weights_cache_path(lats, lons, adm3_shapefile,
                                         adm3_name_col, work_dir)

    cache_key = str(cache_path)
    if cache_key in _WEIGHTS_DF_CACHE:
        return _WEIGHTS_DF_CACHE[cache_key]

    if cache_path.exists():
        mapping = pd.read_csv(cache_path)
        mapping = mapping.rename(columns={"latitude": "lat", "longitude": "lon"})
    else:
        mapping = build_grid_to_adm3_weights(
            adm3_shapefile, lats, lons, name_col=adm3_name_col
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(cache_path, index=False)
        print(f"adm3 weights cached to: {cache_path}")

    _WEIGHTS_DF_CACHE[cache_key] = mapping
    return mapping


def _build_sparse_weights(mapping, lat_values, lon_values):
    """CSR matrix (n_lat*n_lon, n_adm3) of grid-cell-to-district weights."""
    pixel_lookup = {
        _coord_key(lat, lon): idx
        for idx, (lat, lon) in enumerate(
            (lat, lon) for lat in lat_values for lon in lon_values
        )
    }

    adm3_names = pd.Index(mapping["adm3_name"].astype(str).drop_duplicates())
    adm3_lookup = {name: idx for idx, name in enumerate(adm3_names)}

    rows, cols, data = [], [], []
    missing_pixels = 0
    for rec in mapping.itertuples(index=False):
        pixel_idx = pixel_lookup.get(_coord_key(rec.lat, rec.lon))
        if pixel_idx is None:
            missing_pixels += 1
            continue
        rows.append(pixel_idx)
        cols.append(adm3_lookup[str(rec.adm3_name)])
        data.append(float(rec.weight))

    if not data:
        raise ValueError("No adm3 weight rows matched the input lat/lon grid; "
                         "the weights were probably built for a different grid.")
    if missing_pixels:
        print(f"  Skipped {missing_pixels} weight rows with no matching grid cell.")

    weights = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(lat_values) * len(lon_values), len(adm3_names)),
    )
    return weights, adm3_names.to_numpy()


def _sparse_weights_for_grid(mapping, lat_values, lon_values):
    key = (
        np.round(np.asarray(lat_values, float), 6).tobytes(),
        np.round(np.asarray(lon_values, float), 6).tobytes(),
        id(mapping),
    )
    if key not in _SPARSE_CACHE:
        _SPARSE_CACHE[key] = _build_sparse_weights(mapping, lat_values, lon_values)
    return _SPARSE_CACHE[key]


def aggregate_da_to_adm3(da, mapping):
    """NaN-aware area-weighted mean of a (…, lat, lon) DataArray per district."""
    weights, adm3_names = _sparse_weights_for_grid(mapping, da["lat"].values, da["lon"].values)

    spatial_dims = ["lat", "lon"]
    other_dims = [dim for dim in da.dims if dim not in spatial_dims]
    ordered = da.transpose(*other_dims, *spatial_dims)

    other_shape = tuple(ordered.sizes[dim] for dim in other_dims)
    values = np.asarray(ordered.values)
    flat = values.reshape((-1, ordered.sizes["lat"] * ordered.sizes["lon"]))

    valid = np.isfinite(flat)
    weighted_sum = weights.T.dot(np.nan_to_num(flat, nan=0.0).T).T
    effective_weight = weights.T.dot(valid.astype(float).T).T

    with np.errstate(invalid="ignore", divide="ignore"):
        out = weighted_sum / effective_weight
    out[effective_weight <= 0] = np.nan
    out = np.asarray(out).reshape(other_shape + (len(adm3_names),))

    coords = {dim: da.coords[dim] for dim in other_dims if dim in da.coords}
    # fixed-width unicode (not object) dtype so NetCDF encoding works
    coords["adm3"] = np.asarray(adm3_names, dtype=str)
    return xr.DataArray(out, dims=other_dims + ["adm3"], coords=coords, name=da.name)


def maybe_to_adm3(obj, *, benchmark_space=None, adm3_shapefile=None,
                  adm3_name_col="adm3_name", adm3_weights_cache=None,
                  work_dir=None, **kwargs):
    """Remap a Dataset/DataArray to adm3 units when benchmark_space == 'adm3'.

    No-op for grid benchmarking, scalars, and objects without lat/lon dims
    (e.g. data already on adm3 units).
    """
    if benchmark_space != "adm3":
        return obj
    if np.isscalar(obj) or not isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj
    if "lat" not in obj.dims or "lon" not in obj.dims:
        return obj

    mapping = get_adm3_weights(
        obj["lat"].values, obj["lon"].values,
        adm3_shapefile=adm3_shapefile, adm3_name_col=adm3_name_col,
        adm3_weights_cache=adm3_weights_cache, work_dir=work_dir,
    )

    if isinstance(obj, xr.DataArray):
        return aggregate_da_to_adm3(obj, mapping)

    remapped = {}
    for var_name in obj.data_vars:
        da = obj[var_name]
        if "lat" in da.dims and "lon" in da.dims:
            remapped[var_name] = aggregate_da_to_adm3(da, mapping)
    return xr.Dataset(remapped)
