import numpy as np
from matplotlib.path import Path
#from pathlib import Path
from typing import Union
import regionmask
import xarray as xr
from momp.params.region_def import polygon_boundary
from matplotlib.patches import Polygon
from momp.utils.standard import dim_fmt


# Caches for grid-static objects (land-sea mask, country geometry/mask, nc mask).
# These are rebuilt identically on every region_select call otherwise, which is
# expensive for the regionmask/shapely computations. Keys include the rounded
# grid coordinates so a different grid never reuses a stale mask.
_STATIC_MASK_CACHE = {}


def _grid_cache_key(lat_values, lon_values):
    lat = np.round(np.asarray(lat_values, dtype=float), 6)
    lon = np.round(np.asarray(lon_values, dtype=float), 6)
    return (lat.tobytes(), lon.tobytes())


# Function to find grid points inside a polygon (For core-monsoon zone analysis)
def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """
    Find grid points that are inside a polygon.

    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points

    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """

    #print("\n polygon_lon = ", polygon_lon)
    #print("\n grid_lons = ", grid_lons)
    #print(type(polygon_lon))

    # Create polygon path
    #polygon_path = Path(list(zip(polygon_lon, polygon_lat)))
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    #print("\npolygon_vertices = ", polygon_vertices)
    polygon_path = Path(polygon_vertices)

    #print("\npolygon_path = ", polygon_path)

    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)

    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]

    return inside_mask, inside_lons, inside_lats



def polygon_mask(da_model):
    """mask data based on polygon boundary"""

    orig_lat = da_model.lat.values
    orig_lon = da_model.lon.values

    polygon1_lat, polygon1_lon = polygon_boundary(da_model)

    inside_mask, inside_lons, inside_lats = points_inside_polygon(polygon1_lon, polygon1_lat, orig_lon, orig_lat)

    #da_model_slice = da_model.sel(lat=inside_lats, lon=inside_lons) # result in different dim size
    # xarray matches all lats in inside_lats with all lons in inside_lons, forming a cartesian product.
    # so output values are repeated multiple times — every selected latitude is paired with 
    # every selected longitude.

    da_model_slice = da_model.where(inside_mask)

    return da_model_slice



def polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=1.25):
    """ add polygon boundary to basemap """
    #from matplotlib.patches import Polygon
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as ShapelyPolygon
    from cartopy.feature import ShapelyFeature
    
    # Cartopy gridlines create a separate layer that can cover patches added with ax.add_patch()
    # use a ShapelyFeature and add it with ax.add_feature() with a higher zorder 
    # so it appears above gridlines and map features
    poly_geom = ShapelyPolygon(list(zip(polygon1_lon, polygon1_lat)))
    
    # wrap as a Cartopy feature
    poly_feature = ShapelyFeature(
        [poly_geom],
        crs=ccrs.PlateCarree(),  # coordinates are lon/lat
        edgecolor='black',
        facecolor='none',
        linewidth=linewidth
    )
    
    # add to axes above gridlines
    ax.add_feature(poly_feature, zorder=10)


    # the code below doesn't work with gridline set
    # Cartopy gridlines create a separate layer that can cover patches added with ax.add_patch()
    #polygon_lines = Polygon(list(zip(polygon1_lon, polygon1_lat)),
    #                 fill=False, edgecolor='black', linewidth=linewidth, transform=ccrs.PlateCarree())
    #ax.add_patch(polygon_lines)

    return ax



def add_polygon(ax, da, polygon, return_polygon=False, linewidth=1.25):
    #from matplotlib.patches import Polygon
    #from momp.params.region_def import polygon_boundary

    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)

        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    # Add CMZ polygon only if defined
    if polygon_defined:

        #polygon_lines = Polygon(list(zip(polygon1_lon, polygon1_lat)),
        #                 fill=False, edgecolor='black', linewidth=linewidth)
        #ax.add_patch(polygon_lines)

        ax = polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=linewidth)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    else:
        return ax



def get_india_outline(shpfile_path):
    """
    Get region outline coordinates from shapefile.
    """
    import geopandas as gpd
    # Update this path to your India shapefile
    india_gdf = gpd.read_file(shpfile_path)

    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, 'geoms'):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries



def create_land_sea_mask(
    obj: Union[xr.Dataset, xr.DataArray],
    as_boolean: bool = False,
) -> xr.DataArray:
    """Generate a land-sea mask (1 for land, 0 for sea) for a given xarray Dataset or DataArray.
    stemed from pcmdi_metrics.utils.land_sea_mask 

    Parameters
    ----------
    obj : Union[xr.Dataset, xr.DataArray]
        The Dataset or DataArray object.
    as_boolean : bool, optional
        Set mask value to True (land) or False (ocean), by default False, thus 1 (land) and 0 (ocean).

    Returns
    -------
    xr.DataArray
        A DataArray of land-sea mask (1 or 0 for land or sea, or True or False for land or sea).

    Examples
    --------
    >>> mask = create_land_sea_mask(ds)  #  Generate land-sea mask (land: 1, sea: 0)
    >>> mask = create_land_sea_mask(ds, as_boolean=True)  # Generate land-sea mask (land: True, sea: False)
    """

    lon = obj["lon"]
    lat = obj["lat"]

    cache_key = ("land_sea", _grid_cache_key(lat.values, lon.values), as_boolean)
    cached = _STATIC_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Use regionmask
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    # Mask the land-sea mask to match the dataset's coordinates
    land_sea_mask = land_mask.mask(lon, lat=lat)

    if as_boolean:
        # Convert the 0 (land) & nan (ocean) land-sea mask to a boolean mask
        land_sea_mask = xr.where(land_sea_mask, False, True)
    else:
        # Convert the boolean land-sea mask to a 1/0 mask
        land_sea_mask = xr.where(land_sea_mask, 0, 1)

    _STATIC_MASK_CACHE[cache_key] = land_sea_mask
    return land_sea_mask


def mask_land(da, land=True):
    """
    Mask a DataArray to select either land or sea areas.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be masked.
    land : bool, optional
        If True, keep land areas (default). If False, keep sea areas.

    Returns
    -------
    xarray.DataArray
        Masked DataArray with values retained for land or sea, NaNs elsewhere.
    """

    land_sea_mask = create_land_sea_mask(da)

    if land:
        da_masked = da.where(land_sea_mask == 1)
    else:
        da_masked = da.where(land_sea_mask == 0)

    return da_masked


def get_shp(region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ Create country boundaries"""

    import cartopy.io.shapereader as shpreader

    cache_key = ("shp_geom", region, resolution, category, name)
    if cache_key in _STATIC_MASK_CACHE:
        return _STATIC_MASK_CACHE[cache_key]

    # Load Ethiopia shapefile
    ethiopia_shp = shpreader.natural_earth(resolution=resolution, category=category, name=name)

    # Find Ethiopia geometry
    region_geom = None
    for country in shpreader.Reader(ethiopia_shp).records():
        if country.attributes['NAME'] == region:
            region_geom = country.geometry
            break

    _STATIC_MASK_CACHE[cache_key] = region_geom
    return region_geom


def shp_mask(da, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries', 
             return_mask=False, **kwargs):
    """ Create mask based on country boundaries"""
    from shapely.vectorized import contains

    # get region boundary
    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        print("    WARNING - specified region is not in cartopy.io.shapereader")
        return da

    cache_key = ("shp_mask", region, resolution, category, name,
                 _grid_cache_key(da.lat.values, da.lon.values))
    mask_da = _STATIC_MASK_CACHE.get(cache_key)
    if mask_da is None:
        # Create mask for Ethiopia
        lons, lats = np.meshgrid(da.lon, da.lat)
        points = np.column_stack((lons.ravel(), lats.ravel()))
        mask = contains(region_geom, points[:, 0], points[:, 1])
        mask = mask.reshape(lons.shape)
        mask_da = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': da.lat, 'lon': da.lon})
        _STATIC_MASK_CACHE[cache_key] = mask_da

    # Apply mask to data
    da_masked = da.where(mask_da)

    #return da_masked, mask_da

    if return_mask:
        return da_masked, mask_da
    else:
        return da_masked


def shp_outline(ax, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ add shapefile country boundaries to plot"""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # get region boundary
    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        print("WARNING - specified region is not in cartopy.io.shapereader")
        return ax

    # Plot boundary
#    ax.add_geometries(
#        [region_geom],
#        crs=ccrs.PlateCarree(),
#        facecolor="none",
#        edgecolor="black",
#        linewidth=1.5
#    )
    
    if hasattr(ax, "add_geometries"):
        # Cartopy GeoAxes
        ax.add_geometries(
            [region_geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=10
        )
    else:
        # Regular matplotlib Axes — handle both Polygon and MultiPolygon
        geoms = [region_geom] if region_geom.geom_type == 'Polygon' else list(region_geom.geoms)
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.5, zorder=10)

    # Optional: background
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

    return ax



def apply_nc_mask(ds, nc_mask, mask_var=None, keep_value=1):
    """
    Mask an xarray Dataset/DataArray using a 0/1 mask stored in a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Data to mask.
    nc_mask : str
        Path to NetCDF mask file (e.g., "land.nc").
    mask_var : str, optional
        Variable name inside mask file. If None, auto-pick if only one var exists.
    keep_value : int or float, default 1
        Which value means "keep". Commonly 1 (keep) and 0 (mask out).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Masked data (values where mask is False become NaN).
    """

    cache_key = ("nc_mask", str(nc_mask), mask_var, keep_value,
                 _grid_cache_key(ds.lat.values, ds.lon.values))
    mask_bool = _STATIC_MASK_CACHE.get(cache_key)

    if mask_bool is None:
        mask_ds = xr.open_dataset(nc_mask)
        try:
            mask_ds = dim_fmt(mask_ds)
            mask_ds = mask_ds.sel(lat=ds.lat, lon=ds.lon, method="nearest")

            # pick mask variable
            if mask_var is None:
                if len(mask_ds.data_vars) != 1:
                    raise ValueError(
                        f"Mask file has multiple variables {list(mask_ds.data_vars)}; "
                        "please specify mask_var."
                    )
            mask = next(iter(mask_ds.data_vars.values())) if mask_var is None else mask_ds[mask_var]

            # convert 0/1 (or numeric) to boolean
            if mask.dtype == bool:
                mask_bool = mask
            else:
                mask_bool = (mask == keep_value)

            mask_bool = mask_bool.load()
            _STATIC_MASK_CACHE[cache_key] = mask_bool
        finally:
            mask_ds.close()

    # align to prevent accidental broadcasting (and catch mismatched grids)
    mask_bool, ds_aligned = xr.align(mask_bool, ds, join="exact")

    # xarray broadcasts by dimension name, so this works regardless of the
    # dimension order of ds (obs time-first vs forecast lat-first data).
    return ds_aligned.where(mask_bool)

