"""Choropleth maps of per-adm3 metrics using the adm3 boundary shapefile."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GDF_CACHE = {}


def _load_adm3_gdf(adm3_shapefile, adm3_name_col):
    key = (str(adm3_shapefile), adm3_name_col)
    if key not in _GDF_CACHE:
        import geopandas as gpd
        gdf = gpd.read_file(adm3_shapefile)
        if adm3_name_col not in gdf.columns:
            raise ValueError(
                f"Column '{adm3_name_col}' not found in {adm3_shapefile}"
            )
        gdf = gdf[[adm3_name_col, "geometry"]].rename(columns={adm3_name_col: "adm3"})
        gdf["adm3"] = gdf["adm3"].astype(str)
        _GDF_CACHE[key] = gdf
    return _GDF_CACHE[key]


def plot_adm3_choropleth(da, ax, gdf, *, title, cmap, vmin=None, vmax=None,
                         legend_label=None):
    """Draw one per-adm3 metric DataArray as a choropleth on the given axes."""
    values = pd.DataFrame({
        "adm3": np.asarray(da["adm3"].values, dtype=str),
        "value": np.asarray(da.values, dtype=float),
    })
    merged = gdf.merge(values, on="adm3", how="left")

    merged.plot(
        column="value", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={"shrink": 0.8, "label": legend_label or ""},
        edgecolor="grey", linewidth=0.2,
        missing_kwds={"color": "lightgrey", "label": "no data"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_adm3_spatial_metrics(spatial_metrics, *, case_name, adm3_shapefile,
                              adm3_name_col="adm3_name", dir_fig=".",
                              show_plot=False, figsize=(18, 6), **kwargs):
    """1x3 choropleth panel of mean MAE, FAR, and Miss Rate per adm3 unit.

    adm3-mode counterpart of momp.graphics.maps.plot_spatial_metrics.
    """
    gdf = _load_adm3_gdf(adm3_shapefile, adm3_name_col)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_adm3_choropleth(
        spatial_metrics["mean_mae"], axes[0], gdf,
        title="Mean MAE (days)", cmap="YlOrRd", legend_label="days",
    )
    plot_adm3_choropleth(
        spatial_metrics["false_alarm_rate"] * 100, axes[1], gdf,
        title="False Alarm Rate (%)", cmap="YlOrRd", vmin=0, vmax=100,
        legend_label="%",
    )
    plot_adm3_choropleth(
        spatial_metrics["miss_rate"] * 100, axes[2], gdf,
        title="Miss Rate (%)", cmap="YlOrRd", vmin=0, vmax=100,
        legend_label="%",
    )

    fig.suptitle(f"Onset metrics per adm3 unit — {case_name}")
    fig.tight_layout()

    fout = os.path.join(dir_fig, f"spatial_metrics_{case_name}.png")
    fig.savefig(fout, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {fout}")

    if show_plot:
        plt.show()
    plt.close(fig)
