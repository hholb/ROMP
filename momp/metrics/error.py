#from momp.stats.benchmark import
import numpy as np
import xarray as xr
import pandas as pd

from momp.utils.standard import loc_cols


def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
    """Create spatial maps of False Alarm Rate, Miss Rate, yearly MAE, and mean MAE across years."""
    first_year = list(onset_da_dict.keys())[0]
    first_da = onset_da_dict[first_year]
    dims = list(first_da.dims)
    coords = {dim: first_da[dim].values for dim in dims}

    print(f"Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
    print(f"Spatial dimensions: " + " x ".join(f"{len(coords[d])} {d}s" for d in dims))
    print(f"Years: {list(metrics_df_dict.keys())}")

    loc = loc_cols(next(iter(metrics_df_dict.values())))

    def to_spatial_da(frame, col, name, attrs):
        """Series of per-location values -> DataArray on the onset_da grid."""
        if len(frame) == 0:
            data = np.full([len(coords[d]) for d in dims], np.nan)
            return xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=attrs)
        da = frame.set_index(loc)[col].astype(float).to_xarray()
        return da.reindex(coords).rename(name).assign_attrs(attrs)

    # combine all years, keeping only locations with a valid observed onset
    # that year (matches the original per-point NaN skip)
    frames = []
    for year, metrics_df in metrics_df_dict.items():
        obs_df = onset_da_dict[year].to_dataframe(name="obs_val").reset_index()
        merged = metrics_df.merge(obs_df[loc + ["obs_val"]], on=loc, how="left")
        merged = merged[merged["obs_val"].notna()].copy()
        merged["year"] = year
        frames.append(merged)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=loc)

    spatial_metrics = {}

    if len(all_df) > 0:
        agg = all_df.groupby(loc, sort=False).agg(
            FP=("false_positive", "sum"),
            TN=("true_negative", "sum"),
            FN=("false_negative", "sum"),
            num_onset=("num_onset", "sum"),
            mean_mae=("mae_combined", "mean"),
        ).reset_index()
        with np.errstate(invalid="ignore", divide="ignore"):
            agg["false_alarm_rate"] = np.where(
                (agg["FP"] + agg["TN"]) > 0, agg["FP"] / (agg["FP"] + agg["TN"]), 0.0
            )
            agg["miss_rate"] = np.where(
                agg["num_onset"] > 0, agg["FN"] / agg["num_onset"], 0.0
            )
    else:
        agg = pd.DataFrame(columns=loc + ["false_alarm_rate", "miss_rate", "mean_mae"])

    spatial_metrics["false_alarm_rate"] = to_spatial_da(
        agg, "false_alarm_rate", "false_alarm_rate",
        {"description": "False Alarm Rate = sum(FP) / sum(FP + TN) across all valid years"},
    )
    spatial_metrics["miss_rate"] = to_spatial_da(
        agg, "miss_rate", "miss_rate",
        {"description": "Miss Rate = sum(FN) / sum(total_onsets) across all valid years"},
    )
    spatial_metrics["mean_mae"] = to_spatial_da(
        agg, "mean_mae", "mean_mae",
        {"description": "Mean MAE across all valid years (omitting NaN values)"},
    )

    for year, frame in zip(metrics_df_dict.keys(), frames):
        spatial_metrics[f"mae_{year}"] = to_spatial_da(
            frame, "mae_combined", f"mae_{year}",
            {"description": f"Mean Absolute Error for year {year}"},
        )

    return spatial_metrics
