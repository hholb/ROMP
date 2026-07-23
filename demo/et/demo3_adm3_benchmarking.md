# Demo 3. Benchmarking on adm3 (woreda) boundaries

ROMP can evaluate onset forecasts on admin-level-3 polygons instead of the
native lat-lon grid. Rainfall (observations, forecasts, and any spatial
threshold file) is remapped to each district as an area-weighted mean right
after loading, and onset detection plus all metrics then run per district.
This matches the methodology of the operational `onset_blending-adm3`
pipeline.

## How the remapping works

1. **Weights (one-time).** For a given grid + shapefile combination, ROMP
   intersects every grid cell with every district polygon (geopandas) and
   stores `weight = intersection_area / cell_area` for each overlapping pair.
   The weights are written to `{work_dir}/adm3_weights/<shapefile>_<hash>.csv`
   and reused on every later run — the hash covers the grid coordinates and
   the shapefile, so a different grid or an updated shapefile rebuilds them
   automatically.
2. **Aggregation (every run, fast).** The weights are applied as a sparse
   matrix product: each district value is the NaN-aware weighted mean of the
   grid cells it overlaps. Cells masked to NaN (land/sea, country boundary)
   simply drop out of the mean.

Because observations and forecasts share one grid in ROMP, a single weight
matrix serves obs, forecast, and threshold data.

## Config keys

```
benchmark_space    = "adm3"   # "grid" (default) or "adm3"
adm3_shapefile     = "demo/data/complete_data/shapefile/manual_zones_woredas_nobox4.shp"
adm3_name_col      = "adm3_name"   # attribute column holding district names
adm3_weights_cache = None          # optional: point at an existing weights CSV
                                   # (columns: lat, lon, adm3_name, weight)
```

Everything else in the config keeps its usual meaning.

## Run the demo

From the repository root:

```bash
momp-run -p demo/et/config_et_adm3.in
```

This benchmarks the deterministic AIFS forecast
(`demo/data/complete_data/aifs_single_v1`) against CHIRPS_SAT observations
for 2015-2016, with a climatology reference, using the ICPAC onset definition
(wet_init=1, wet_threshold=20, wet_spell=3, dry_spell=7, dry_extent=21).

Outputs land in `demo/et/output_adm3/` and `demo/et/figure_adm3/`:

- `spatial_metrics_<case>.nc` — FAR / Miss Rate / mean MAE / per-year MAE on a
  1-D `adm3` dimension (district names as coordinates)
- `spatial_metrics_<case>.csv` — the same as a per-district table
  (`adm3_name, false_alarm_rate, miss_rate, mean_mae, mae_2015, mae_2016`)
- `spatial_metrics_<case>.png` — choropleth panel of MAE / FAR / MR per woreda
  (grey districts = no valid data)

## Probabilistic (ensemble) run

Point the model at an ensemble dataset and enable probabilistic metrics —
three lines change:

```
model_list = ("AIFS_ens",)
model_dir_list = ("demo/data/complete_data/aifs_ens_v1", )
probabilistic = True
```

This produces `overall_skill_scores_*.csv` and `binned_skill_scores_*.csv`
(Fair Brier, Fair RPS, AUC vs the climatological reference) pooled over all
districts, plus reliability and skill heatmap figures. `gencast` (32 members)
and `ngcm_ethiopia` work the same way.

## Notes and caveats

- District-mean-rainfall onset is not the same quantity as per-cell onset
  aggregated afterwards; results are per-woreda by construction.
- The demo forecast files carry 46 lead days, so with `dry_extent = 21` the
  maximum usable `max_forecast_day` is 25.
- The `complete_data` init times fall on irregular weekdays; the demo config
  keeps them all with `init_days = (0, 1, 2, 3, 4, 5, 6)`.
- `plot_climatology_onset` draws a lat-lon map and is skipped in adm3 mode.
- The first run builds the weights (~10 s for the 0.25 deg Ethiopia grid x 293
  woredas); later runs load the cached CSV.
