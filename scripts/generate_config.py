"""
Generate a ROMP config.in from environment variables.

Required env vars:
  ROMP_OBS_DIR          - path to observation NetCDF files
  ROMP_MODEL_DIR        - path to forecast model NetCDF files
  ROMP_MODEL_NAME       - model identifier (e.g. AIFS)
  ROMP_DIR_OUT          - output directory for CSVs / NetCDFs
  ROMP_DIR_FIG          - output directory for figures

Optional env vars (with defaults matching the demo config):
  ROMP_OBS              - observation dataset name (default: CHIRPS_IMERG)
  ROMP_OBS_FILE_PATTERN - file naming pattern (default: {}.nc)
  ROMP_OBS_VAR          - rainfall variable name in obs file (default: RAINFALL)
  ROMP_MODEL_VAR        - rainfall variable name in forecast file (default: tp)
  ROMP_FILE_PATTERN     - forecast file naming pattern (default: {}.nc)
  ROMP_REGION           - target region (default: Ethiopia)
  ROMP_NC_MASK          - path to land/region mask NetCDF (default: None)
  ROMP_THRESH_FILE      - path to spatial threshold NetCDF (default: None)
  ROMP_WET_THRESHOLD    - scalar rainfall threshold mm (default: 20)
  ROMP_WET_INIT         - minimum wet day threshold mm (default: 1)
  ROMP_WET_SPELL        - wet spell length days (default: 3)
  ROMP_DRY_SPELL        - dry spell length days (default: 7)
  ROMP_DRY_EXTENT       - dry spell search window days (default: 0)
  ROMP_ONSET_RULE       - onset definition name from momp.stats.onset_rule (default: legacy)
  ROMP_ONSET_RULE_PARAMS- JSON dict of parameters for ROMP_ONSET_RULE (default: rule defaults)
  ROMP_START_DATE       - evaluation start as YYYY-MM-DD (default: 2019-05-01)
  ROMP_END_DATE         - evaluation end as YYYY-MM-DD (default: 2024-07-31)
  ROMP_START_YEAR_CLIM  - climatology start year (default: 1998)
  ROMP_END_YEAR_CLIM    - climatology end year (default: 2024)
  ROMP_MAX_FORECAST_DAY - max forecast lead day (default: 30)
  ROMP_PROBABILISTIC    - True/False (default: False)
  ROMP_MEMBERS          - ensemble members, All or comma-separated ints (default: All)
  ROMP_PARALLEL         - True/False (default: True)
  ROMP_REF_MODEL        - reference model name (default: climatology)
  ROMP_REF_MODEL_DIR    - reference model data dir (default: same as obs dir)
  ROMP_INIT_DAYS        - forecast init weekdays as comma-sep ints (0=Mon, default: 0,3)
  ROMP_DATE_FILTER_YEAR - reference year for init-day calendar alignment (default: start_date year)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return val


def opt(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_date(val: str) -> tuple:
    dt = datetime.strptime(val, "%Y-%m-%d")
    return (dt.year, dt.month, dt.day)


def main():
    obs_dir     = require("ROMP_OBS_DIR")
    model_dir   = require("ROMP_MODEL_DIR")
    model_name  = require("ROMP_MODEL_NAME")
    dir_out     = require("ROMP_DIR_OUT")
    dir_fig     = require("ROMP_DIR_FIG")

    obs             = opt("ROMP_OBS",              "CHIRPS_IMERG")
    obs_file_pat    = opt("ROMP_OBS_FILE_PATTERN", "{}.nc")
    obs_var         = opt("ROMP_OBS_VAR",          "RAINFALL")
    model_var       = opt("ROMP_MODEL_VAR",        "tp")
    file_pattern    = opt("ROMP_FILE_PATTERN",     "{}.nc")
    region          = opt("ROMP_REGION",           "Ethiopia")
    nc_mask_raw     = opt("ROMP_NC_MASK",          "")
    thresh_file_raw = opt("ROMP_THRESH_FILE",      "")
    wet_threshold   = opt("ROMP_WET_THRESHOLD",    "20")
    wet_init        = opt("ROMP_WET_INIT",         "1")
    wet_spell       = opt("ROMP_WET_SPELL",        "3")
    dry_spell       = opt("ROMP_DRY_SPELL",        "7")
    dry_extent      = opt("ROMP_DRY_EXTENT",       "0")
    onset_rule      = opt("ROMP_ONSET_RULE",       "legacy")
    onset_rule_params_raw = opt("ROMP_ONSET_RULE_PARAMS", "")
    start_date_str  = opt("ROMP_START_DATE",       "2019-05-01")
    end_date_str    = opt("ROMP_END_DATE",         "2024-07-31")
    start_yr_clim   = opt("ROMP_START_YEAR_CLIM",  "1998")
    end_yr_clim     = opt("ROMP_END_YEAR_CLIM",    "2024")
    max_fc_day      = opt("ROMP_MAX_FORECAST_DAY", "30")
    probabilistic   = opt("ROMP_PROBABILISTIC",    "False")
    members         = opt("ROMP_MEMBERS",          "All")
    parallel        = opt("ROMP_PARALLEL",         "True")
    ref_model       = opt("ROMP_REF_MODEL",        "climatology")
    ref_model_dir   = opt("ROMP_REF_MODEL_DIR",    obs_dir)
    init_days_raw      = opt("ROMP_INIT_DAYS",         "0,3")
    date_filter_year_s = opt("ROMP_DATE_FILTER_YEAR",  "")

    start_date = parse_date(start_date_str)
    end_date   = parse_date(end_date_str)

    date_filter_year = int(date_filter_year_s) if date_filter_year_s else start_date[0]

    if onset_rule_params_raw.strip():
        parsed = json.loads(onset_rule_params_raw)
        if not isinstance(parsed, dict):
            print("ERROR: ROMP_ONSET_RULE_PARAMS must be a JSON object", file=sys.stderr)
            sys.exit(1)
        onset_rule_params = repr(parsed)
    else:
        onset_rule_params = "None"

    init_days = "(" + ", ".join(init_days_raw.split(",")) + ",)"

    nc_mask     = f'"{nc_mask_raw}"'     if nc_mask_raw     else "None"
    thresh_file = f'"{thresh_file_raw}"' if thresh_file_raw else "None"

    # members: "All" stays as string; "0,1,2" becomes a tuple of ints
    if members.strip().lower() == "all":
        members_val = '"All"'
    else:
        ints = ", ".join(members.split(","))
        members_val = f"({ints},)"

    config = f"""
project_name = "ROMP container run"
work_dir = "{dir_out}"
pkg_dir = "/app"

layout = ("model", "verification_window")

model_list = ("{model_name}",)

obs = "{obs}"
obs_dir = "{obs_dir}"
obs_file_pattern = ("{obs_file_pat}",)
obs_var = "{obs_var}"
obs_unit_cvt = None

ref_model = "{ref_model}"
ref_model_dir = "{ref_model_dir}"
ref_model_file_pattern = "{obs_file_pat}"
ref_model_var = "{obs_var}"
ref_model_unit_cvt = None

model_dir_list = ("{model_dir}",)
model_var_list = ("{model_var}",)
unit_cvt_list = (None,)
file_pattern_list = ("{file_pattern}",)

region = "{region}"
nc_mask = {nc_mask}
shpfile_dir = None
polygon = False

wet_init = {wet_init}
wet_threshold = {wet_threshold}
wet_spell = {wet_spell}
dry_threshold = 1
dry_spell = {dry_spell}
dry_extent = {dry_extent}
thresh_file = {thresh_file}
thresh_var = None
onset_percentage_threshold = 0.5
onset_rule = "{onset_rule}"
onset_rule_params = {onset_rule_params}

start_date = {start_date}
end_date = {end_date}
start_year_clim = {start_yr_clim}
end_year_clim = {end_yr_clim}
init_days = {init_days}
date_filter_year = {date_filter_year}

verification_window_list = ((1, 15), (16, 30))
tolerance_days_list = (3, 5)
max_forecast_day = {max_fc_day}
day_bins = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))

FAR = True
MAE = True
MR  = True

probabilistic = {probabilistic}
members = {members_val}

BS          = True
RPS         = True
AUC         = True
Reliability = True
skill_score = True

dir_out = "{dir_out}"
dir_fig = "{dir_fig}"

save_fig = True
save_nc_spatial_far_mr_mae = True
save_csv_score = True
save_nc_climatology = True

plot_spatial_far_mr_mae    = True
plot_heatmap_bss_auc       = True
plot_reliability           = True
plot_climatology_onset     = True
plot_panel_heatmap_error   = True
plot_panel_heatmap_skill   = True
plot_bar_bss_rpss_auc      = True

show_plot  = False
show_panel = False

parallel = {parallel}

debug = False
"""

    output_path = os.environ.get("ROMP_CONFIG_PATH", "/tmp/romp_job.in")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(config)
    print(f"Config written to {output_path}")


if __name__ == "__main__":
    main()
