from momp.io.input import load_thresh_file, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly, get_forecast_deterministic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_deterministic_model, compute_onset_for_all_members
from momp.stats.climatology import compute_climatological_onset, compute_climatology_as_forecast
from momp.utils.practical import restore_args
from momp.utils.printing import tuple_to_str_range
from momp.utils.standard import loc_cols
#from momp.stats.parallel import parallel_climatological_onset

import numpy as np
import pandas as pd
import os
import sys


def compute_onset_metrics_with_windows(onset_df, *, tolerance_days, verification_window, **kwargs):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows."""
    print(f"Computing onset metrics with tolerance = {tolerance_days} days")
    print(f"Verification window starts {verification_window} days after initialization")
    print(f"Forecast window length: {verification_window[1]} days")

    #forecast_bin_start = kwargs["forecast_bin"][0]
    #forecast_bin_end = kwargs["forecast_bin"][1]
    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]

    loc = loc_cols(onset_df)

    df = onset_df[loc + ['init_time', 'onset_date', 'obs_onset_date']].copy()
    df['obs_onset_dt'] = pd.to_datetime(df['obs_onset_date'])
    df['model_onset_dt'] = pd.to_datetime(df['onset_date'])
    df['init_dt'] = pd.to_datetime(df['init_time'])

    print(f"Processing {len(df[loc].drop_duplicates())} unique locations...")

    # per-location observed onset (constant within a location-year)
    gt = df.groupby(loc, sort=False)['obs_onset_dt'].transform('first')

    valid_window_start = df['init_dt'] + pd.Timedelta(days=forecast_bin_start)
    valid_window_end = df['init_dt'] + pd.Timedelta(days=forecast_bin_end)
    whole_window_start = df['init_dt'] + pd.Timedelta(days=1)
    whole_window_end = valid_window_end

    is_obs_onset_in_whole_window = (whole_window_start <= gt) & (gt <= whole_window_end)

    has_model_onset = df['model_onset_dt'].notna()
    is_model_in_valid_window = (
        has_model_onset
        & (valid_window_start <= df['model_onset_dt'])
        & (df['model_onset_dt'] <= valid_window_end)
    )
    abs_diff_days = (df['model_onset_dt'] - gt).dt.days.abs()

    df['tp'] = is_model_in_valid_window & (abs_diff_days <= tolerance_days)
    df['fp'] = is_model_in_valid_window & ~df['tp']
    # model onset after the valid window, or no model onset at all, count as
    # FN/TN; an onset before the valid window start counts as neither
    miss_like = (~has_model_onset) | (has_model_onset & ~is_model_in_valid_window
                                      & (df['model_onset_dt'] > valid_window_end))
    df['fn'] = miss_like & is_obs_onset_in_whole_window
    df['tn'] = miss_like & ~is_obs_onset_in_whole_window
    df['num_onset_flag'] = is_obs_onset_in_whole_window
    df['mae_val'] = abs_diff_days.where(df['tp'] | df['fp'])
    df['mae_tp_val'] = abs_diff_days.where(df['tp'])

    grouped = df.groupby(loc, sort=False)
    metrics_df = grouped.agg(
        total_forecasts=('init_dt', 'size'),
        true_positive=('tp', 'sum'),
        true_negative=('tn', 'sum'),
        false_positive=('fp', 'sum'),
        false_negative=('fn', 'sum'),
        num_onset=('num_onset_flag', 'sum'),
        mae_combined=('mae_val', 'mean'),
        mae_tp_only=('mae_tp_val', 'mean'),
        num_fp_errors=('fp', 'sum'),
    ).reset_index()

    metrics_df['num_no_onset'] = metrics_df['total_forecasts'] - metrics_df['num_onset']
    metrics_df['num_tp_errors'] = metrics_df['true_positive']
    metrics_df['tolerance_days'] = tolerance_days
    metrics_df['verification_window'] = [verification_window] * len(metrics_df)
    metrics_df['forecast_days'] = forecast_bin_end

    metrics_df = metrics_df[loc + [
        'total_forecasts', 'true_positive', 'true_negative', 'false_positive',
        'false_negative', 'num_onset', 'num_no_onset', 'mae_combined',
        'mae_tp_only', 'num_tp_errors', 'num_fp_errors', 'tolerance_days',
        'verification_window', 'forecast_days'
    ]]

    summary_stats = {
        'total_grid_points': len(metrics_df),
        'total_forecasts': metrics_df['total_forecasts'].sum(),
        'overall_true_positive': metrics_df['true_positive'].sum(),
        'overall_true_negative': metrics_df['true_negative'].sum(),
        'overall_false_positive': metrics_df['false_positive'].sum(),
        'overall_false_negative': metrics_df['false_negative'].sum(),
        'overall_num_onset': metrics_df['num_onset'].sum(),
        'overall_num_no_onset': metrics_df['num_no_onset'].sum(),
        'overall_mae_combined': metrics_df['mae_combined'].mean(),
        'overall_mae_tp_only': metrics_df['mae_tp_only'].mean(),
        'tolerance_days': tolerance_days,
        'verification_window': verification_window,
        'forecast_days': forecast_bin_end
    }

    return metrics_df, summary_stats



def compute_metrics_multiple_years(*, obs_dir, obs_file_pattern, obs_var, 
                                   thresh_file, thresh_var, wet_threshold, 
                                   wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, 
                                   start_date, end_date, fallback_date, mok, years, years_clim,
                                   model_dir, model_var, ref_model, date_filter_year, init_days, 
                                   unit_cvt, file_pattern, tolerance_days, verification_window, max_forecast_day, 
                                   members,  onset_percentage_threshold, probabilistic, save_nc_climatology, **kwargs):

    """Compute onset metrics for multiple years."""
#
    #members = kwargs['members']
    #probabilistic = kwargs['probabilistic']

    #mok = kwargs["mok"]
    #window = kwargs["wet_spell"]
    #wet_init = kwargs["wet_init"]
    #dry_spell = kwargs["dry_spell"]
    #dry_extent = kwargs["dry_extent"]
    #dry_threshold = kwargs["dry_threshold"]
    #max_forecast_day = kwargs['max_forecast_day']
    #tolerance_days = kwargs['tolerance_days']
    #verification_window = kwargs['verification_window']

    #climatology = kwargs['climatology']

    #forecast_bin_start = verification_window[0]
    #forecast_bin_end = verification_window[1]

    kwargs = restore_args(compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresh_file(**kwargs)

    #ref_clim = (ref_model == 'climatology')
    ref_clim = (kwargs['model'] == 'climatology')

    # calculate obs climatology onset 
    if ref_clim:
        print("Generating climatology onset doy for each year as onset_da_dict")
        climatological_onset_doy = compute_climatological_onset(**kwargs)
        #if kwargs.get('parallel'):
        #    climatological_onset_doy = parallel_climatological_onset(**kwargs)
        #else:
        #    climatological_onset_doy = compute_climatological_onset(**kwargs)

#        from datetime import datetime, timedelta
#        clim_onset_doy = climatological_onset_doy.sel(lat=11.75,lon=40.5)
#        clim_onset_date = datetime(2020, 1, 1) + timedelta(days=int(clim_onset_doy) - 1)
#        clim_onset_date = pd.to_datetime(clim_onset_date)
#        print(f"\n\nClim onset date {clim_onset_date}, at lat, lon")

        # climatological onset as "obs" for climatology baseline metrics
        onset_da_dict = {year: climatological_onset_doy for year in years}
        #onset_da_dict = {year: climatological_onset_doy for year in years_clim}

#        # save climatological onset to netcdf
#        if save_nc_climatology:
#            fout = os.path.join(kwargs['dir_out'], "climatology_onset_doy_{}.nc")
#            fout = fout.format(tuple_to_str_range(years_clim))
#            climatological_onset_doy.to_netcdf(fout)

        #for year in years_clim:
        #    print(f"\n{'-'*50}")
        #    print(f"Processing year {year}")
    
        #    # obs onset
        #    imd = load_imd_rainfall(year, **kwargs)
        #    onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)

        #    # climatology as forecast
        #    print("Computing onset for climatology as forecast ...")
        #    init_dates = get_initialization_dates(year, **kwargs)
        #    onset_df = compute_climatology_as_forecast(
        #        climatological_onset_doy, year, init_dates, onset_da, **kwargs)

        #    print("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        #    metrics_df, summary_stats = compute_onset_metrics_with_windows(
        #        onset_df, **kwargs
        #    )
    
        #    metrics_df_dict[year] = metrics_df


        #return metrics_df_dict, onset_da_dict


    # model onset vs obs onset
    for year in years:
        print(f"\n{'-'*50}")
        print(f"Processing year {year}")
        #print(f"{'='*50}")

#        print("MOKMOKMOK ", kwargs['mok'])

        # obs onset
        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)
#        print(f"\n\nMMMMMMM onset_da 11.5, 41.25 {onset_da.sel(lat=11.5,lon=41.25)}")
#        print(f"\n\nMMMMMMM onset_da 11.5, 41.25 {onset_da.sel(lat=11.5,lon=41)}")
#        sys.exit()

        # extract forecast at approporiate init dates
        if probabilistic:
            print("Extracting ensemble forecast data ...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
        elif not ref_clim:
            print("Extracting model forecast data ...")
            p_model = get_forecast_deterministic_twice_weekly(year, **kwargs)


        # detect onset dates
        if probabilistic: # emsemble forecast onset
            print("Computing onset for ensemble forecast ...")
            _, onset_df = compute_onset_for_all_members(
                p_model, thresh_da, onset_da, **kwargs
            )

        elif not ref_clim: # deterministic model onset
            print("Computing onset for model forecast ...")
            onset_df = compute_onset_for_deterministic_model(
                p_model, thresh_da, onset_da, **kwargs
            )

        elif ref_clim: # climatology as forecast
            print("Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da,
                **kwargs
            )


#        df_sel = onset_df[(onset_df["lat"] == 10) & (onset_df["lon"] == 40)]
#        #print("\n\n onset df = ", onset_df)
#        print("\n\n onset df = ", df_sel)
#        import sys
#        sys.exit()


        # onset-obs metrics TP,TN,FP,FN for all locations and init times
        print("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df, **kwargs
        )

        metrics_df_dict[year] = metrics_df


        #slat = 11.5
        #slon = 41.25

#        df_sel = metrics_df[(metrics_df["lat"] == 9.75) & (metrics_df["lon"] == 38.5)]
        #df_sel = metrics_df[(metrics_df["lat"] == 14.25) & (metrics_df["lon"] == 39.75)]
        #df_sel = metrics_df[(metrics_df["lat"] == 11.25) & (metrics_df["lon"] == 41.25)]

        #df_sel = onset_df[(onset_df["lat"] == 11.75) & (onset_df["lon"] == 40.5)]
        #df_sel = onset_df[(onset_df["lat"] == 12) & (onset_df["lon"] == 41)]

#        df_sel = onset_df[(onset_df["lat"] == slat) & (onset_df["lon"] == slon)]
#        print("\n\n onset df = ", df_sel[['onset_date','obs_onset_date']].to_string(index=False))
#        df_sel = metrics_df[(metrics_df["lat"] == slat) & (metrics_df["lon"] == slon)]
#        print("\n\n metrics df = ", df_sel[['true_positive', 'false_positive', 'true_negative', 'false_negative']].to_string(index=False))
#        print("\n\n metrics df = ", df_sel[['mae_combined']].to_string(index=False))

#        import sys
#        sys.exit()

        if not ref_clim:
            onset_da_dict[year] = onset_da


        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
              f"FP={summary_stats['overall_false_positive']}, "
              f"FN={summary_stats['overall_false_negative']}, "
              f"TN={summary_stats['overall_true_negative']}")


    return metrics_df_dict, onset_da_dict




def ens_compute_metrics_multiple_years(*, obs_dir, obs_file_pattern, obs_var,
                                   thresh_file, thresh_var, wet_threshold,
                                   wet_init, wet_spell, dry_spell, dry_threshold, dry_extent,
                                   start_date, end_date, fallback_date, mok, years, years_clim,
                                   model_dir, model_var, ref_model, date_filter_year, init_days,
                                   unit_cvt, file_pattern, tolerance_days, verification_window, max_forecast_day,
                                   members,  onset_percentage_threshold, probabilistic, save_nc_climatology, **kwargs):

    """Compute onset metrics for multiple years."""

    kwargs = restore_args(ens_compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresh_file(**kwargs)

    ref_clim = (kwargs['model'] == 'climatology')

    # calculate obs climatology onset
    if ref_clim:
        climatological_onset_doy = compute_climatological_onset(**kwargs)

        # climatological onset as "obs" for climatology baseline metrics
        onset_da_dict = {year: climatological_onset_doy for year in years}

    for year in years:
        print(f"{'-'*50}")
        print(f"Processing year {year}")
        #print(f"{'='*50}")

        # obs onset
        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)

        # extract forecast at approporiate init dates
        if probabilistic and not ref_clim:
            print("-------Extracting ensemble forecast data ...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            print("...")

            print("-------Computing onset for ensemble forecast ...")
            _, onset_df = compute_onset_for_all_members(p_model, thresh_da, onset_da, **kwargs)
            print("...")


        elif ref_clim: # climatology as forecast
            print("-------Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da,
                **kwargs
            )


        # onset-obs metrics TP,TN,FP,FN for all locations and init times
        print("-------Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df, **kwargs
        )

        metrics_df_dict[year] = metrics_df

        if not ref_clim:
            onset_da_dict[year] = onset_da


        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
              f"FP={summary_stats['overall_false_positive']}, "
              f"FN={summary_stats['overall_false_negative']}, "
              f"TN={summary_stats['overall_true_negative']}")


    return metrics_df_dict, onset_da_dict

