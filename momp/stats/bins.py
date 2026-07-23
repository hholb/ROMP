import pandas as pd
import numpy as np
from momp.io.input import load_thresh_file, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_all_members
#from momp.lib.control import restore_args
from momp.utils.practical import restore_args
from momp.utils.standard import loc_cols
#from momp.stats.climatology import compute_climatological_onset_dataset
from itertools import product


def extract_day_range(bin_label):
    """ extract the start day of each bin for sorting purpose """
    if 'Days ' in bin_label:
        try:
            day_part = bin_label.replace('Days ', '').split('-')[0]
            return int(day_part)
        except:
            return 999
    return 999


def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)

    target_bins = []
    for bin_label in common_bins:
        if (bin_label.startswith('Days ') and
            not bin_label.startswith('After') and
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)

    return sorted(target_bins, key=extract_day_range)



# Function to create forecast-observation pairs with specified day bins for probabilistic verification
def create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, *, day_bins, max_forecast_day, **kwargs):
    """
    Create forecast-observation pairs using specified day bins, including a final bin for "after max_forecast_day".
    
    Parameters:
    -----------
    onset_all_members : DataFrame
        DataFrame with ensemble member onset predictions
    onset_da : xarray.DataArray
        Observed onset dates
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day. Members without onset get assigned to "after day X" bin
    """
    #day_bins = kwargs["stats_day_bins"]
    #max_forecast_day = kwargs["max_forecast_day"]

    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    loc = loc_cols(onset_all_members)
    keys = ['init_time'] + loc

    # Add the "after max_forecast_day" bin
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)

    df = onset_all_members[keys + ['onset_day', 'obs_onset_date']].copy()
    df['init_dt'] = pd.to_datetime(df['init_time'])
    df['obs_dt'] = pd.to_datetime(df['obs_onset_date'])

    # keep only valid cases (observed onset exists and init precedes it);
    # member rows are only ever produced for valid cases, so this is a
    # safety net matching the original per-group skips
    df = df[df['obs_dt'].notna() & (df['init_dt'] < df['obs_dt'])]

    onset_day = pd.to_numeric(df['onset_day'], errors='coerce')

    print(f"Processing {len(df[keys].drop_duplicates())} forecast cases with day bins: {day_bins}")
    print(f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window")

    obs_days_from_init = (df['obs_dt'] - df['init_dt']).dt.days

    # per-member membership and per-case observed indicator for every bin
    bin_frames = []
    for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
        if bin_start == -float('inf'):
            bin_label = f'Before day {min_forecast_day}'
            member_in_bin = onset_day.notna() & (onset_day < min_forecast_day)
            observed_onset = obs_days_from_init < min_forecast_day
        elif bin_start > max_forecast_day:
            bin_label = f'After day {max_forecast_day}'
            member_in_bin = onset_day.isna() | (onset_day > max_forecast_day)
            observed_onset = obs_days_from_init > max_forecast_day
        else:
            bin_label = f'Days {bin_start}-{bin_end}'
            member_in_bin = onset_day.notna() & (onset_day >= bin_start) & (onset_day <= bin_end)
            observed_onset = (obs_days_from_init >= bin_start) & (obs_days_from_init <= bin_end)

        per_case = pd.DataFrame({
            **{k: df[k] for k in keys},
            'members_with_onset': member_in_bin.astype(int),
            'observed_onset': observed_onset.astype(int),
            'obs_dt': df['obs_dt'],
        }).groupby(keys, as_index=False).agg(
            members_with_onset=('members_with_onset', 'sum'),
            observed_onset=('observed_onset', 'first'),
            total_members=('members_with_onset', 'size'),
            obs_dt=('obs_dt', 'first'),
        )
        per_case['bin_start'] = bin_start
        per_case['bin_end'] = bin_end
        per_case['bin_label'] = bin_label
        per_case['bin_index'] = bin_idx
        bin_frames.append(per_case)

    forecast_obs_df = pd.concat(bin_frames, ignore_index=True)
    forecast_obs_df = forecast_obs_df.sort_values(keys + ['bin_index'],
                                                  kind='stable').reset_index(drop=True)
    forecast_obs_df['predicted_prob'] = (
        forecast_obs_df['members_with_onset'] / forecast_obs_df['total_members']
    )
    forecast_obs_df['year'] = pd.to_datetime(forecast_obs_df['init_time']).dt.year
    forecast_obs_df['obs_onset_date'] = forecast_obs_df['obs_dt'].dt.strftime('%Y-%m-%d')

    forecast_obs_df = forecast_obs_df[keys + [
        'bin_start', 'bin_end', 'bin_label', 'predicted_prob', 'observed_onset',
        'members_with_onset', 'total_members', 'year', 'obs_onset_date', 'bin_index'
    ]]
#    print("result_list = ", results_list)
#    print("forecast_obs_df = ", forecast_obs_df)
    #if 2 > 1:
    #    import os
    #    import pickle
    #    fout = os.path.join(kwargs['dir_out'], "forecast_obs_df.pkl")
    #    with open(fout, "wb") as f:
    #        pickle.dump(forecast_obs_df, f)

    print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean'
    }).round(3)
    print(bin_stats)

    return forecast_obs_df


## This function creates forecast-observation pairs using climatological ensemble where each year is a member
def create_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, *, 
                                             day_bins, max_forecast_day, **kwargs):
    """
    Create forecast-observation pairs using climatological ensemble where each year is a member.
    Uses day-of-year instead of calendar dates for onset comparison.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates for all years
    target_year : int
        The year to use as "truth" for observations
    init_dates : list or pandas.DatetimeIndex
        Initialization dates for forecasts
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day

    #mok : bool, default=True
    #    Whether to use MOK date filter (June 2nd)

    Returns:
    --------
    DataFrame with forecast-observation pairs
    """

    # adjust max_forecast_day to the end of day_bins
    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    # Get the observed onset for the target year
    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")

    obs_onset_da = clim_onset.sel(year=target_year)

    # Use ALL years as ensemble members (including target year) ### redundant
    ensemble_years = list(clim_onset.year.values)
    total_members = len(ensemble_years)

    # Create extended bins including "before initialization" and "after max_forecast_day" bins
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)

    print(f"Creating climatological forecasts for target year {target_year}")
    print(f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}")
    print(f"Processing {len(init_dates)} initialization dates")
    print(f"Day bins: {day_bins}")
    print(f"Extended bins include: 'Before day {min_forecast_day}' and 'After day {max_forecast_day}' ")
    print(f"Using day-of-year method for onset comparison")

    loc = [d for d in obs_onset_da.dims]
    print("Processing " + " x ".join(f"{obs_onset_da.sizes[d]} {d}" for d in loc) + " locations")

    # long member table: one row per (location, ensemble-year member)
    ens_df = clim_onset.to_dataframe(name='ens_onset').reset_index()
    ens_df['ens_doy'] = pd.to_datetime(ens_df['ens_onset']).dt.dayofyear
    ens_df = ens_df.sort_values(loc + ['year'], kind='stable')

    obs_df = obs_onset_da.to_dataframe(name='obs_onset').reset_index()[loc + ['obs_onset']]
    obs_df = obs_df[obs_df['obs_onset'].notna()].copy()
    obs_df['obs_onset_dt'] = pd.to_datetime(obs_df['obs_onset'])
    obs_df['obs_onset_doy'] = obs_df['obs_onset_dt'].dt.dayofyear

    frames = []
    for init_date in pd.to_datetime(init_dates):
        init_doy = int(init_date.dayofyear)

        # only forecasts initialized before the observed onset (by day of year)
        valid_obs = obs_df[obs_df['obs_onset_doy'] > init_doy]
        if valid_obs.empty:
            continue

        m = ens_df.merge(
            valid_obs[loc + ['obs_onset_dt', 'obs_onset_doy']], on=loc, how='inner'
        )
        member_days = m['ens_doy'] - init_doy
        obs_days = m['obs_onset_doy'] - init_doy

        for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
            if bin_start == -float('inf'):
                bin_label = f'Before day {min_forecast_day}'
                in_bin = member_days.notna() & (member_days <= min_forecast_day - 1)
                # original quirk: contributing years for this bin use <= 0,
                # not <= min_forecast_day - 1
                contributes = member_days.notna() & (member_days <= 0)
                observed = (obs_days <= min_forecast_day - 1)
            elif bin_start > max_forecast_day:
                bin_label = f'After day {max_forecast_day}'
                in_bin = member_days.notna() & (member_days > max_forecast_day)
                contributes = in_bin
                observed = (obs_days > max_forecast_day)
            else:
                bin_label = f'Days {bin_start}-{bin_end}'
                in_bin = member_days.notna() & (member_days >= bin_start) & (member_days <= bin_end)
                contributes = in_bin
                observed = (obs_days >= bin_start) & (obs_days <= bin_end)

            m[f'in_{bin_idx}'] = in_bin.astype(int)
            # rows are year-sorted within each location, so concatenation
            # reproduces the original sorted year list
            m[f'ct_{bin_idx}'] = np.where(contributes, m['year'].astype(str) + ',', '')
            m[f'ob_{bin_idx}'] = observed.astype(int)

        n_bins = len(extended_bins)
        agg_spec = {'obs_onset_dt': ('obs_onset_dt', 'first'),
                    'obs_onset_doy': ('obs_onset_doy', 'first')}
        for k in range(n_bins):
            agg_spec[f'in_{k}'] = (f'in_{k}', 'sum')
            agg_spec[f'ob_{k}'] = (f'ob_{k}', 'first')
            agg_spec[f'ct_{k}'] = (f'ct_{k}', 'sum')
        g = m.groupby(loc, as_index=False, sort=True).agg(**agg_spec)
        for k in range(n_bins):
            g[f'ct_{k}'] = g[f'ct_{k}'].str.rstrip(',')

        # Skip locations where no member showed onset in any bin
        g['total_members_with_onset'] = g[[f'in_{k}' for k in range(n_bins)]].sum(axis=1)
        g = g[g['total_members_with_onset'] > 0]
        if g.empty:
            continue

        for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
            if bin_start == -float('inf'):
                bin_label = f'Before day {min_forecast_day}'
            elif bin_start > max_forecast_day:
                bin_label = f'After day {max_forecast_day}'
            else:
                bin_label = f'Days {bin_start}-{bin_end}'

            frame = g[loc].copy()
            frame['init_time'] = init_date.strftime('%Y-%m-%d')
            frame['bin_start'] = bin_start
            frame['bin_end'] = bin_end
            frame['bin_label'] = bin_label
            frame['members_with_onset'] = g[f'in_{bin_idx}'].values
            frame['total_members'] = total_members
            frame['total_members_with_onset'] = g['total_members_with_onset'].values
            frame['predicted_prob'] = frame['members_with_onset'] / frame['total_members_with_onset']
            frame['observed_onset'] = g[f'ob_{bin_idx}'].values
            frame['contributing_years'] = g[f'ct_{bin_idx}'].values
            frame['n_contributing_years'] = frame['contributing_years'].str.count(',') + (
                frame['contributing_years'].str.len() > 0).astype(int)
            frame['year'] = target_year
            frame['obs_onset_date'] = g['obs_onset_dt'].dt.strftime('%Y-%m-%d').values
            frame['obs_onset_doy'] = g['obs_onset_doy'].values
            frame['init_doy'] = init_doy
            frame['obs_days_from_init_doy'] = g['obs_onset_doy'].values - init_doy
            frame['bin_index'] = bin_idx
            frame['forecast_type'] = 'climatological_doy'
            frames.append(frame)

    if frames:
        forecast_obs_df = pd.concat(frames, ignore_index=True)
        forecast_obs_df = forecast_obs_df[[
            'init_time'] + loc + ['bin_start', 'bin_end', 'bin_label', 'predicted_prob',
            'observed_onset', 'members_with_onset', 'total_members',
            'total_members_with_onset', 'contributing_years', 'n_contributing_years',
            'year', 'obs_onset_date', 'obs_onset_doy', 'init_doy',
            'obs_days_from_init_doy', 'bin_index', 'forecast_type']]
    else:
        forecast_obs_df = pd.DataFrame()

    if len(forecast_obs_df) == 0:
        print("Warning: No forecast-observation pairs generated")
        return forecast_obs_df

    print(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Verify uniqueness
    unique_locations_in_output = len(forecast_obs_df[loc].drop_duplicates())
    print(f"Unique locations in output: {unique_locations_in_output}")

    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean',
        'n_contributing_years': 'mean',
        'total_members_with_onset': 'mean'
    }).round(3)
    print(bin_stats)

    return forecast_obs_df



# This function creates the observed forecast pairs for multiple years (core monsoon zone grids) and combines them
def multi_year_forecast_obs_pairs(*, years, obs_dir, obs_file_pattern, obs_var,
                                  thresh_file, thresh_var, wet_threshold,
                                  date_filter_year, init_days, start_date, end_date,
                                  model_dir, model_var, unit_cvt, file_pattern,
                                  wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, mok,
                                  members, onset_percentage_threshold, max_forecast_day, day_bins, **kwargs):
    """Main function to perform multi-year reliability analysis."""

    kwargs = restore_args(multi_year_forecast_obs_pairs, kwargs, locals())
#    print("\n\n\nmax_forecast_day = ", max_forecast_day)
#    print("max_forecast_day kwargs = ", kwargs['max_forecast_day'])

    #members = kwargs['members']
    #probabilistic = kwargs['probabilistic']

    #mok = kwargs["mok"]
    #window = kwargs["wet_spell"]
    #wet_init = kwargs["wet_init"]
    #dry_spell = kwargs["dry_spell"]
    #dry_extent = kwargs["dry_extent"]
    #dry_threshold = kwargs["dry_threshold"]
    #max_forecast_day = kwargs['max_forecast_day']

    print(f"Processing years: {years}")

    thresh_slice = load_thresh_file(**kwargs)

    # Initialize list to store all forecast-observation pairs
    all_forecast_obs_pairs = []
                 
    # Process each year
    for year in years:
        print(f"\n{'-'*50}")
        print(f"Processing year {year}")
        #print(f"{'='*50}")
                        
        try:            
            # Load model and observation data
            print("Loading S2S model data...")
            #p_model,_ = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
#            p_model_slice = p_model.sel(lat=inside_lats, lon=inside_lons)
            p_model_slice = p_model # !!!!! region subset
                    
            print("Loading observational rainfall data...")
            rainfall_ds = load_imd_rainfall(year, **kwargs)
#            rainfall_ds_slice = rainfall_ds.sel(lat=inside_lats, lon=inside_lons)
            rainfall_ds_slice = rainfall_ds #!!!!! region subset

            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, **kwargs)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")
                        
            print("Computing onset for all ensemble members...")
            onset_all_members, _ = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, **kwargs)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")
    
#            print("onset_all_members = ", onset_all_members)
#            print("onset_da = ", onset_da)
            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, **kwargs)

#            print("DONE!!!")
            # Add to master list
            all_forecast_obs_pairs.append(forecast_obs_pairs)

            print(f"Year {year} completed: {len(forecast_obs_pairs)} forecast-observation pairs")

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            raise
            continue

    # Combine all years
    print(f"\n{'-'*50}")
    print("Combining all years")
    #print(f"{'='*50}")

    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    # Print final summary statistics
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")
    return combined_forecast_obs



#def multi_year_climatological_forecast_obs_pairs(clim_onset, *, years_clim, day_bins, max_forecast_day, date_filter_year, init_days, start_date, end_date, **kwargs):
def multi_year_climatological_forecast_obs_pairs(clim_onset, *, years, day_bins, max_forecast_day, date_filter_year, init_days, start_date, end_date, **kwargs):
    """
    Create climatological forecast-observation pairs for multiple target years.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    #years_clim : list
    years : list
        Years to use as truth for observations
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins
    max_forecast_day : int, default=15
        Maximum forecast day
    #mok : bool, default=True
    #    Whether to use MOK date filter

    Returns:
    --------
    DataFrame with combined forecast-observation pairs from all target years
    """

    kwargs = restore_args(multi_year_climatological_forecast_obs_pairs, kwargs, locals())

    clim_onset_slice = clim_onset #!!!!! region subset

    all_forecast_obs_pairs = []

    #for target_year in years_clim:
    for target_year in years:
        print(f"\n{'-'*50}")
        print(f"Processing target year {target_year}")
        #print(f"{'='*50}")

        try:
            # Get initialization dates for this year
            init_dates = get_initialization_dates(target_year, **kwargs)


            # Create forecast-observation pairs for this year
            forecast_obs_pairs = create_climatological_forecast_obs_pairs(
                clim_onset_slice,
                target_year,
                init_dates,
                **kwargs
            )

            if len(forecast_obs_pairs) > 0:
                all_forecast_obs_pairs.append(forecast_obs_pairs)
                print(f"Target year {target_year} completed: {len(forecast_obs_pairs)} pairs")
            else:
                print(f"No pairs generated for target year {target_year}")

        except Exception as e:
            print(f"Error processing target year {target_year}: {e}")
            continue

    # Combine all years
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any target year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    print(f"\n{'='*50}")
    print("CLIMATOLOGICAL FORECAST SUMMARY")
    print(f"{'='*50}")
    #print(f"Target years processed: {years_clim}")
    print(f"Target years processed: {years}")
    print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    print(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    print(f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")

    return combined_forecast_obs



