import pandas as pd


def loc_cols(df):
    """Location key columns of a results DataFrame.

    Returns ['adm3'] for admin-unit (adm3) benchmarking results, otherwise
    the default grid keys ['lat', 'lon'].
    """
    return ["adm3"] if "adm3" in df.columns else ["lat", "lon"]


def dim_fmt(ds):
    """Standardize dimension names"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

#    if "time" not in coord_list:
#        print("time NOT in coords --> ")  # , model_name)
#        time_coords = [variable for variable in coord_list if "TIME" in variable][0]
#        ds = ds.rename({time_coords: "time"})

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        keywords = ["time", 'date']
        time_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({time_coords: "time"})

    return ds


def dim_fmt_model(ds):
    """Standardize dimension names for deterministic reforecast model data"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

    if "init_time" not in coord_list:
        #print("init_time NOT in coords --> ")  # , model_name)
        time_coords = [variable for variable in coord_list if "time" in variable.lower()][0]
        ds = ds.rename({time_coords: "init_time"})

    if "step" not in coord_list:
        keywords = ["day", "prediction_timedelta"]
        step_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({step_coords: "step"})

    # convert TimedeltaIndex to integer (days)
    if isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    return ds


def dim_fmt_model_ensemble(ds):
    """Standardize dimension names for probabilistic reforecast model data"""

    ds = dim_fmt_model(ds)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        keywords = ["number", "sample"]
        ensemble_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({ensemble_coords: "member"})

    return ds



