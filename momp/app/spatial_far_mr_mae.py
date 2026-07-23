import os
import xarray as xr
from dataclasses import asdict
from itertools import product
import copy

from momp.metrics.error import create_spatial_far_mr_mae
from momp.stats.benchmark import compute_metrics_multiple_years
from momp.lib.control import iter_list, make_case, ref_model_case
from momp.lib.convention import Case
#from momp.lib.loader import cfg,setting
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.maps import plot_spatial_metrics
from momp.graphics.choropleth import plot_adm3_spatial_metrics
from momp.graphics.onset_map import plot_spatial_climatology_onset
from momp.graphics.panel_portrait_error import panel_portrait_mae_far_mr
from momp.io.output import save_metrics_to_netcdf, save_adm3_metrics_to_csv
#from momp.io.output import file_path
from momp.io.output import set_nested
from momp.utils.printing import tuple_to_str_range
from momp.stats.parallel import parallel_metrics_multiple_years


def spatial_far_mr_mae_map(cfg=None, setting=None):#, **kwargs):

    if cfg is None:
        cfg = get_cfg()
    if setting is None:
        setting = get_setting()

    # only executed for deterministic forecasts
    #if cfg.get('probabilistic'):
    #if getattr(cfg, "probabilistic", False):
    if cfg.probabilistic:
        return

    results = {}

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window "
                f"{case.verification_window}, case: {case.case_name}")
        #print(f"processing model onset evaluation for {case.case_name}")
        #print(f"\n verification window = {case.verification_window}\n")

        case_cfg = {**asdict(case), **asdict(setting)}

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(**case_cfg)
        else:
            metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg)
        #print("AAAAA")
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        #current = results
        #for key in combi[:-1]:
        #    current = current.setdefault(key, {})
        #current[combi[-1]] = value


        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:

            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate
                    for monsoon onset predictions""",
            }

            save_metrics_to_netcdf(spatial_metrics, case_cfg, desc_dict=desc_dict)

            if case_cfg.get('benchmark_space') == 'adm3':
                save_adm3_metrics_to_csv(spatial_metrics, case_cfg)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            if case_cfg.get('benchmark_space') == 'adm3':
                plot_adm3_spatial_metrics(spatial_metrics, **case_cfg)
            else:
                plot_spatial_metrics(spatial_metrics, **case_cfg)


#        # make climatological onset plot
#        #if case.model=='climatology' and case_cfg['plot_climatology_onset']:
#        if case_cfg['plot_climatology_onset']:
#            plot_spatial_climatology_onset(onset_da_dict, **case_cfg)


    # ------------------------------------------------------------------------
    # baseline metrics (climatology or user specified model)

    #if not cfg['ref_model']:
    if not cfg.ref_model:
        return results 
        #pass

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    #print("cfg_ref['model_list'] = ", cfg_ref['model_list'])
    layout_pool = iter_list(vars(cfg_ref))
    #print("cfg_ref layout_pool = ", layout_pool)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window"
                f"{case.verification_window}, case: {case.case_name}")
        #print(f"processing model onset evaluation for {case.case_name}")

        #case, case_cfg_reg = ref_model_case(case, case_cfg)

        case_ref = {'model_dir': case_cfg['ref_model_dir'],
                    'model_var': case_cfg['ref_model_var'],
                    'file_pattern': case_cfg['ref_model_file_pattern'],
                    'unit_cvt': case_cfg['ref_model_unit_cvt']
                    }

        case.update(case_ref)

        #if case.model == 'climatology':
        #    case.years = case.years_clim

        case_cfg_ref = {**asdict(case), **asdict(setting)}

        #print("case_cfg_ref = \n", case_cfg_ref)

        #case_cfg_ref = {**case_cfg,
        #              #'model': case_cfg['ref_model'],
        #              'model_dir': case_cfg['ref_model_dir'],
        #              'model_var': case_cfg['ref_model_var'],
        #              'file_pattern': case_cfg['ref_model_file_pattern'],
        #              'unit_cvt': case_cfg['ref_model_unit_cvt']
        #              }

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(**case_cfg_ref)
        else:
            metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate
                    for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case_cfg_ref, desc_dict=desc_dict)

            if case_cfg_ref.get('benchmark_space') == 'adm3':
                save_adm3_metrics_to_csv(spatial_metrics, case_cfg_ref)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            if case_cfg_ref.get('benchmark_space') == 'adm3':
                plot_adm3_spatial_metrics(spatial_metrics, **case_cfg_ref)
            else:
                plot_spatial_metrics(spatial_metrics, **case_cfg_ref)


    # save climatological onset to netcdf
    if case.model == 'climatology' and case_cfg['save_nc_climatology']:
        fout = os.path.join(case_cfg_ref['dir_out'], "climatology_onset_doy_{}.nc")
        fout = fout.format(tuple_to_str_range(case_cfg_ref['years_clim']))
        climatological_onset_doy = next(iter(onset_da_dict.values()))
        climatological_onset_doy.attrs["years"] = case.years_clim
        climatological_onset_doy.to_netcdf(fout)

    # spatial map of climatology onset day
    if case_cfg['plot_climatology_onset']:
        if case_cfg_ref.get('benchmark_space') == 'adm3':
            print("plot_climatology_onset is not supported for adm3 benchmarking; skipping.")
        else:
            plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

#    if 2 > 1:
#        import pickle
#        fout = os.path.join(cfg['dir_out'],"combi_error_results.pkl")
#        with open(fout, "wb") as f:
#            pickle.dump(results, f)
    

    # panel portrait plot of mae, far, mr
    if case_cfg['plot_panel_heatmap_error']:
       panel_portrait_mae_far_mr(results, **case_cfg_ref) 


    #print("\n\n\n results dict = ", results)
    return results

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    results = spatial_far_mr_mae_map()

#    import pickle
#    fout = os.path.join(cfg['dir_out'],"combi_error_results.pkl")
#    with open(fout, "rb") as f:
#        my_dict = pickle.load(f)
    

