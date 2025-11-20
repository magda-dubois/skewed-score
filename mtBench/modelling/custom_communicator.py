from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay

@communicate
def comprehensive_grader_forest_plot(
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (12, 10),
    rank_by_baseline: bool = True,
    hdi_prob: float = 0.95,
    show_mode: str = "shifts",
    variable_labels: dict[str, str] | None = None,
    title: str | None = None,  
    *args,
    **kwargs,
):
    """
    Forest plot with three display modes.
    """
    
    default_labels = {
        'gpt4_shift_from_baseline': 'GPT-4 deviation',
        'human_shift_from_baseline': 'Human deviation',
        'gpt4_absolute_quality': 'GPT-4 quality',
        'human_absolute_quality': 'Human quality',
        'model_mean': 'Consensus quality',
        'grader_disagreement': 'Grader disagreement',
        'length_bias_per_grader': 'Length bias',
        'length_squared_bias_per_grader': 'Length² bias',
        'position_bias_per_grader': 'Position bias',
    }
    
    if variable_labels:
        default_labels.update(variable_labels)
    
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        
        nonlocal title
        
        models_to_run = [state.get_best_model()] if best_model else state.models
        if best_model and models_to_run[0] is None:
            raise ValueError("No best model found.")

        for model_analysis in models_to_run:
            if not model_analysis.is_fitted:
                continue
                
            idata = xr.Dataset(model_analysis.inference_data.posterior)
            
            if 'grader_model_shift' not in idata:
                if display:
                    display.logger.warning("No grader_model_shift found")
                continue
            
            effects = idata['grader_model_shift']
            model_mean = idata['model_mean']
            model_coords = effects.coords['left_model'].values
            grader_coords = effects.coords['grader'].values
            
            try:
                gpt4_idx = list(grader_coords).index('gpt4_pair')
                human_idx = list(grader_coords).index('human')
            except ValueError as e:
                if display:
                    display.logger.error(f"Could not find graders: {e}")
                continue
            
            # Create views
            gpt4_shift = effects.isel(grader=gpt4_idx)
            human_shift = effects.isel(grader=human_idx)
            gpt4_absolute = model_mean + gpt4_shift
            human_absolute = model_mean + human_shift
            grader_disagreement = gpt4_shift - human_shift
            
            idata['gpt4_shift_from_baseline'] = gpt4_shift
            idata['human_shift_from_baseline'] = human_shift
            idata['gpt4_absolute_quality'] = gpt4_absolute
            idata['human_absolute_quality'] = human_absolute
            idata['grader_disagreement'] = grader_disagreement
            
            # Rename covariates
            if 'grader_length_diff_prop_effects' in idata:
                idata['length_bias_per_grader'] = idata['grader_length_diff_prop_effects']
            
            if 'grader_length_diff_prop_squared_effects' in idata:
                idata['length_squared_bias_per_grader'] = idata['grader_length_diff_prop_squared_effects']
            
            if 'grader_position_numeric_effects' in idata:
                idata['position_bias_per_grader'] = idata['grader_position_numeric_effects']
            
            # Ranking
            coords_dict = {}
            sorted_coords = model_coords
            
            if rank_by_baseline:
                posterior_means = model_mean.mean(dim=["chain", "draw"])
                sorted_idx = np.argsort(posterior_means.values)[::-1]
                sorted_coords = model_coords[sorted_idx].tolist()
                coords_dict['left_model'] = sorted_coords
                
                if display:
                    display.logger.info(f"Ranked by baseline: {sorted_coords}")
            
            # Compute significance
            significant_disagreements = {}
            for model in model_coords:
                hdi = az.hdi(grader_disagreement.sel(left_model=model), hdi_prob=hdi_prob)
                hdi_vals = hdi.to_array().values.flatten() if hasattr(hdi, 'to_array') else \
                           np.array(list(hdi.data_vars.values())[0].values).flatten()
                significant_disagreements[model] = (hdi_vals[0] > 0 or hdi_vals[1] < 0)
            
            # Choose variables
            mode_vars = {
                "shifts": ['gpt4_shift_from_baseline', 'human_shift_from_baseline'],
                "absolute": ['gpt4_absolute_quality', 'human_absolute_quality'],
                "mean_and_diff": ['model_mean', 'grader_disagreement'],
            }
            
            plot_vars = mode_vars.get(show_mode, mode_vars["shifts"]) + \
                       ['length_bias_per_grader', 'length_squared_bias_per_grader', 'position_bias_per_grader']
            plot_vars = [v for v in plot_vars if v in idata]
            
            # Apply custom labels by renaming variables
            rename_map = {}
            for old_name in plot_vars:
                if old_name in default_labels:
                    new_name = default_labels[old_name]
                    if new_name != old_name:
                        rename_map[old_name] = new_name
            
            if rename_map:
                idata = idata.rename(rename_map)
                plot_vars = [rename_map.get(v, v) for v in plot_vars]
                
                if display:
                    display.logger.info(f"Applied labels: {rename_map}")
            
            # Convert back to InferenceData for plotting
            from arviz import InferenceData
            plot_idata = InferenceData(posterior=idata)
            
            # Plot
            ax = az.plot_forest(
                plot_idata,
                var_names=plot_vars,
                coords=coords_dict if coords_dict else None,
                figsize=figsize,
                *args,
                **kwargs,
            )

            if vertical_line is not None:
                ax[0].axvline(x=vertical_line, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            
            if title is None:
                mode_titles = {
                    "shifts": "Grader Deviations from Baseline",
                    "absolute": "Grader Perceived Quality",
                    "mean_and_diff": "Consensus & Disagreement",
                }
                title = f"{mode_titles.get(show_mode, 'Grader Analysis')}\n(Ranked by consensus quality)"

            ax[0].set_title(title, fontsize=13, pad=15)
            
            fig = plt.gcf()
            plt.tight_layout()
            
            state.add_plot(
                plot=fig,
                plot_name=f"grader_{show_mode}_{model_analysis.model_name}",
            )
            
            # Console output
            if display:
                display.logger.info("\n" + "="*60)
                display.logger.info(f"GRADER ANALYSIS ({show_mode.upper()}):")
                display.logger.info("="*60)
                
                if show_mode == "shifts":
                    display.logger.info(f"{'Model':<20} {'GPT-4':>10} {'Human':>10}")
                    display.logger.info("-" * 45)
                    for model in sorted_coords:
                        gpt4_val = float(gpt4_shift.sel(left_model=model).mean().values)
                        human_val = float(human_shift.sel(left_model=model).mean().values)
                        marker = "⚠️ " if significant_disagreements.get(model, False) else "   "
                        display.logger.info(f"{marker}{model:<20} {gpt4_val:>10.2f} {human_val:>10.2f}")
                
                elif show_mode == "absolute":
                    display.logger.info(f"{'Model':<20} {'GPT-4':>10} {'Human':>10} {'Diff':>10}")
                    display.logger.info("-" * 55)
                    for model in sorted_coords:
                        gpt4_val = float(gpt4_absolute.sel(left_model=model).mean().values)
                        human_val = float(human_absolute.sel(left_model=model).mean().values)
                        diff = gpt4_val - human_val
                        marker = "⚠️ " if significant_disagreements.get(model, False) else "   "
                        display.logger.info(f"{marker}{model:<20} {gpt4_val:>10.2f} {human_val:>10.2f} {diff:>10.2f}")
                
                else:  # mean_and_diff
                    display.logger.info(f"{'Model':<20} {'Mean':>10} {'GPT-4 Δ':>10}")
                    display.logger.info("-" * 45)
                    for model in sorted_coords:
                        mean_val = float(model_mean.sel(left_model=model).mean().values)
                        diff_val = float(grader_disagreement.sel(left_model=model).mean().values)
                        marker = "⚠️ " if significant_disagreements.get(model, False) else "   "
                        display.logger.info(f"{marker}{model:<20} {mean_val:>10.2f} {diff_val:>10.2f}")
                
                display.logger.info("="*60 + "\n")
        
        return state, "pass"

    return communicate