from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
from matplotlib.gridspec import GridSpec
from pathlib import Path

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


@communicate
def model_comparison_with_forest(
    ic: str = "waic",
    figsize: tuple = (20, 10),
    width_ratios: tuple = (1, 1.5),
    # Forest plot parameters
    vertical_line: float | None = None,
    best_model: bool = True,
    rank_by_baseline: bool = True,
    hdi_prob: float = 0.95,
    show_mode: str = "absolute",
    variable_labels: dict[str, str] | None = None,
    forest_title: str | None = None,
    save_comparison_table: bool = True,
    *args,
    **kwargs,
):
    """
    Combined plot: model comparison (left) + comprehensive grader forest plot (right).
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
        
        nonlocal forest_title
        
        # Create figure with two subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=width_ratios, wspace=0.5)
        
        # ========== LEFT: Model Comparison ==========
        ax_compare = fig.add_subplot(gs[0])
        
        # Gather inference data for all fitted models
        data_dict = {
            ma.model_name: ma.inference_data 
            for ma in state.models if ma.is_fitted
        }
        
        if len(data_dict) < 2:
            if display:
                display.logger.warning("Need at least 2 models for comparison")
            return state, "Error"
        
        # Compute comparison
        comparisons = az.compare(data_dict, ic=ic)
        
        # ========== SAVE COMPARISON TABLE ==========
        if save_comparison_table:
            # Save to current directory (simple approach)
            output_path = Path(f"model_comparison_{ic}.csv")
            comparisons.to_csv(output_path)
            
            if display:
                display.logger.info(f"\n{'='*60}")
                display.logger.info(f"Saved model comparison table to {output_path.absolute()}")
                display.logger.info(f"\n{ic.upper()} Model Comparison:")
                display.logger.info(f"{'='*60}")
                display.logger.info(f"\n{comparisons.to_string()}")
                display.logger.info(f"{'='*60}\n")
        
        # Plot comparison
        az.plot_compare(comparisons, ax=ax_compare, plot_standard_error=True)
        ax_compare.set_title(f"Model Comparison ({ic.upper()})", fontsize=14, fontweight='bold', pad=15)
        
        # ========== RIGHT: Comprehensive Forest Plot ==========
        ax_forest = fig.add_subplot(gs[1])
        
        # Get model to plot
        models_to_run = [state.get_best_model()] if best_model else state.models
        if best_model and models_to_run[0] is None:
            if display:
                display.logger.error("No best model found")
            plt.close(fig)
            return state, "Error"
        
        model_analysis = models_to_run[0]
        
        if not model_analysis.is_fitted:
            if display:
                display.logger.error(f"Model {model_analysis.model_name} not fitted")
            plt.close(fig)
            return state, "Error"
        
        idata = xr.Dataset(model_analysis.inference_data.posterior)
        
        if 'grader_model_shift' not in idata:
            if display:
                display.logger.warning("No grader_model_shift found")
            plt.close(fig)
            return state, "Error"
        
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
            plt.close(fig)
            return state, "Error"
        
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
        
        # Choose variables
        mode_vars = {
            "shifts": ['gpt4_shift_from_baseline', 'human_shift_from_baseline'],
            "absolute": ['gpt4_absolute_quality', 'human_absolute_quality'],
            "mean_and_diff": ['model_mean', 'grader_disagreement'],
        }
        
        plot_vars = mode_vars.get(show_mode, mode_vars["shifts"]) + \
                   ['length_bias_per_grader', 'length_squared_bias_per_grader', 'position_bias_per_grader']
        plot_vars = [v for v in plot_vars if v in idata]
        
        # Apply custom labels
        rename_map = {}
        for old_name in plot_vars:
            if old_name in default_labels:
                new_name = default_labels[old_name]
                if new_name != old_name:
                    rename_map[old_name] = new_name
        
        if rename_map:
            idata = idata.rename(rename_map)
            plot_vars = [rename_map.get(v, v) for v in plot_vars]
        
        # Convert back to InferenceData
        from arviz import InferenceData
        plot_idata = InferenceData(posterior=idata)
        
        # Plot forest on the right subplot
        az.plot_forest(
            plot_idata,
            var_names=plot_vars,
            coords=coords_dict if coords_dict else None,
            ax=ax_forest,
            combined=True,
            *args,
            **kwargs,
        )
        
        if vertical_line is not None:
            ax_forest.axvline(x=vertical_line, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        
        if forest_title is None:
            mode_titles = {
                "shifts": "Grader Deviations from Baseline",
                "absolute": "Grader Perceived Quality",
                "mean_and_diff": "Consensus & Disagreement",
            }
            forest_title = f"{mode_titles.get(show_mode, 'Grader Analysis')}\n(Ranked by consensus quality)"
        
        ax_forest.set_title(forest_title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        state.add_plot(
            plot=fig,
            plot_name=f"model_comparison_with_forest_{show_mode}",
        )
        
        return state, "pass"
    
    return communicate