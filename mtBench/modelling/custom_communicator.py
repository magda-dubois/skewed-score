from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import arviz as az

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


def drop_not_present_vars(vars: list[str], idata) -> tuple[list[str], list[str]]:
    """
    Drop variables that are not present in the inference data.
    
    Returns:
        Tuple of (present_vars, dropped_vars)
    """
    present = []
    dropped = []
    
    for var in vars:
        if var in idata.posterior:
            present.append(var)
        else:
            dropped.append(var)
    
    return present, dropped


def compute_cohens_kappa(obs, pred):
    """
    Compute Cohen's Kappa manually.
    
    κ = (p_o - p_e) / (1 - p_e)
    """
    p_o = (obs == pred).mean()
    
    p_obs_0 = (obs == 0).mean()
    p_obs_1 = (obs == 1).mean()
    p_pred_0 = (pred == 0).mean()
    p_pred_1 = (pred == 1).mean()
    
    p_e = p_obs_0 * p_pred_0 + p_obs_1 * p_pred_1
    
    if p_e == 1.0:
        return 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


@communicate
def forest_plot_custom(
    vars: list[str] | None = None,
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 5),
    transform: bool = False,
    rank_vars: list[str] | None = None,  
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Communicate the results of a model analysis.
        """
        nonlocal vars
        if best_model:
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            model_vars = vars
            if model_analysis.is_fitted:
                model_vars, dropped = (
                    drop_not_present_vars(model_vars, model_analysis.inference_data)
                    if model_vars
                    else (None, None)
                )
                if dropped and display:
                    display.logger.warning(
                        f"Variables {dropped} were not found in the model {model_analysis.model_name} inference data."
                    )
                if model_vars is None:
                    model_vars = model_analysis.model_config.get_plot_params()

                # Work with original data
                idata = model_analysis.inference_data
                coords_dict = {}  # NEW: store custom coordinate orders
                
                # NEW: Handle ranking
                if rank_vars:
                    import copy
                    # Make a deep copy to avoid modifying original
                    idata = copy.deepcopy(idata)
                    
                    for var in rank_vars:
                        if var in idata.posterior:
                            # Compute posterior means
                            posterior_means = idata.posterior[var].mean(dim=["chain", "draw"])
                            
                            # Get the coordinate dimension
                            coord_dim = list(posterior_means.dims)[0]
                            
                            # Get sorted order (descending)
                            sorted_idx = np.argsort(posterior_means.values)[::-1]
                            sorted_coords = posterior_means[coord_dim].values[sorted_idx]
                            
                            # Store for coords parameter
                            coords_dict[coord_dim] = sorted_coords.tolist()
                            
                            if display:
                                display.logger.info(f"Ranked variable '{var}' by posterior mean")
                                display.logger.info(f"Original order: {posterior_means[coord_dim].values.tolist()}")
                                display.logger.info(f"Sorted order: {sorted_coords.tolist()}")
                                display.logger.info(f"Means: {posterior_means.values[sorted_idx].tolist()}")
                            
                            # Reorder the variable in the posterior
                            idata.posterior[var] = idata.posterior[var].isel({coord_dim: sorted_idx})
                            # Update the coordinate labels
                            idata.posterior[var][coord_dim] = sorted_coords

                ax = az.plot_forest(
                    idata,
                    var_names=model_vars,
                    figsize=figsize,
                    coords=coords_dict if coords_dict else None,  # NEW: pass custom order
                    transform=model_analysis.link_function if transform else None,
                    *args,
                    **kwargs,
                )

                if vertical_line is not None:
                    ax[0].axvline(
                        x=vertical_line,
                        color="red",
                        linestyle="--",
                    )
                fig = plt.gcf()
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_{'-'.join(model_vars) if model_vars else ''}_forest",
                )
        return state, "pass"

    return communicate

def compute_bias_corrected_probs_direct(
    raw_probs: np.ndarray,
    bias_to_remove: dict,
):
    """
    Remove bias from existing predictions by adjusting log-odds mathematically.
    
    Args:
        raw_probs: Original predicted probabilities
        bias_to_remove: Dict mapping parameter name to (effects, feature_values, group_indices)
    """
    from scipy.special import logit, expit
    
    # Convert probs to log-odds
    log_odds = logit(np.clip(raw_probs, 1e-7, 1-1e-7))
    
    # Subtract each bias term
    for param_name, (effects, feature_vals, group_idx) in bias_to_remove.items():
        contribution = effects[group_idx] * feature_vals
        log_odds = log_odds - contribution
    
    # Convert back to probabilities
    probs_corrected = expit(log_odds)
    
    return probs_corrected

@communicate
def model_vs_observed_agreement(
    n_posterior_samples: int = 1000,
    ci_level: float = 0.95,
    figsize: tuple[int, int] = (16, 5),
    compute_bias_corrected: bool = True,
    bias_parameters_to_zero: list[str] | None = None,
    zero_only_graders: list[str] | None = None,
    compare_multiple_corrections: bool = True,  # NEW: compute all 3 scenarios
    agreement_xlim: tuple[float, float] | None = None,  # NEW: e.g., (0.85, 0.90) to zoom
):
    """
    Compare Bayesian model's posterior predictive choices against observed choices.
    
    Args:
        compare_multiple_corrections: If True, compute raw, GPT-4-only, and both corrections
        agreement_xlim: Tuple of (min, max) for agreement rate x-axis. If None, uses (0, 1)
    """
    
    def communicate(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue
                
            if not model_analysis.inference_data.get("posterior_predictive"):
                if display:
                    display.logger.warning(
                        f"No posterior predictive samples for {model_analysis.model_name}"
                    )
                continue
            
            # Get observed data
            df = state.processed_data.copy()
            observed_choices = df['left_chosen'].values
            
            if display:
                display.logger.info(
                    f"\n{'='*60}\n"
                    f"Model: {model_analysis.model_name}\n"
                    f"N comparisons: {len(observed_choices)}\n"
                    f"{'='*60}"
                )
            
            # Determine how many samples to use
            total_draws = model_analysis.inference_data.posterior.sizes["draw"]
            n_samples = min(n_posterior_samples, total_draws)
            posterior_idx = np.arange(n_samples)
            
            if display:
                display.logger.info(f"Computing agreement over {n_samples} posterior samples (chain 0)")
            
            # Pre-compute grader mapping
            grader_map = {g: idx for idx, g in enumerate(sorted(df['grader'].unique()))}
            grader_idx = np.array([grader_map[g] for g in df['grader'].values])
            
            # Storage for all scenarios
            all_scenarios = {}
            
            # 1. RAW AGREEMENT (with all biases)
            if display:
                display.logger.info("\n📊 Computing RAW agreement...")
            
            agreement_rates = []
            kappas = []
            
            for i in posterior_idx:
                raw_probs = model_analysis.inference_data.posterior_predictive.obs_probs.values[0, i, :]
                pred_choices = (raw_probs > 0.5).astype(int)
                
                agreement = (pred_choices == observed_choices).mean()
                agreement_rates.append(agreement)
                
                kappa = compute_cohens_kappa(observed_choices, pred_choices)
                kappas.append(kappa)
            
            all_scenarios['Raw'] = {
                'agreement_rates': np.array(agreement_rates),
                'kappas': np.array(kappas),
                'color': '#3498db',
            }
            
            mean_agreement = all_scenarios['Raw']['agreement_rates'].mean()
            mean_kappa = all_scenarios['Raw']['kappas'].mean()
            
            if display:
                display.logger.info(
                    f"  Agreement Rate: {mean_agreement:.3f}\n"
                    f"  Cohen's Kappa:  {mean_kappa:.3f}"
                )
            
            # 2. BIAS-CORRECTED SCENARIOS
            if compute_bias_corrected and bias_parameters_to_zero:
                
                # Determine which scenarios to compute
                if compare_multiple_corrections:
                    scenarios_to_compute = [
                        ('GPT-4 only', list(grader_map.keys())[0] if 'gpt4' in str(grader_map.keys()).lower() else zero_only_graders[0] if zero_only_graders else None),
                        ('Both graders', None),
                    ]
                elif zero_only_graders:
                    scenarios_to_compute = [
                        (f"{', '.join(zero_only_graders)} only", zero_only_graders),
                    ]
                else:
                    scenarios_to_compute = [
                        ('Both graders', None),
                    ]
                
                colors_bc = ['#2ecc71', '#e74c3c', '#f39c12']
                
                for scenario_idx, (scenario_name, graders_to_zero) in enumerate(scenarios_to_compute):
                    if graders_to_zero is None:
                        # Zero all graders
                        zero_list = None
                    elif isinstance(graders_to_zero, str):
                        zero_list = [graders_to_zero]
                    else:
                        zero_list = graders_to_zero
                    
                    if display:
                        if zero_list:
                            display.logger.info(f"\n📊 Computing {scenario_name} bias correction...")
                        else:
                            display.logger.info(f"\n📊 Computing bias correction for all graders...")
                    
                    agreement_rates_bc = []
                    kappas_bc = []
                    
                    for i_idx, i in enumerate(posterior_idx):
                        raw_probs = model_analysis.inference_data.posterior_predictive.obs_probs.values[0, i, :]
                        
                        # Build bias terms to remove
                        bias_to_remove = {}
                        for param_name in bias_parameters_to_zero:
                            if param_name == 'grader_length_diff_prop_effects':
                                effects = model_analysis.inference_data.posterior[param_name].values[0, i, :]
                                feature_vals = df['length_diff_prop'].values
                                
                                if zero_list:
                                    effects_modified = effects.copy()
                                    for grader_name in zero_list:
                                        if grader_name in grader_map:
                                            grader_idx_to_zero = grader_map[grader_name]
                                            effects_modified[grader_idx_to_zero] = 0.0
                                    bias_to_remove[param_name] = (effects - effects_modified, feature_vals, grader_idx)
                                else:
                                    bias_to_remove[param_name] = (effects, feature_vals, grader_idx)
                            
                            elif param_name == 'grader_position_numeric_effects':
                                effects = model_analysis.inference_data.posterior[param_name].values[0, i, :]
                                feature_vals = df['position_numeric'].values
                                
                                if zero_list:
                                    effects_modified = effects.copy()
                                    for grader_name in zero_list:
                                        if grader_name in grader_map:
                                            grader_idx_to_zero = grader_map[grader_name]
                                            effects_modified[grader_idx_to_zero] = 0.0
                                    bias_to_remove[param_name] = (effects - effects_modified, feature_vals, grader_idx)
                                else:
                                    bias_to_remove[param_name] = (effects, feature_vals, grader_idx)
                        
                        # Compute bias-corrected probabilities
                        pred_probs_bc = compute_bias_corrected_probs_direct(raw_probs, bias_to_remove)
                        pred_choices_bc = (pred_probs_bc > 0.5).astype(int)
                        
                        agreement_bc = (pred_choices_bc == observed_choices).mean()
                        agreement_rates_bc.append(agreement_bc)
                        
                        kappa_bc = compute_cohens_kappa(observed_choices, pred_choices_bc)
                        kappas_bc.append(kappa_bc)
                    
                    all_scenarios[scenario_name] = {
                        'agreement_rates': np.array(agreement_rates_bc),
                        'kappas': np.array(kappas_bc),
                        'color': colors_bc[scenario_idx % len(colors_bc)],
                    }
                    
                    mean_agreement_bc = all_scenarios[scenario_name]['agreement_rates'].mean()
                    mean_kappa_bc = all_scenarios[scenario_name]['kappas'].mean()
                    
                    if display:
                        display.logger.info(
                            f"  Agreement Rate: {mean_agreement_bc:.3f} (Δ {mean_agreement_bc - mean_agreement:+.4f})\n"
                            f"  Cohen's Kappa:  {mean_kappa_bc:.3f} (Δ {mean_kappa_bc - mean_kappa:+.4f})"
                        )
            
            # 3. PLOT - TWO SUBPLOTS
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            positions = list(range(len(all_scenarios)))
            labels = list(all_scenarios.keys())
            
            # LEFT PLOT: Agreement Rate
            data_agree = [s['agreement_rates'] for s in all_scenarios.values()]
            colors = [s['color'] for s in all_scenarios.values()]
            
            parts1 = ax1.violinplot(
                data_agree,
                positions=positions,
                vert=False,
                widths=0.7,
                showmeans=True,
                showmedians=False,
            )
            
            for pc, color in zip(parts1['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add mean dots
            means_agree = [s['agreement_rates'].mean() for s in all_scenarios.values()]
            ax1.scatter(means_agree, positions,
                       color='darkgreen', s=200, zorder=10, marker='o', edgecolor='black', linewidth=2)
            
            ax1.axvline(0.80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='MT-bench (0.80)')
            ax1.set_yticks(positions)
            ax1.set_yticklabels(labels)
            ax1.set_xlabel('Agreement Rate', fontsize=12)
            
            # NEW: Zoomed x-axis
            if agreement_xlim:
                ax1.set_xlim(agreement_xlim)
            else:
                ax1.set_xlim([0, 1])
            
            ax1.grid(axis='x', alpha=0.3)
            ax1.legend(loc='lower right', fontsize=9)
            ax1.set_title('Agreement Rate', fontsize=12, fontweight='bold')
            
            # RIGHT PLOT: Cohen's Kappa
            data_kappa = [s['kappas'] for s in all_scenarios.values()]
            
            parts2 = ax2.violinplot(
                data_kappa,
                positions=positions,
                vert=False,
                widths=0.7,
                showmeans=True,
                showmedians=False,
            )
            
            for pc, color in zip(parts2['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add mean dots
            means_kappa = [s['kappas'].mean() for s in all_scenarios.values()]
            ax2.scatter(means_kappa, positions,
                       color='darkgreen', s=200, zorder=10, marker='o', edgecolor='black', linewidth=2)
            
            ax2.axvline(0.6, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Substantial (0.60)')
            ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No agreement (0)')
            ax2.set_yticks(positions)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel("Cohen's Kappa", fontsize=12)
            ax2.set_xlim([-0.2, 1])
            ax2.grid(axis='x', alpha=0.3)
            ax2.legend(loc='lower right', fontsize=9)
            ax2.set_title("Cohen's Kappa", fontsize=12, fontweight='bold')
            
            # Overall title
            title = f'Model vs Observed Agreement: {model_analysis.model_name}'
            if bias_parameters_to_zero:
                title += f'\n(Bias correction on {", ".join(bias_parameters_to_zero)})'
            fig.suptitle(title, fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            state.add_plot(
                plot=fig,
                plot_name=f"model_{model_analysis.model_name}_agreement"
            )
            
            plt.close(fig)
        
        return state, "pass"
    
    return communicate

def debug_bias_correction(
    i_idx: int,
    raw_probs: np.ndarray,
    pred_probs_bc: np.ndarray,
    observed_choices: np.ndarray,
    bias_to_remove: dict,
    bias_parameters_to_zero: list[str],
    zero_only_graders: list[str] | None = None,  # NEW
):
    """Debug function to print bias correction details for first sample."""
    if i_idx != 0:
        return
    
    raw_choices_0 = (raw_probs > 0.5).astype(int)
    bc_choices_0 = (pred_probs_bc > 0.5).astype(int)
    
    print(f"\n{'='*60}")
    print(f"🐛 DEBUG (sample 0)")
    print(f"{'='*60}")
    
    print(f"\n📋 Predictions:")
    print(f"   Raw (first 10):          {raw_choices_0[:10]}")
    print(f"   Bias-corrected (first 10): {bc_choices_0[:10]}")
    print(f"   Observed (first 10):     {observed_choices[:10]}")
    
    print(f"\n📊 Agreement:")
    raw_agree_0 = (raw_choices_0 == observed_choices).mean()
    bc_agree_0 = (bc_choices_0 == observed_choices).mean()
    n_changed = (raw_choices_0 != bc_choices_0).sum()
    
    raw_kappa_0 = compute_cohens_kappa(observed_choices, raw_choices_0)
    bc_kappa_0 = compute_cohens_kappa(observed_choices, bc_choices_0)
    
    print(f"   Raw agreement:       {raw_agree_0:.4f} (κ={raw_kappa_0:.4f})")
    print(f"   BC agreement:        {bc_agree_0:.4f} (κ={bc_kappa_0:.4f})")
    print(f"   Difference:          {bc_agree_0 - raw_agree_0:+.4f}")
    print(f"   Predictions changed: {n_changed} / {len(raw_choices_0)} ({n_changed/len(raw_choices_0)*100:.1f}%)")
    
    print(f"\n📈 Base rates:")
    print(f"   Observed: {observed_choices.mean():.3f} choose left")
    print(f"   Raw pred: {raw_choices_0.mean():.3f} choose left")
    print(f"   BC pred:  {bc_choices_0.mean():.3f} choose left")
    
    print(f"\n🔍 Bias removal:")
    print(f"   Parameters to zero: {bias_parameters_to_zero}")
    if zero_only_graders:
        print(f"   Only for graders:   {zero_only_graders}")
    else:
        print(f"   For all graders")
    print(f"   Bias terms found:   {list(bias_to_remove.keys())}")
    
    for param_name, (effects, feature_vals, idx) in bias_to_remove.items():
        contrib = effects[idx] * feature_vals
        print(f"\n   {param_name}:")
        print(f"     Effects shape:        {effects.shape}")
        print(f"     Effects values:       {effects}")
        print(f"     Feature vals shape:   {feature_vals.shape}")
        print(f"     Contribution (first 10): {contrib[:10]}")
        print(f"     Contribution stats:")
        print(f"       mean = {contrib.mean():.4f}")
        print(f"       std  = {contrib.std():.4f}")
        print(f"       min  = {contrib.min():.4f}")
        print(f"       max  = {contrib.max():.4f}")
    
    print(f"\n{'='*60}\n")


@communicate
def gpt4_vs_human_agreement(
    gpt4_grader: str = 'gpt4_pair',
    human_grader: str = 'human',
    n_posterior_samples: int = 1000,
    ci_level: float = 0.95,
    compute_bias_corrected: bool = True,
    bias_parameters_to_zero: list[str] | None = None,
    figsize: tuple[int, int] = (14, 5),
):
    """
    Compute agreement between GPT-4 and human judges using posterior predictive samples.
    
    Compares:
    1. Raw GPT-4 vs Human (with all biases)
    2. Bias-corrected GPT-4 vs Human (GPT-4's bias removed)
    """
    
    def communicate(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue
                
            if not model_analysis.inference_data.get("posterior_predictive"):
                if display:
                    display.logger.warning(f"No posterior predictive samples")
                continue
            
            df = state.processed_data.copy()
            
            # Add row index to track position in original dataframe
            df['original_idx'] = np.arange(len(df))
            
            # Get GPT-4 and human data
            gpt4_data = df[df['grader'] == gpt4_grader].copy()
            human_data = df[df['grader'] == human_grader].copy()
            
            if display:
                display.logger.info(
                    f"\n{'='*60}\n"
                    f"GPT-4 vs Human Agreement (MT-bench style)\n"
                    f"{'='*60}"
                )
                display.logger.info(f"GPT-4 judgments: {len(gpt4_data)}")
                display.logger.info(f"Human judgments: {len(human_data)}")
            
            # Merge on same comparisons
            merged = gpt4_data.merge(
                human_data,
                on=['question_id', 'left_model', 'right_model'],
                suffixes=('_gpt4', '_human'),
                how='inner'
            )
            
            if len(merged) == 0:
                if display:
                    display.logger.warning("No overlapping comparisons!")
                return state, CommunicateResult(passed=False)
            
            if display:
                display.logger.info(f"Overlapping comparisons: {len(merged)}")
            
            # Get the original indices for GPT-4 rows in the merged data
            gpt4_original_indices = merged['original_idx_gpt4'].values
            
            # Get actual observed human choices for comparison
            human_choices_observed = merged['left_chosen_human'].values
            
            # Prepare for posterior sampling
            total_draws = model_analysis.inference_data.posterior.sizes["draw"]
            n_samples = min(n_posterior_samples, total_draws)
            posterior_idx = np.arange(n_samples)
            
            grader_map = {g: idx for idx, g in enumerate(sorted(df['grader'].unique()))}
            grader_idx_full = np.array([grader_map[g] for g in df['grader'].values])
            
            all_scenarios = {}
            
            # 1. RAW GPT-4 vs Human
            if display:
                display.logger.info("\n📊 Computing RAW GPT-4 vs Human agreement...")
            
            agreement_rates_raw = []
            kappas_raw = []
            
            for i in posterior_idx:
                raw_probs = model_analysis.inference_data.posterior_predictive.obs_probs.values[0, i, :]
                
                # Get GPT-4 predictions for overlapping comparisons using original indices
                gpt4_pred_choices = (raw_probs[gpt4_original_indices] > 0.5).astype(int)
                
                # Compare to human observations
                agreement = (gpt4_pred_choices == human_choices_observed).mean()
                agreement_rates_raw.append(agreement)
                
                kappa = compute_cohens_kappa(human_choices_observed, gpt4_pred_choices)
                kappas_raw.append(kappa)
            
            all_scenarios['Raw GPT-4'] = {
                'agreement_rates': np.array(agreement_rates_raw),
                'kappas': np.array(kappas_raw),
                'color': '#3498db',
            }
            
            mean_agree_raw = all_scenarios['Raw GPT-4']['agreement_rates'].mean()
            mean_kappa_raw = all_scenarios['Raw GPT-4']['kappas'].mean()
            
            if display:
                ci_lower = np.percentile(agreement_rates_raw, (1 - ci_level) / 2 * 100)
                ci_upper = np.percentile(agreement_rates_raw, (1 + ci_level) / 2 * 100)
                display.logger.info(
                    f"  Agreement: {mean_agree_raw:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]\n"
                    f"  Kappa:     {mean_kappa_raw:.3f}"
                )
            
            # 2. BIAS-CORRECTED GPT-4 vs Human
            if compute_bias_corrected and bias_parameters_to_zero:
                if display:
                    display.logger.info("\n📊 Computing BIAS-CORRECTED GPT-4 vs Human agreement...")
                
                agreement_rates_bc = []
                kappas_bc = []
                
                for i in posterior_idx:
                    raw_probs = model_analysis.inference_data.posterior_predictive.obs_probs.values[0, i, :]
                    
                    # Build bias correction (only for GPT-4)
                    bias_to_remove = {}
                    for param_name in bias_parameters_to_zero:
                        if param_name == 'grader_length_diff_prop_effects':
                            effects = model_analysis.inference_data.posterior[param_name].values[0, i, :]
                            feature_vals = df['length_diff_prop'].values
                            
                            # Zero only GPT-4's effect
                            effects_modified = effects.copy()
                            gpt4_idx_in_grader_map = grader_map[gpt4_grader]
                            effects_modified[gpt4_idx_in_grader_map] = 0.0
                            
                            bias_to_remove[param_name] = (effects - effects_modified, feature_vals, grader_idx_full)
                        
                        elif param_name == 'grader_position_numeric_effects':
                            effects = model_analysis.inference_data.posterior[param_name].values[0, i, :]
                            feature_vals = df['position_numeric'].values
                            
                            effects_modified = effects.copy()
                            gpt4_idx_in_grader_map = grader_map[gpt4_grader]
                            effects_modified[gpt4_idx_in_grader_map] = 0.0
                            
                            bias_to_remove[param_name] = (effects - effects_modified, feature_vals, grader_idx_full)
                    
                    # Apply bias correction
                    pred_probs_bc = compute_bias_corrected_probs_direct(raw_probs, bias_to_remove)
                    
                    # Get GPT-4 bias-corrected predictions using original indices
                    gpt4_pred_choices_bc = (pred_probs_bc[gpt4_original_indices] > 0.5).astype(int)
                    
                    # Compare to human observations
                    agreement_bc = (gpt4_pred_choices_bc == human_choices_observed).mean()
                    agreement_rates_bc.append(agreement_bc)
                    
                    kappa_bc = compute_cohens_kappa(human_choices_observed, gpt4_pred_choices_bc)
                    kappas_bc.append(kappa_bc)
                
                all_scenarios['Bias-corrected GPT-4'] = {
                    'agreement_rates': np.array(agreement_rates_bc),
                    'kappas': np.array(kappas_bc),
                    'color': '#2ecc71',
                }
                
                mean_agree_bc = all_scenarios['Bias-corrected GPT-4']['agreement_rates'].mean()
                mean_kappa_bc = all_scenarios['Bias-corrected GPT-4']['kappas'].mean()
                
                if display:
                    ci_lower_bc = np.percentile(agreement_rates_bc, (1 - ci_level) / 2 * 100)
                    ci_upper_bc = np.percentile(agreement_rates_bc, (1 + ci_level) / 2 * 100)
                    display.logger.info(
                        f"  Agreement: {mean_agree_bc:.3f} [{ci_lower_bc:.3f}, {ci_upper_bc:.3f}]\n"
                        f"  Kappa:     {mean_kappa_bc:.3f}\n"
                        f"  Δ Agreement: {mean_agree_bc - mean_agree_raw:+.4f}\n"
                        f"  Δ Kappa:     {mean_kappa_bc - mean_kappa_raw:+.4f}"
                    )
            
            # 3. PLOT
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            positions = list(range(len(all_scenarios)))
            labels = list(all_scenarios.keys())
            colors = [s['color'] for s in all_scenarios.values()]
            
            # LEFT: Agreement Rate
            data_agree = [s['agreement_rates'] for s in all_scenarios.values()]
            
            parts1 = ax1.violinplot(
                data_agree,
                positions=positions,
                vert=False,
                widths=0.7,
                showmeans=True,
                showmedians=False,
            )
            
            for pc, color in zip(parts1['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            means_agree = [s['agreement_rates'].mean() for s in all_scenarios.values()]
            ax1.scatter(means_agree, positions, color='darkgreen', s=200, zorder=10,
                       marker='o', edgecolor='black', linewidth=2)
            
            ax1.axvline(0.80, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label='MT-bench (0.80)')
            ax1.set_yticks(positions)
            ax1.set_yticklabels(labels)
            ax1.set_xlabel('Agreement Rate', fontsize=12)
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
            ax1.legend(loc='lower right', fontsize=9)
            ax1.set_title('Agreement Rate\n(GPT-4 vs Human)', fontsize=12, fontweight='bold')
            
            # RIGHT: Cohen's Kappa
            data_kappa = [s['kappas'] for s in all_scenarios.values()]
            
            parts2 = ax2.violinplot(
                data_kappa,
                positions=positions,
                vert=False,
                widths=0.7,
                showmeans=True,
                showmedians=False,
            )
            
            for pc, color in zip(parts2['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            means_kappa = [s['kappas'].mean() for s in all_scenarios.values()]
            ax2.scatter(means_kappa, positions, color='darkgreen', s=200, zorder=10,
                       marker='o', edgecolor='black', linewidth=2)
            
            ax2.axvline(0.6, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                       label='Substantial (0.60)')
            ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_yticks(positions)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel("Cohen's Kappa", fontsize=12)
            ax2.set_xlim([-0.2, 1])
            ax2.grid(axis='x', alpha=0.3)
            ax2.legend(loc='lower right', fontsize=9)
            ax2.set_title("Cohen's Kappa\n(GPT-4 vs Human)", fontsize=12, fontweight='bold')
            
            fig.suptitle(
                f'GPT-4 vs Human Agreement ({len(merged)} overlapping comparisons)\n'
                f'Using Posterior Predictive Distributions',
                fontsize=13, fontweight='bold'
            )
            
            plt.tight_layout()
            
            state.add_plot(
                plot=fig,
                plot_name=f"gpt4_vs_human_agreement"
            )
            
            plt.close(fig)
        
        return state, "pass"
    
    return communicate


@communicate
def gpt4_vs_human_agreement_point_estimate(
    gpt4_grader: str = 'gpt4_pair',
    human_grader: str = 'human',
    compute_bias_corrected: bool = True,
    bias_parameters_to_zero: list[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Compare GPT-4 vs Human using POSTERIOR MEAN predictions (not full distribution).
    This gives a clearer signal of the debiasing effect.
    """
    
    def communicate(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        
        for model_analysis in state.models:
            if not model_analysis.is_fitted:
                continue
                
            if not model_analysis.inference_data.get("posterior_predictive"):
                continue
            
            df = state.processed_data.copy()
            df['original_idx'] = np.arange(len(df))
            
            # Get GPT-4 and human data
            gpt4_data = df[df['grader'] == gpt4_grader].copy()
            human_data = df[df['grader'] == human_grader].copy()
            
            if display:
                display.logger.info(
                    f"\n{'='*60}\n"
                    f"GPT-4 vs Human Agreement (Posterior Mean)\n"
                    f"{'='*60}"
                )
            
            # Merge on same comparisons
            merged = gpt4_data.merge(
                human_data,
                on=['question_id', 'left_model', 'right_model'],
                suffixes=('_gpt4', '_human'),
                how='inner'
            )
            
            if len(merged) == 0:
                if display:
                    display.logger.warning("No overlapping comparisons!")
                return state, CommunicateResult(passed=False)
            
            if display:
                display.logger.info(f"Overlapping comparisons: {len(merged)}")
            
            gpt4_original_indices = merged['original_idx_gpt4'].values
            human_choices_observed = merged['left_chosen_human'].values
            
            # Get POSTERIOR MEAN predictions (not sampling)
            raw_probs_mean = model_analysis.inference_data.posterior_predictive.obs_probs.mean(
                dim=['chain', 'draw']
            ).values
            
            # 1. RAW: Posterior mean predictions
            gpt4_pred_raw = (raw_probs_mean[gpt4_original_indices] > 0.5).astype(int)
            agreement_raw = (gpt4_pred_raw == human_choices_observed).mean()
            kappa_raw = compute_cohens_kappa(human_choices_observed, gpt4_pred_raw)
            
            if display:
                display.logger.info(
                    f"\n📊 RAW (posterior mean):\n"
                    f"  Agreement: {agreement_raw:.4f}\n"
                    f"  Kappa:     {kappa_raw:.4f}"
                )
            
            results = {
                'Raw': (agreement_raw, kappa_raw),
            }
            
            # 2. BIAS-CORRECTED: Using posterior mean effects
            if compute_bias_corrected and bias_parameters_to_zero:
                
                # Get posterior mean of effects
                grader_map = {g: idx for idx, g in enumerate(sorted(df['grader'].unique()))}
                grader_idx_full = np.array([grader_map[g] for g in df['grader'].values])
                
                bias_to_remove = {}
                
                for param_name in bias_parameters_to_zero:
                    if param_name == 'grader_length_diff_prop_effects':
                        # Posterior mean effects
                        effects_mean = model_analysis.inference_data.posterior[param_name].mean(
                            dim=['chain', 'draw']
                        ).values
                        feature_vals = df['length_diff_prop'].values
                        
                        # Zero only GPT-4
                        effects_modified = effects_mean.copy()
                        gpt4_idx = grader_map[gpt4_grader]
                        effects_modified[gpt4_idx] = 0.0
                        
                        bias_to_remove[param_name] = (
                            effects_mean - effects_modified, 
                            feature_vals, 
                            grader_idx_full
                        )
                        
                        if display:
                            display.logger.info(
                                f"\n🔧 Removing length bias:\n"
                                f"  GPT-4 effect: {effects_mean[gpt4_idx]:.3f} → 0.0\n"
                                f"  Human effect: {effects_mean[1-gpt4_idx]:.3f} (unchanged)"
                            )
                
                # Apply bias correction
                probs_bc = compute_bias_corrected_probs_direct(raw_probs_mean, bias_to_remove)
                
                gpt4_pred_bc = (probs_bc[gpt4_original_indices] > 0.5).astype(int)
                agreement_bc = (gpt4_pred_bc == human_choices_observed).mean()
                kappa_bc = compute_cohens_kappa(human_choices_observed, gpt4_pred_bc)
                
                # How many predictions changed?
                n_changed = (gpt4_pred_raw != gpt4_pred_bc).sum()
                
                if display:
                    display.logger.info(
                        f"\n📊 BIAS-CORRECTED (posterior mean):\n"
                        f"  Agreement: {agreement_bc:.4f} (Δ {agreement_bc - agreement_raw:+.4f})\n"
                        f"  Kappa:     {kappa_bc:.4f} (Δ {kappa_bc - kappa_raw:+.4f})\n"
                        f"  Predictions changed: {n_changed}/{len(gpt4_pred_raw)} ({n_changed/len(gpt4_pred_raw)*100:.1f}%)"
                    )
                
                results['Bias-corrected'] = (agreement_bc, kappa_bc)
            
            # 3. PLOT
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            scenarios = list(results.keys())
            agreements = [results[s][0] for s in scenarios]
            kappas = [results[s][1] for s in scenarios]
            colors = ['#3498db', '#2ecc71']
            
            # LEFT: Agreement
            bars1 = ax1.bar(scenarios, agreements, color=colors[:len(scenarios)], 
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            ax1.axhline(0.80, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label='MT-bench (0.80)')
            ax1.set_ylabel('Agreement Rate', fontsize=12)
            ax1.set_ylim([0, 1])
            ax1.set_title('Agreement Rate', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Add values
            for bar, val in zip(bars1, agreements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # RIGHT: Kappa
            bars2 = ax2.bar(scenarios, kappas, color=colors[:len(scenarios)],
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            ax2.axhline(0.60, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                       label='Substantial (0.60)')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax2.set_ylabel("Cohen's Kappa", fontsize=12)
            ax2.set_ylim([-0.2, 1])
            ax2.set_title("Cohen's Kappa", fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # Add values
            for bar, val in zip(bars2, kappas):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            fig.suptitle(
                f'GPT-4 vs Human Agreement (Posterior Mean, {len(merged)} comparisons)',
                fontsize=13, fontweight='bold'
            )
            
            plt.tight_layout()
            
            state.add_plot(
                plot=fig,
                plot_name=f"gpt4_vs_human_agreement_point"
            )
            
            plt.close(fig)
        
        return state, "pass"
    
    return communicate

@communicate
def forest_plot_with_grader_diffs(
    vars: list[str] | None = None,
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (12, 10),
    transform: bool = False,
    rank_vars: list[str] | None = None,
    gpt4_grader: str = 'gpt4_pair',
    human_grader: str = 'human',
    *args,
    **kwargs,
):
    """
    Forest plot with grader preference differences computed on-the-fly.
    """
    
    def communicate(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        
        nonlocal vars
        if best_model:
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            model_vars = vars
            if model_analysis.is_fitted:
                
                # Work with inference data
                idata = model_analysis.inference_data
                
                # Check if we need to compute grader preference differences
                if 'grader_model_shift' in idata.posterior:
                    param_name = 'grader_model_shift'
                elif 'grader_left_model_effects' in idata.posterior:
                    param_name = 'grader_left_model_effects'
                else:
                    param_name = None
                
                # Compute preference differences if requested
                if param_name is not None and model_vars and 'grader_preference_diff' in model_vars:
                    import xarray as xr
                    import copy
                    
                    # Make a copy to avoid modifying original
                    idata = copy.deepcopy(idata)
                    
                    effects = idata.posterior[param_name]  # xarray DataArray
                    
                    if display:
                        display.logger.info(f"\nProcessing '{param_name}' to compute preference differences:")
                        display.logger.info(f"  Dims: {effects.dims}")
                        display.logger.info(f"  Coords: {list(effects.coords.keys())}")
                    
                    # Get the dimension names
                    dims = list(effects.dims)
                    grader_dim = dims[2]  # First non-chain/draw dim
                    model_dim = dims[3]   # Second non-chain/draw dim
                    
                    # Get coordinate values
                    grader_coords = effects.coords[grader_dim].values
                    model_coords = effects.coords[model_dim].values
                    
                    if display:
                        display.logger.info(f"  Grader dim: '{grader_dim}'")
                        display.logger.info(f"  Grader values: {grader_coords}")
                        display.logger.info(f"  Model dim: '{model_dim}'")
                        display.logger.info(f"  Model values: {model_coords}")
                    
                    # Find grader indices with error checking
                    gpt4_matches = np.where(grader_coords == gpt4_grader)[0]
                    human_matches = np.where(grader_coords == human_grader)[0]
                    
                    if len(gpt4_matches) == 0:
                        if display:
                            display.logger.error(
                                f"Grader '{gpt4_grader}' not found in coordinates!\n"
                                f"Available: {grader_coords}\n"
                                f"Skipping grader_preference_diff computation."
                            )
                        # Remove grader_preference_diff from vars
                        model_vars = [v for v in model_vars if v != 'grader_preference_diff']
                    elif len(human_matches) == 0:
                        if display:
                            display.logger.error(
                                f"Grader '{human_grader}' not found in coordinates!\n"
                                f"Available: {grader_coords}\n"
                                f"Skipping grader_preference_diff computation."
                            )
                        model_vars = [v for v in model_vars if v != 'grader_preference_diff']
                    else:
                        # Found both graders
                        gpt4_idx = gpt4_matches[0]
                        human_idx = human_matches[0]
                        
                        # Select using isel
                        gpt4_effects = effects.isel({grader_dim: gpt4_idx})
                        human_effects = effects.isel({grader_dim: human_idx})
                        
                        # Compute difference
                        diff = gpt4_effects - human_effects
                        
                        # Rename dimension to something meaningful
                        diff = diff.rename({model_dim: 'left_model'})
                        
                        # Add to posterior as new variable
                        idata.posterior['grader_preference_diff'] = diff
                        
                        if display:
                            display.logger.info(
                                f"  ✓ Computed grader_preference_diff: {gpt4_grader} - {human_grader}\n"
                                f"    Shape: {diff.shape}, Dims: {diff.dims}"
                            )
                
                # Check which vars are present
                model_vars, dropped = (
                    drop_not_present_vars(model_vars, idata)
                    if model_vars
                    else (None, None)
                )
                if dropped and display:
                    display.logger.warning(
                        f"Variables {dropped} were not found in {model_analysis.model_name}"
                    )
                if model_vars is None:
                    model_vars = model_analysis.model_config.get_plot_params()

                # Handle ranking
                coords_dict = {}
                if rank_vars:
                    for var in rank_vars:
                        if var in idata.posterior:
                            posterior_means = idata.posterior[var].mean(dim=["chain", "draw"])
                            coord_dim = list(posterior_means.dims)[0]
                            sorted_idx = np.argsort(posterior_means.values)[::-1]
                            sorted_coords = posterior_means[coord_dim].values[sorted_idx]
                            coords_dict[coord_dim] = sorted_coords.tolist()
                            
                            if display:
                                display.logger.info(f"Ranked variable '{var}' by posterior mean")
                            
                            # Reorder
                            idata.posterior[var] = idata.posterior[var].isel({coord_dim: sorted_idx})
                            idata.posterior[var][coord_dim] = sorted_coords
                
                # Plot forest
                ax = az.plot_forest(
                    idata,
                    var_names=model_vars,
                    figsize=figsize,
                    coords=coords_dict if coords_dict else None,
                    transform=model_analysis.link_function if transform else None,
                    *args,
                    **kwargs,
                )

                if vertical_line is not None:
                    ax[0].axvline(
                        x=vertical_line,
                        color="red",
                        linestyle="--",
                    )
                
                fig = plt.gcf()
                state.add_plot(
                    plot=fig,
                    plot_name=f"model_{model_analysis.model_name}_{'-'.join(model_vars) if model_vars else ''}_forest",
                )
        
        return state, "pass"

    return communicate