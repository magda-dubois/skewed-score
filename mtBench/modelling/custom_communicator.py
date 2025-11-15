from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import arviz as az  # NEW

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


def drop_not_present_vars(vars: list[str], idata) -> tuple[list[str], list[str]]:  # NEW
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