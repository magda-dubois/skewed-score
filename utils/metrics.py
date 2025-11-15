import numpy as np
import jax
import krippendorff
import jax.numpy as jnp
from numpyro.infer import Predictive
from utils.modeling import ordered_logistic_model

def compute_observed_krippendorff(df):
    """Compute raw Krippendorff's α on observed data"""
    matrix = df.groupby(['grader', 'item'])['score'].mean().unstack()
    matrix_array = matrix.values
    alpha_obs = krippendorff.alpha(matrix_array, level_of_measurement='interval')
    return alpha_obs


def compute_krippendorff_from_predictions(pred_scores, df):
    """
    Convert predicted scores to grader×item matrix and compute Krippendorff's α
    """
    df_pred = df.copy()
    df_pred['score_pred'] = pred_scores
    
    matrix = df_pred.groupby(['grader', 'item'])['score_pred'].mean().unstack()
    matrix_array = matrix.values
    matrix_array = np.where(np.isnan(matrix_array), np.nan, matrix_array)
    
    alpha = krippendorff.alpha(matrix_array, level_of_measurement='interval')
    return alpha


def compute_bias_corrected_alpha_simple(posterior_sample_idx, df, idata, covariates_dict, main_effects, interactions):
    """
    Compute α after removing systematic grader bias by zeroing out grader effects in posterior samples
    This matches the description: "subtracts the estimated main effect of the corresponding grader 
    from the linear predictor before mapping it to a categorical score"
    """
    # Get all posterior samples for this iteration
    single_sample = {}
    for param in idata.posterior.data_vars:
        if param == 'intercept':
            single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx]
        elif param == 'first_cutpoint':
            single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx]
        elif param == 'cutpoint_diffs':
            single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx, :]
        elif param.startswith('b_grader'):
            # INTERVENTION: Set grader effects to zero
            # This removes systematic differences in baseline scoring
            if param == 'b_grader':
                original_shape = idata.posterior[param].values[0, posterior_sample_idx, :].shape
                single_sample[param] = jnp.zeros(original_shape)
            elif param == 'b_grader_full':
                original_shape = idata.posterior[param].values[0, posterior_sample_idx, :].shape
                single_sample[param] = jnp.zeros(original_shape)
        elif param in ['b_interaction_grader_item', 'b_interaction_grader_item_full']:
            # KEEP grader-item interactions - these are NOT main effects
            # We only want to remove systematic baseline differences (main effects)
            if param == 'b_interaction_grader_item':
                single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx, :]
            elif param == 'b_interaction_grader_item_full':
                single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx, :, :]
        else:
            # Keep all other parameters as they were
            if len(idata.posterior[param].shape) == 2:  # (chain, sample)
                single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx]
            elif len(idata.posterior[param].shape) == 3:  # (chain, sample, param)
                single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx, :]
            elif len(idata.posterior[param].shape) == 4:  # (chain, sample, param1, param2)
                single_sample[param] = idata.posterior[param].values[0, posterior_sample_idx, :, :]
    
    # Rest of the function remains the same...
    rng_key = jax.random.PRNGKey(42 + posterior_sample_idx)
    
    predictive = Predictive(
        ordered_logistic_model,
        posterior_samples={k: jnp.array([v]) for k, v in single_sample.items()},
        return_sites=["obs"]
    )
    
    counterfactual_samples = predictive(
        rng_key,
        covariates_dict=covariates_dict,
        y=None,
        main_effects=main_effects,
        interactions=interactions
    )
    
    predicted_scores = counterfactual_samples["obs"][0]
    
    df_counterfactual = df.copy()
    df_counterfactual['score_counterfactual'] = predicted_scores
    
    matrix = df_counterfactual.groupby(['grader', 'item'])['score_counterfactual'].mean().unstack()
    alpha = krippendorff.alpha(matrix.values, level_of_measurement='interval')
    
    return alpha