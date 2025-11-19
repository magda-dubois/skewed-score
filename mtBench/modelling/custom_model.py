from typing import List, Optional, Tuple

import jax.numpy as jnp
from jax.nn import sigmoid
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features

@model
def pairwise_logistic_model_nc(
    hierarchical_effects: Optional[List[Tuple[str, str]]] = None,
    hierarchical_continuous: Optional[List[str]] = None,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_hierarchical_mean_loc: float = 0.0,
    prior_hierarchical_mean_scale: float = 0.5,
    prior_hierarchical_sigma_scale: float = 1.0,
) -> Model:
    """
    Binary logistic regression for pairwise comparisons with hierarchical effects.
    
    - Model rankings with question-specific variation
    - Grader-specific biases for continuous covariates (position, length)
    - Non-centered parameterization for better sampling
    """

    def model(features: Features) -> None:
        continuous = set(hierarchical_continuous or [])
        categorical = set()
        
        for group_var, effect_var in (hierarchical_effects or []):
            categorical.add(group_var if group_var not in continuous else effect_var)
            if effect_var not in continuous:
                categorical.add(effect_var)
        
        # Build requirements
        required = ["obs"]
        for cat in categorical:
            required.extend([f"{cat}_index", f"num_{cat}"])
        required.extend(continuous)
        
        if 'left_model' in categorical:
            required.extend(['right_model_index', 'num_right_model'])
        
        check_features(features, required)

        # Intercept
        eta = numpyro.sample("intercept", dist.Normal(prior_intercept_loc, prior_intercept_scale))

        # Track if we've created shared model_mean
        model_mean_created = False
        model_mean = None
        model_sigma = None

        # Hierarchical effects
        for group_var, effect_var in (hierarchical_effects or []):
            is_continuous = effect_var in continuous
            effects_name = f"{group_var}_{effect_var}_effects"
            
            # Pairwise model comparisons
            if effect_var == 'left_model':
                n_models = features["num_left_model"]
                
                # Create shared model_mean and model_sigma only once
                if not model_mean_created:
                    model_mean = numpyro.sample(
                        "model_mean",
                        dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                        .expand([n_models]).to_event(1)
                    )
                    model_sigma = numpyro.sample("model_sigma", dist.HalfNormal(prior_hierarchical_sigma_scale))
                    model_mean_created = True
                
                # Create group-specific deviations
                raw = numpyro.sample(
                    f"{effects_name}_raw",
                    dist.Normal(0, 1).expand([features[f"num_{group_var}"], n_models]).to_event(2)
                )
                effects = numpyro.deterministic(effects_name, model_mean[None, :] + model_sigma * raw)
                
                eta = eta + effects[features[f"{group_var}_index"], features["left_model_index"]] \
                          - effects[features[f"{group_var}_index"], features["right_model_index"]]
            
            # Grader-specific slopes for continuous variables
            elif is_continuous:
                mean = numpyro.sample(
                    f"{effect_var}_slope_mean",
                    dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                )
                sigma = numpyro.sample(
                    f"{effect_var}_slope_sigma",
                    dist.HalfNormal(prior_hierarchical_sigma_scale)
                )
                raw = numpyro.sample(
                    f"{effects_name}_raw",
                    dist.Normal(0, 1).expand([features[f"num_{group_var}"]]).to_event(1)
                )
                effects = numpyro.deterministic(effects_name, mean + sigma * raw)
                eta = eta + effects[features[f"{group_var}_index"]] * features[effect_var]

        # Save probabilities for bias correction analysis
        probs = numpyro.deterministic("obs_probs", sigmoid(eta))
        
        # Likelihood
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return model


@model
def pairwise_logistic_model_nc_clean(
    hierarchical_effects: Optional[List[Tuple[str, str]]] = None,
    hierarchical_continuous: Optional[List[str]] = None,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_hierarchical_mean_loc: float = 0.0,
    prior_hierarchical_mean_scale: float = 0.5,
    prior_hierarchical_sigma_scale: float = 1.0,
) -> Model:
    """
    Cleaner parameterization: explicit grader-specific model shifts.
    """

    def model(features: Features) -> None:
        continuous = set(hierarchical_continuous or [])
        categorical = set()
        
        for group_var, effect_var in (hierarchical_effects or []):
            categorical.add(group_var if group_var not in continuous else effect_var)
            if effect_var not in continuous:
                categorical.add(effect_var)
        
        required = ["obs"]
        for cat in categorical:
            required.extend([f"{cat}_index", f"num_{cat}"])
        required.extend(continuous)
        
        if 'left_model' in categorical:
            required.extend(['right_model_index', 'num_right_model'])
        
        check_features(features, required)

        # Intercept
        eta = numpyro.sample("intercept", dist.Normal(prior_intercept_loc, prior_intercept_scale))

        # Track model_mean creation
        model_mean_created = False
        model_mean = None
        model_sigma = None

        for group_var, effect_var in (hierarchical_effects or []):
            is_continuous = effect_var in continuous
            effects_name = f"{group_var}_{effect_var}_effects"
            
            if effect_var == 'left_model':
                n_models = features["num_left_model"]
                
                # Create shared model_mean only once
                if not model_mean_created:
                    model_mean = numpyro.sample(
                        "model_mean",
                        dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                        .expand([n_models]).to_event(1)
                    )
                    model_mean_created = True
                
                # NEW: If this is grader × model, create explicit SHIFTS (not deviations with sigma)
                if group_var == 'grader':
                    # Grader-specific shifts from baseline (constrained to sum to 0)
                    shift_sigma = numpyro.sample(
                        "grader_model_shift_sigma",
                        dist.HalfNormal(prior_hierarchical_sigma_scale)
                    )
                    
                    raw_shifts = numpyro.sample(
                        "grader_model_shift_raw",
                        dist.Normal(0, 1).expand([features[f"num_{group_var}"], n_models]).to_event(2)
                    )
                    
                    # Grader-specific shifts (mean 0 across graders for identifiability)
                    shifts = numpyro.deterministic(
                        "grader_model_shift",
                        shift_sigma * raw_shifts
                    )
                    
                    # Total quality = baseline + grader shift
                    left_quality = model_mean[features["left_model_index"]] + \
                                  shifts[features[f"{group_var}_index"], features["left_model_index"]]
                    right_quality = model_mean[features["right_model_index"]] + \
                                   shifts[features[f"{group_var}_index"], features["right_model_index"]]
                    
                    eta = eta + left_quality - right_quality
                
                else:
                    # Question-specific: use hierarchical structure with model_sigma
                    if model_sigma is None:
                        model_sigma = numpyro.sample("model_sigma", dist.HalfNormal(prior_hierarchical_sigma_scale))
                    
                    raw = numpyro.sample(
                        f"{effects_name}_raw",
                        dist.Normal(0, 1).expand([features[f"num_{group_var}"], n_models]).to_event(2)
                    )
                    effects = numpyro.deterministic(effects_name, model_mean[None, :] + model_sigma * raw)
                    
                    eta = eta + effects[features[f"{group_var}_index"], features["left_model_index"]] \
                              - effects[features[f"{group_var}_index"], features["right_model_index"]]
            
            elif is_continuous:
                # Continuous covariates (length, position) - unchanged
                mean = numpyro.sample(
                    f"{effect_var}_slope_mean",
                    dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                )
                sigma = numpyro.sample(
                    f"{effect_var}_slope_sigma",
                    dist.HalfNormal(prior_hierarchical_sigma_scale)
                )
                raw = numpyro.sample(
                    f"{effects_name}_raw",
                    dist.Normal(0, 1).expand([features[f"num_{group_var}"]]).to_event(1)
                )
                effects = numpyro.deterministic(effects_name, mean + sigma * raw)
                eta = eta + effects[features[f"{group_var}_index"]] * features[effect_var]

        # Save probabilities
        probs = numpyro.deterministic("obs_probs", sigmoid(eta))
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return model