from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from jax.nn import sigmoid
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features


@model
def pwl(
    hierarchical_effects: Optional[List[Union[Tuple[str, str], Tuple[str, str, str]]]] = None,
    hierarchical_continuous: Optional[List[str]] = None,
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_hierarchical_mean_loc: float = 0.0,
    prior_hierarchical_mean_scale: float = 0.5,
    prior_hierarchical_sigma_scale: float = 1.0,
) -> Model:
    """
    Flexible model supporting both 2-way and 3-way hierarchical interactions.
    """

    def model(features: Features) -> None:
        continuous = set(hierarchical_continuous or [])
        categorical = set()
        
        # Parse both 2-way and 3-way effect specifications
        for effect_spec in (hierarchical_effects or []):
            if len(effect_spec) == 2:
                group_var, effect_var = effect_spec
                categorical.add(group_var if group_var not in continuous else effect_var)
                if effect_var not in continuous:
                    categorical.add(effect_var)
            elif len(effect_spec) == 3:
                group_var, cat_var, cont_var = effect_spec
                categorical.add(group_var)
                categorical.add(cat_var)
                continuous.add(cont_var)
        
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

        for effect_spec in (hierarchical_effects or []):
            
            # ========== 2-WAY INTERACTIONS ==========
            if len(effect_spec) == 2:
                group_var, effect_var = effect_spec
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
                    
                    # Grader × model: explicit SHIFTS
                    if group_var == 'grader':
                        shift_sigma = numpyro.sample(
                            "grader_model_shift_sigma",
                            dist.HalfNormal(prior_hierarchical_sigma_scale)
                        )
                        
                        raw_shifts = numpyro.sample(
                            "grader_model_shift_raw",
                            dist.Normal(0, 1).expand([features[f"num_{group_var}"], n_models]).to_event(2)
                        )
                        
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
                        # Question-specific: hierarchical structure
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
                    # Continuous covariates (length, position, length²)
                    # ========== FIXED: Add group_var to parameter names ==========
                    mean = numpyro.sample(
                        f"{group_var}_{effect_var}_slope_mean",  # ✅ CHANGED
                        dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                    )
                    sigma = numpyro.sample(
                        f"{group_var}_{effect_var}_slope_sigma",  # ✅ CHANGED
                        dist.HalfNormal(prior_hierarchical_sigma_scale)
                    )
                    raw = numpyro.sample(
                        f"{effects_name}_raw",
                        dist.Normal(0, 1).expand([features[f"num_{group_var}"]]).to_event(1)
                    )
                    effects = numpyro.deterministic(effects_name, mean + sigma * raw)
                    eta = eta + effects[features[f"{group_var}_index"]] * features[effect_var]
            
            # ========== 3-WAY INTERACTIONS ==========
            elif len(effect_spec) == 3:
                group_var, cat_var, cont_var = effect_spec
                
                # Validate
                assert cat_var in categorical, f"{cat_var} must be categorical"
                assert cont_var in continuous, f"{cont_var} must be continuous"
                
                # Hierarchical 3-way: group × category × continuous
                interaction_mean = numpyro.sample(
                    f"{group_var}_{cat_var}_{cont_var}_mean",
                    dist.Normal(prior_hierarchical_mean_loc, prior_hierarchical_mean_scale)
                )
                
                interaction_sigma = numpyro.sample(
                    f"{group_var}_{cat_var}_{cont_var}_sigma",
                    dist.HalfNormal(prior_hierarchical_sigma_scale)
                )
                
                interaction_raw = numpyro.sample(
                    f"{group_var}_{cat_var}_{cont_var}_raw",
                    dist.Normal(0, 1).expand([
                        features[f"num_{group_var}"],
                        features[f"num_{cat_var}"]
                    ]).to_event(2)
                )
                
                interaction_effects = numpyro.deterministic(
                    f"{group_var}_{cat_var}_{cont_var}_interaction",
                    interaction_mean + interaction_sigma * interaction_raw
                )
                
                # Add to linear predictor
                if cat_var == 'left_model':
                    # Pairwise comparison case
                    left_interaction = interaction_effects[
                        features[f"{group_var}_index"],
                        features["left_model_index"]
                    ]
                    right_interaction = interaction_effects[
                        features[f"{group_var}_index"],
                        features["right_model_index"]
                    ]
                    eta = eta + (left_interaction - right_interaction) * features[cont_var]
                else:
                    # Standard case
                    eta = eta + interaction_effects[
                        features[f"{group_var}_index"],
                        features[f"{cat_var}_index"]
                    ] * features[cont_var]

        # Likelihood
        probs = numpyro.deterministic("obs_probs", sigmoid(eta))
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return model