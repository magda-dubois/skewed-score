import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, log_likelihood, MCMC, NUTS
import arviz as az
import jax
import jax.numpy as jnp


PRIOR_CUTPOINT_DIFFS = dist.LogNormal(-0.5, 0.3)
PRIOR_FIRST_CUTPOINT = dist.Normal(-4.0, 0.2)
PRIOR_GRADER = dist.Normal(0, 1)
PRIOR_H_SIGMA_TYPE = dist.HalfCauchy(1)
PRIOR_H_MEAN_TYPE = dist.Normal(0, 3)
PRIOR_INTERCEPT = dist.Normal(0, 1)
PRIOR_INTERACTION = dist.Normal(0, 1)
PRIOR_MAIN_EFFECTS = dist.Normal(0, 1)
PRIOR_RAW_EFFECT = dist.Normal(0, 1)


def sample_scores(mean, clip_range=(0, 10)):

    # n_samples is randomly chosen
    n_samples = np.random.randint(80, 100)

    # std
    std = 0.8

    # Generate scores using a normal distribution, then clip and round them
    scores = np.clip(np.random.normal(loc=mean, scale=std, size=n_samples), *clip_range).round().astype(int)

    return scores


def encode_categorical_covariates(df, covariates):
    covariates_dict = {}
    category_maps = {}
    for col in covariates:
        categories = pd.Categorical(df[col])
        codes = categories.codes
        covariates_dict[col] = codes
        covariates_dict[f'n_{col}s'] = len(categories.categories)
        category_maps[col] = dict(enumerate(categories.categories))
    return covariates_dict, category_maps


def create_interaction_effects_effectCoded(name1, name2, covariates_dict, prior):
    n1 = covariates_dict[f'n_{name1}s']
    n2 = covariates_dict[f'n_{name2}s']
    
    if n1 == 1 or n2 == 1:
        return jnp.zeros((n1, n2))

    raw = numpyro.sample(
        f'b_interaction_{name1}_{name2}',
        prior.expand([(n1 - 1) * (n2 - 1)])
    ).reshape((n1 - 1, n2 - 1))

    b_full = jnp.zeros((n1, n2)).at[:n1 - 1, :n2 - 1].set(raw)
    b_full = b_full.at[n1 - 1, :n2 - 1].set(-jnp.sum(b_full[:n1 - 1, :n2 - 1], axis=0))
    b_full = b_full.at[:, n2 - 1].set(-jnp.sum(b_full[:, :n2 - 1], axis=1))

    numpyro.deterministic(f'b_interaction_{name1}_{name2}_full', b_full)
    return b_full


def create_interaction_effects_dummyCoded(name1, name2, covariates_dict, prior):
    n1 = covariates_dict[f'n_{name1}s']
    n2 = covariates_dict[f'n_{name2}s']

    interaction_matrix = numpyro.sample(
        f"b_interaction_{name1}_{name2}",
        prior.expand([n1 * n2])
    ).reshape((n1, n2))

    numpyro.deterministic(f"b_interaction_{name1}_{name2}_full", interaction_matrix)
    return interaction_matrix


def ordered_logistic_model(
    covariates_dict,
    y,
    main_effects=None,
    interactions=None,
    num_classes=11,
    prior=PRIOR_MAIN_EFFECTS,
    interaction_prior=PRIOR_INTERACTION,
    effect_coding_for_main_effects=True
):
    main_effects = main_effects or []
    interactions = interactions or []

    intercept = numpyro.sample("intercept", PRIOR_INTERCEPT)
    eta = intercept

    # Add main effects
    for effect in main_effects:
        n_levels = covariates_dict[f"n_{effect}s"]
        idx = covariates_dict[effect]
        if effect_coding_for_main_effects:
            free_coefs = numpyro.sample(f"b_{effect}", prior.expand([n_levels - 1]))
            last_coef = -jnp.sum(free_coefs)
            coefs = jnp.concatenate([free_coefs, jnp.array([last_coef])])
            numpyro.deterministic(f"b_{effect}_full", coefs) 
        else:
            coefs = numpyro.sample(f"b_{effect}", prior.expand([n_levels]))
        eta += coefs[idx]

    # Add interactions
    for name1, name2 in interactions:
        interaction_matrix = create_interaction_effects_dummyCoded(name1, name2, covariates_dict, prior=interaction_prior)
        idx1 = covariates_dict[name1]
        idx2 = covariates_dict[name2]
        eta += interaction_matrix[idx1, idx2]

    # Sample first cutpoint with tighter prior
    first_cutpoint = numpyro.sample("first_cutpoint", PRIOR_FIRST_CUTPOINT)

    # Use LogNormal with carefully tuned parameters
    cutpoint_diffs = numpyro.sample(
        "cutpoint_diffs",
        PRIOR_CUTPOINT_DIFFS,  
        sample_shape=(num_classes - 2,)
    )

    # Ensure minimum spacing between cutpoints
    min_spacing = 0.3
    adjusted_diffs = cutpoint_diffs + min_spacing

    # Build cutpoints
    cutpoints = jnp.concatenate([
        jnp.array([first_cutpoint]),
        first_cutpoint + jnp.cumsum(adjusted_diffs)
    ])

    # Likelihood
    numpyro.sample("obs", dist.OrderedLogistic(eta, cutpoints), obs=y)

def hierarchical_ordered_logistic_model(
    covariates_dict,
    y,
    main_effects=None,
    interactions=None,
    num_classes=11,
    prior=PRIOR_MAIN_EFFECTS,
    interaction_prior=PRIOR_INTERACTION,
    effect_coding_for_main_effects=True
):
    """
    Hierarchical ordered logistic regression model with separate means and variances
    for each grader type. Uses non-centered parameterization for better sampling.
    
    Parameters:
    -----------
    covariates_dict : dict
        Dictionary containing covariate data and metadata
    y : array
        Outcome variable (ordinal responses)
    main_effects : list
        List of variables to include as main effects (should include 'LLM' but not 'grader')
    interactions : list of tuples
        List of pairs of variables to include as interactions
    num_classes : int
        Number of ordinal classes in the outcome
    prior : numpyro distribution
        Prior for main effects
    interaction_prior : numpyro distribution
        Prior for interaction effects
    effect_coding_for_main_effects : bool
        Whether to use effect coding (sum-to-zero constraints)
    """
    main_effects = main_effects or []
    interactions = interactions or []

    intercept = numpyro.sample("intercept", PRIOR_INTERCEPT)
    eta = intercept

    # Make sure grader_type exists in the covariates dictionary
    if "grader_type" not in covariates_dict:
        raise ValueError("'grader_type' must be included in covariates_dict for hierarchical model")
    
    # Handle grader effect hierarchically
    if "grader" in covariates_dict:
        # Remove grader from main effects if it's there
        main_effects = [e for e in main_effects if e != "grader"]
        
        # Sample explicit means for grader types using weaker priors
        type_0_mean = numpyro.sample("b_grader_type_0", PRIOR_H_MEAN_TYPE)
        type_1_mean = numpyro.sample("b_grader_type_1", PRIOR_H_MEAN_TYPE)
        
        # Create a vector of means indexed by type
        grader_type_means = jnp.array([type_0_mean, type_1_mean])
        
        # Sample type-specific standard deviations with HalfCauchy prior
        sigma_type_0 = numpyro.sample("sigma_grader_type_0", PRIOR_H_SIGMA_TYPE)
        sigma_type_1 = numpyro.sample("sigma_grader_type_1", PRIOR_H_SIGMA_TYPE)
        
        # Create vector of sigmas indexed by type
        grader_type_sigmas = jnp.array([sigma_type_0, sigma_type_1])
        
        # Create mapping from grader to grader_type
        grader_to_type = {}
        for i in range(len(covariates_dict["grader"])):
            grader_idx = covariates_dict["grader"][i]
            type_idx = covariates_dict["grader_type"][i]
            grader_to_type[grader_idx] = type_idx
        
        # Sample individual grader effects with non-centered parameterization
        n_graders = covariates_dict["n_graders"]
        grader_effects = []
        
        for i in range(n_graders):
            # Get this grader's type (default to 0 if not found)
            type_idx = grader_to_type.get(i, 0)
            
            # Non-centered parameterization
            raw_effect = numpyro.sample(f"b_grader_raw_{i}", PRIOR_RAW_EFFECT)
            effect = grader_type_means[type_idx] + grader_type_sigmas[type_idx] * raw_effect
            numpyro.deterministic(f"b_grader_{i}", effect)
            grader_effects.append(effect)
        
        # Combine all grader effects
        grader_effects = jnp.array(grader_effects)
        numpyro.deterministic("b_grader", grader_effects)
        
        # Also save the difference between type 1 and type 0 means
        numpyro.deterministic("type_1_vs_type_0", type_1_mean - type_0_mean)
        
        # Add grader effect to linear predictor
        grader_idx = covariates_dict["grader"]
        eta += grader_effects[grader_idx]
    
    # Process regular main effects
    for effect in main_effects:
        n_levels = covariates_dict[f"n_{effect}s"]
        idx = covariates_dict[effect]
        
        if effect_coding_for_main_effects:
            # Effect coding (sum-to-zero constraint)
            free_coefs = numpyro.sample(f"b_{effect}", prior.expand([n_levels - 1]))
            last_coef = -jnp.sum(free_coefs)
            coefs = jnp.concatenate([free_coefs, jnp.array([last_coef])])
            numpyro.deterministic(f"b_{effect}_full", coefs)
        else:
            # Regular dummy coding
            coefs = numpyro.sample(f"b_{effect}", prior.expand([n_levels]))
            
        eta += coefs[idx]

    # Add interactions
    for name1, name2 in interactions:
        if name1 == "grader" and name2 == "LLM" or name1 == "LLM" and name2 == "grader":
            # Handle grader-LLM interaction specially
            n_graders = covariates_dict["n_graders"]
            n_llms = covariates_dict["n_LLMs"]
            
            # Sample interaction effects
            grader_llm_interaction = numpyro.sample(
                "b_grader_LLM", 
                interaction_prior.expand([n_graders, n_llms])
            )
            
            # Add to linear predictor
            grader_idx = covariates_dict["grader"]
            llm_idx = covariates_dict["LLM"]
            eta += grader_llm_interaction[grader_idx, llm_idx]
        else:
            # Handle other interactions
            interaction_matrix = create_interaction_effects_effectCoded(name1, name2, covariates_dict, prior=interaction_prior)
            idx1 = covariates_dict[name1]
            idx2 = covariates_dict[name2]
            eta += interaction_matrix[idx1, idx2]

    # Sample first cutpoint
    first_cutpoint = numpyro.sample("first_cutpoint", PRIOR_FIRST_CUTPOINT)

    # Use LogNormal
    cutpoint_diffs = numpyro.sample(
        "cutpoint_diffs",
        PRIOR_CUTPOINT_DIFFS,  
        sample_shape=(num_classes - 2,)
    )

    # Ensure minimum spacing between cutpoints
    min_spacing = 0.3
    adjusted_diffs = cutpoint_diffs + min_spacing

    # Build cutpoints
    cutpoints = jnp.concatenate([
        jnp.array([first_cutpoint]),
        first_cutpoint + jnp.cumsum(adjusted_diffs)
    ])

    # Likelihood
    numpyro.sample("obs", dist.OrderedLogistic(eta, cutpoints), obs=y)

def fit_model_and_extract_idata(
    rng_key, covariates_dict, y, main_effects, interactions, 
    category_maps, model=ordered_logistic_model
):
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000, num_chains=1)
    mcmc.run(
        rng_key, 
        covariates_dict, 
        y=y, 
        main_effects=main_effects, 
        interactions=interactions
    )

    coords, dims = build_model_metadata(category_maps, main_effects, interactions)

    idata = get_idata(
        mcmc, 
        covariates_dict, 
        y, 
        main_effects, 
        interactions, 
        coords=coords, 
        dims=dims,
        model=model 
    )
    return mcmc, idata

def get_idata(mcmc, covariates_dict, y, main_effects, interactions=None, coords=None, dims=None, model=ordered_logistic_model):
    """
    Extract inference data from MCMC results.
    
    Parameters:
    -----------
    mcmc : numpyro.infer.MCMC
        MCMC object with samples
    covariates_dict : dict
        Dictionary of covariates
    y : array
        Observed data
    main_effects : list
        Main effects in the model
    interactions : list, optional
        Interactions in the model
    coords : dict, optional
        Coordinates for InferenceData
    dims : dict, optional
        Dimensions for InferenceData
    model : callable, optional
        Model function to use for posterior predictive and log-likelihood
    """
    # Ensure interactions is a list if None
    interactions = interactions or []

    # Posterior samples
    posterior_samples = mcmc.get_samples()
    posterior_reshaped = {
        k: np.expand_dims(np.array(v), axis=0)
        for k, v in posterior_samples.items()
    }

    # Posterior predictive
    ppc_key = jax.random.PRNGKey(1)
    predictive = Predictive(
        model,  # Use the provided model
        posterior_samples=posterior_samples,
        return_sites=["obs"]
    )
    ppc_samples = predictive(
        ppc_key,
        covariates_dict=covariates_dict,
        y=None,
        main_effects=main_effects,
        interactions=interactions
    )
    posterior_predictive = {"obs": np.expand_dims(np.array(ppc_samples["obs"]), axis=0)}

    # Prior predictive
    prior_key = jax.random.PRNGKey(2)
    prior_samples = Predictive(
        model,  
        num_samples=500,
        return_sites=["obs"]
    )(prior_key, covariates_dict=covariates_dict, y=None,
      main_effects=main_effects, interactions=interactions)
    prior_predictive = {"obs": np.expand_dims(np.array(prior_samples["obs"]), axis=0)}

    # Observed data
    observed = {"obs": y.astype(np.int64)}

    # Compute log_likelihood when not doing prior predictive checks
    if y is not None:
        log_liks = log_likelihood(
            model,  
            posterior_samples,
            covariates_dict=covariates_dict,
            y=y,
            main_effects=main_effects,
            interactions=interactions
        )
        log_liks_reshaped = {
            k: np.expand_dims(np.array(v), axis=0) for k, v in log_liks.items()
        }
    else:
        log_liks_reshaped = None

    # InferenceData
    idata = az.from_dict(
        posterior=posterior_reshaped,
        posterior_predictive=posterior_predictive,
        prior_predictive=prior_predictive,
        observed_data=observed,
        log_likelihood=log_liks_reshaped,
        coords=coords,
        dims=dims
    )

    return idata

def build_model_metadata(category_maps, main_effects, interactions=None, effect_coding=True):
    coords = {}
    dims = {}

    # ensure lists
    interactions = interactions or []

    # coordinates for ordinal-model cutpoints
    cutpoint_labels = [f"{i+1}-{i}" for i in range(9)]
    coords["cutpoint_diffs_dim_0"] = cutpoint_labels
    dims["cutpoint_diffs"] = ["cutpoint_diffs_dim_0"]

    # Main effects
    for effect in main_effects:
        if effect == "length_diff":
            continue  # length_diff numeric, explicitly handled below
        labels = list(category_maps[effect].values())
        coords[effect] = labels
        n_levels = len(labels)
        has_multiple_levels = n_levels > 1
        var_full = f"b_{effect}_full"
        var_contrast = f"b_{effect}"

        if effect_coding and has_multiple_levels:
            dims[var_full] = [effect]
        elif not effect_coding and has_multiple_levels:
            dims[var_contrast] = [effect]

    # interactions handled as usual
    for eff1, eff2 in interactions:
        coords[eff1] = list(category_maps[eff1].values())
        coords[eff2] = list(category_maps[eff2].values())
        dims[f"b_interaction_{eff1}_{eff2}_full"] = [eff1, eff2]

    # explicit handling of length slope (numeric, grader-specific)
    if "length_diff" in main_effects:
        dims["b_grader_length_slope_full"] = ["grader"]

    return coords, dims

def pairwise_logistic_model(covariates_dict, y=None, main_effects=None, interactions=None):

    # Number of data points
    n_data = covariates_dict["llm_pair"].shape[0]

    # Intercept
    intercept = numpyro.sample("intercept", dist.Normal(0, 1))
    linear_predictor = jnp.full((n_data,), intercept)

    # LLM pair effect
    n_pairs = covariates_dict["n_llm_pairs"]
    b_llm_pair_full = numpyro.sample(
        "b_llm_pair_full", dist.Normal(0, 1).expand([n_pairs]).to_event(1)
    )
    idx_pairs = covariates_dict["llm_pair"]
    linear_predictor += b_llm_pair_full[idx_pairs]

    # Grader effect
    n_graders = covariates_dict["n_graders"]
    b_grader_full = numpyro.sample(
        "b_grader_full", dist.Normal(0, 1).expand([n_graders]).to_event(1)
    )
    idx_graders = covariates_dict["grader"]
    linear_predictor += b_grader_full[idx_graders]

    # Grader-specific length slope IF length_diff in main_effects
    if 'length_diff' in main_effects:
        length_slope_mean = numpyro.sample("length_slope_mean", dist.Normal(0, 0.5))
        sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(1.0))
        b_grader_length_slope_full = numpyro.sample(
            "b_grader_length_slope_full",
            dist.Normal(length_slope_mean, sigma_slope).expand([n_graders]).to_event(1)
        )

        # Numeric length_diff array
        length_diff = covariates_dict["length_diff"]
        linear_predictor += b_grader_length_slope_full[idx_graders] * length_diff

    # Logistic link
    p = jax.nn.sigmoid(linear_predictor)
    numpyro.sample("obs", dist.Bernoulli(p), obs=y)