import arviz as az
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

with open('../data/all.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

print(f"Loaded {len(df)} comparisons")

counts = df['question_id'].value_counts().sort_index()
print(f"Min: {counts.min()}")
print(f"Max: {counts.max()}")
print(f"Median: {counts.median()}")
print(f"Mean: {counts.mean()}")
print(f"\nDistribution:")
print(counts.describe())


# Load posterior from your actual file
idata = az.from_netcdf('./output/communicate/models/pairwise_logistic_grader_question_position_length/inference_data.nc')
posterior = idata.posterior

print(f"Posterior shape: {posterior}")
print(f"Available variables: {list(posterior.data_vars)}")

# Get model/grader names
models = posterior['model_mean'].coords['left_model'].values
graders = posterior['grader_model_shift'].coords['grader'].values

print(f"Models: {models}")
print(f"Graders: {graders}")

grader_map = {'gpt4_pair': 0, 'human': 1}

# Extract posterior samples (flatten chains)
n_draws = 1000
print(f"\nExtracting {n_draws} posterior draws...")

posterior_samples = {
    'model_mean': posterior['model_mean'].values.reshape(-1, len(models))[:n_draws],
    'grader_shift': posterior['grader_model_shift'].values.reshape(-1, len(graders), len(models))[:n_draws],
    'position_bias': posterior['grader_position_numeric_effects'].values.reshape(-1, len(graders))[:n_draws],
    'length_bias': posterior['grader_length_diff_prop_effects'].values.reshape(-1, len(graders))[:n_draws],
}

print(f"Sample shapes:")
for key, val in posterior_samples.items():
    print(f"  {key}: {val.shape}")

# Storage for results
actual_by_draw = np.zeros((n_draws, len(models)))
counterfactual_by_draw = np.zeros((n_draws, len(models)))
debiased_by_draw = np.zeros((n_draws, len(models)))

print(f"\nRunning counterfactual simulation...")

# For each posterior draw
for draw_idx in tqdm(range(n_draws)):
    # Extract parameters for this draw
    model_mean = posterior_samples['model_mean'][draw_idx]
    grader_shift = posterior_samples['grader_shift'][draw_idx]
    position_bias = posterior_samples['position_bias'][draw_idx]
    length_bias = posterior_samples['length_bias'][draw_idx]
    
    # === SCENARIO 1: ACTUAL (AS OBSERVED) ===
    actual_scores_per_comparison = []
    for _, row in df.iterrows():
        left_idx = np.where(models == row['left_model'])[0][0]
        right_idx = np.where(models == row['right_model'])[0][0]
        grader_idx = grader_map[row['grader']]
        
        # Quality difference
        quality_diff = (model_mean[left_idx] + grader_shift[grader_idx, left_idx]) - \
                       (model_mean[right_idx] + grader_shift[grader_idx, right_idx])
        
        # Bias contribution (as actually occurred)
        bias_contribution = position_bias[grader_idx] * row['position_numeric'] + \
                           length_bias[grader_idx] * row['length_diff_prop']
        
        actual_scores_per_comparison.append({
            'left_model': row['left_model'],
            'right_model': row['right_model'],
            'score': quality_diff + bias_contribution
        })
    
    # Aggregate by model
    for m_idx, model in enumerate(models):
        left_scores = [s['score'] for s in actual_scores_per_comparison if s['left_model'] == model]
        right_scores = [-s['score'] for s in actual_scores_per_comparison if s['right_model'] == model]
        all_scores = left_scores + right_scores
        if len(all_scores) > 0:
            actual_by_draw[draw_idx, m_idx] = np.mean(all_scores)
    
    # === SCENARIO 2: COUNTERFACTUAL (BALANCED POSITIONS) ===
    np.random.seed(42 + draw_idx)  # Different randomization per draw
    cf_scores_per_comparison = []
    for _, row in df.iterrows():
        left_idx = np.where(models == row['left_model'])[0][0]
        right_idx = np.where(models == row['right_model'])[0][0]
        grader_idx = grader_map[row['grader']]
        
        # Quality difference (same as actual)
        quality_diff = (model_mean[left_idx] + grader_shift[grader_idx, left_idx]) - \
                       (model_mean[right_idx] + grader_shift[grader_idx, right_idx])
        
        # COUNTERFACTUAL: Randomize position (50% left, 50% right)
        cf_position = np.random.choice([+1, -1])
        
        # Bias contribution with randomized position
        bias_contribution = position_bias[grader_idx] * cf_position + \
                           length_bias[grader_idx] * row['length_diff_prop']
        
        cf_scores_per_comparison.append({
            'left_model': row['left_model'],
            'right_model': row['right_model'],
            'score': quality_diff + bias_contribution
        })
    
    # Aggregate by model
    for m_idx, model in enumerate(models):
        left_scores = [s['score'] for s in cf_scores_per_comparison if s['left_model'] == model]
        right_scores = [-s['score'] for s in cf_scores_per_comparison if s['right_model'] == model]
        all_scores = left_scores + right_scores
        if len(all_scores) > 0:
            counterfactual_by_draw[draw_idx, m_idx] = np.mean(all_scores)
    
    # === SCENARIO 3: DEBIASED (NO BIASES AT ALL) ===
    for m_idx, model in enumerate(models):
        # Pure quality: model mean + average grader shift
        debiased_by_draw[draw_idx, m_idx] = model_mean[m_idx] + grader_shift[:, m_idx].mean()

# === AGGREGATE RESULTS ===
print("\nComputing summary statistics...")

results = pd.DataFrame({
    'Model': models,
    'Actual': actual_by_draw.mean(axis=0),
    'Actual_CI': [f"[{np.percentile(actual_by_draw[:, i], 2.5):.3f}, {np.percentile(actual_by_draw[:, i], 97.5):.3f}]" 
                  for i in range(len(models))],
    'Counterfactual': counterfactual_by_draw.mean(axis=0),
    'CF_CI': [f"[{np.percentile(counterfactual_by_draw[:, i], 2.5):.3f}, {np.percentile(counterfactual_by_draw[:, i], 97.5):.3f}]" 
              for i in range(len(models))],
    'Debiased': debiased_by_draw.mean(axis=0),
    'Debiased_CI': [f"[{np.percentile(debiased_by_draw[:, i], 2.5):.3f}, {np.percentile(debiased_by_draw[:, i], 97.5):.3f}]" 
                    for i in range(len(models))],
})

results['Change (Act→CF)'] = results['Counterfactual'] - results['Actual']
results['Change (CF→Deb)'] = results['Debiased'] - results['Counterfactual']

results['Actual_Rank'] = results['Actual'].rank(ascending=False).astype(int)
results['CF_Rank'] = results['Counterfactual'].rank(ascending=False).astype(int)
results['Debiased_Rank'] = results['Debiased'].rank(ascending=False).astype(int)

results['Rank_Change'] = results['Actual_Rank'] - results['CF_Rank']

results = results.sort_values('Actual_Rank')

print("\n" + "="*130)
print("COUNTERFACTUAL SIMULATION: Impact of Position Balance and Bias Removal")
print("="*130)
print(results[['Model', 'Actual', 'Actual_CI', 'Counterfactual', 'CF_CI', 
               'Actual_Rank', 'CF_Rank', 'Rank_Change']].to_string(index=False))
print("\n" + "="*130)

print("\nSummary Statistics:")
print(f"  Rank correlation (Actual vs Counterfactual): {results['Actual_Rank'].corr(results['CF_Rank'], method='spearman'):.3f}")
print(f"  Rank correlation (Counterfactual vs Debiased): {results['CF_Rank'].corr(results['Debiased_Rank'], method='spearman'):.3f}")
print(f"  Models with rank change (Actual→CF): {(results['Rank_Change'] != 0).sum()} / {len(models)}")
print(f"  Max score change (Actual→CF): {abs(results['Change (Act→CF)']).max():.3f}")
print(f"  Max score change (CF→Debiased): {abs(results['Change (CF→Deb)']).max():.3f}")
print(f"  Quality range (top to bottom): {results['Actual'].max() - results['Actual'].min():.3f}")