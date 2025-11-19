# Quick check
import pandas as pd
import numpy as np

df = pd.read_parquet("output/model/processed_data.parquet")

# Merge GPT-4 and human on same comparisons
gpt4_data = df[df['grader'] == 'gpt4_pair']
human_data = df[df['grader'] == 'human']

merged = gpt4_data.merge(
    human_data,
    on=['question_id', 'left_model', 'right_model'],
    suffixes=('_gpt4', '_human')
)

merged['disagree'] = merged['left_chosen_gpt4'] != merged['left_chosen_human']

print("Where they AGREE:")
print(f"  N = {(~merged['disagree']).sum()}")
print(f"  |length_diff|: mean={merged[~merged['disagree']]['length_diff_prop_gpt4'].abs().mean():.3f}, "
      f"std={merged[~merged['disagree']]['length_diff_prop_gpt4'].abs().std():.3f}")

print("\nWhere they DISAGREE:")
print(f"  N = {merged['disagree'].sum()}")
print(f"  |length_diff|: mean={merged[merged['disagree']]['length_diff_prop_gpt4'].abs().mean():.3f}, "
      f"std={merged[merged['disagree']]['length_diff_prop_gpt4'].abs().std():.3f}")

# Check if they disagree MORE on close length matches
print(f"\nDisagreement rate when |length_diff| < 0.5: {merged[merged['length_diff_prop_gpt4'].abs() < 0.5]['disagree'].mean():.3f}")
print(f"Disagreement rate when |length_diff| > 1.0: {merged[merged['length_diff_prop_gpt4'].abs() > 1.0]['disagree'].mean():.3f}")