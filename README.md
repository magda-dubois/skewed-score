# LLM Judges on trial (aka skewed score)

This is a proof of concept for how GLMs can be used to quantify different effects in autograders. 

### Step 1
Run simulate_data.ipynb to get the necessary data

### Step 2
In modeling_per_question, each folder refers to one question in the SkewedScore paper. Each contains the relevant data (from step 1) and a notebook to fit the GLM. The figures for the paper are saved in the folder figures.

### Note
For simplicity, only a few model comparisons were ran but proper model comparisons should be done in practice
