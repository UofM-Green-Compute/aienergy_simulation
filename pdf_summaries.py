"""
This script estimates energy consumption per query for open-weight
language models running on H100 GPUs in the traditional-query regime.

The simulation uses Monte Carlo sampling to estimate the distribution
of energy per query, accounting for variability in model throughput,
query characteristics and system efficiency.

Inputs:
- model_throughput_DB.csv: throughput data for different models

Outputs:
- Energy per query estimates
- Figures corresponding to the traditional-query analysis
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression

# Set larger font sizes globally for all figures
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 25,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'figure.titlesize': 25
})

# Set seaborn style with larger fonts
sns.set_context("poster", font_scale=3.0)
sns.set_style("white")

# Seed for reproducibility
np.random.seed(42)

# Load model_throughput_consolidated_DB.csv
model_throughput = pd.read_csv('model_throughput_DB.csv')
print(model_throughput.columns)

# Drop those with Quantization != "FP8"
# Paper for traditional queries makes the assumption that we are working with 
# FP8, balances memory and speed.
model_throughput = model_throughput[model_throughput['Quantization'] == 'FP8']

#%%
# Helper function to get log-normal parameters from (5th, 95th percentile)
def lognorm_params(min_val, max_val):
    """
    Calculates parameters for a log-normal distribution.
    
    The uncertainty range is defined using the 5th and 95th percentiles.
    The value 1.645 corresponds to the 95% confidence interval of a
    standard normal distribution.
    
    Convert minimum and maximum values into parameters of a log-normal
    distribution. The method assumes that min_val and max_val represent
    approximately the 5th and 95th percentiles of the distribution.1.645 is
    the 95th percentile of a standard normal distribution. For a normal 
    distribution, ~90% of values lie between mean - 1.645*sigma and mean
    + 1.645*sigma. 
    """
    sigma = (np.log(max_val) - np.log(min_val)) / (2 * 1.645)
    mu = np.log(min_val) + 1.645 * sigma
    return mu, sigma

def create_tps_regression_models(model_data):
    """
    Create regression models for each model to predict TPS from 
    input/output lengths
    """
    regression_models = {}
    interpolation_models = {}
    max_tps_values = {}  # Store maximum TPS for each model
    
    # Get numeric TPS values
    model_data['TPS_numeric'] = pd.to_numeric(model_data['Tokens per Second (TPS)'], errors='coerce')
    
    # Convert Input Length and Output Length to numeric (handle non-numeric values)
    model_data['Input_Length_numeric'] = pd.to_numeric(model_data['Input Length'], errors='coerce')
    model_data['Output_Length_numeric'] = pd.to_numeric(model_data['Output Length'], errors='coerce')
    
    print("Building TPS Models:")
    print("=" * 50)
    
    for model_name in model_data['Model'].unique():
        model_subset = model_data[model_data['Model'] == model_name].copy()
        
        # Remove any rows with NaN values
        model_subset = model_subset.dropna(subset=['Input_Length_numeric', 'Output_Length_numeric', 'TPS_numeric'])
        
        if len(model_subset) < 2:  # Need at least 2 points for interpolation
            print(f"{model_name}: Insufficient data points ({len(model_subset)}), skipping")
            continue
        
        # Store the maximum TPS value for this model (for capping predictions)
        max_tps_values[model_name] = model_subset['TPS_numeric'].max()
        
        if len(model_subset) < 3:  # Use highest TPS for 2 points
            print(f"{model_name}: Using highest TPS (only {len(model_subset)} data points)")
            
            # Use the highest TPS value from available data
            max_tps = model_subset['TPS_numeric'].max()  # Changed from mean to max
            
            interpolation_models[model_name] = {
                'type': 'max_tps',
                'max_tps': max_tps,
                'n_points': len(model_subset),
                'tps_range': (model_subset['TPS_numeric'].min(), model_subset['TPS_numeric'].max())
            }
            
            print(f"  Data points: {len(model_subset)}")
            print(f"  TPS range: {model_subset['TPS_numeric'].min():.2f} - {model_subset['TPS_numeric'].max():.2f}")
            print(f"  Using max TPS: {max_tps:.2f}")
            print()
            
        else:  # Use regression for 3+ points
            print(f"{model_name}: Using regression ({len(model_subset)} data points)")
            
            # Features: [Input_Length, Output_Length]
            X = model_subset[['Input_Length_numeric', 'Output_Length_numeric']].values
            y = model_subset['TPS_numeric'].values
            
            # Log-linear regression (handle zeros by adding small epsilon)
            X_log = np.log(np.maximum(X, 1e-6))  # Avoid log(0)
            y_log = np.log(np.maximum(y, 1e-6))
            
            # Fit regression model
            reg = LinearRegression()
            reg.fit(X_log, y_log)
            
            regression_models[model_name] = {
                'type': 'regression',
                'model': reg,
                'n_points': len(model_subset),
                'input_range': (X[:, 0].min(), X[:, 0].max()),
                'output_range': (X[:, 1].min(), X[:, 1].max()),
                'max_tps': max_tps_values[model_name]  # Store max TPS for capping
            }
            
            print(f"  Data points: {len(model_subset)}")
            print(f"  Input length range: {X[:, 0].min():.0f} - {X[:, 0].max():.0f}")
            print(f"  Output length range: {X[:, 1].min():.0f} - {X[:, 1].max():.0f}")
            print(f"  Max TPS (cap): {max_tps_values[model_name]:.2f}")
            print()
    
    return regression_models, interpolation_models, max_tps_values

def predict_tps_for_lengths(model_name, input_length, output_length, regression_models, interpolation_models, max_tps_values):
    """
    Predict TPS for given input and output lengths using regression 
    or interpolation, capped at max observed TPS
    """
    
    # Try regression first
    if model_name in regression_models:
        # Log transform the features
        log_features = np.array([[np.log(max(input_length, 1e-6)), np.log(max(output_length, 1e-6))]])
        
        # Predict log(TPS)
        log_tps_pred = regression_models[model_name]['model'].predict(log_features)[0]
        
        # Transform back to original scale
        predicted_tps = np.exp(log_tps_pred)
        
        # Cap at maximum observed TPS for this model
        max_tps = regression_models[model_name]['max_tps']
        return min(predicted_tps, max_tps)
    
    # Try interpolation fallback
    elif model_name in interpolation_models:
        model_info = interpolation_models[model_name]
        
        if model_info['type'] == 'max_tps':
            # Use the maximum TPS value (already capped by design)
            return model_info['max_tps']
    
    else:
        return None

# Build regression models
tps_regression_models, interpolation_models, max_tps_values = create_tps_regression_models(model_throughput)

# Simulation settings --- THESE ARE THE ONES TO CHANGE
n_runs = 10000 # large number of repeats so allows for statistical calculations
median_output_tokens = 300   # Median of the exponential distribution used to randomly sample input prompt lengths.
fixed_input_length = 300  # Constant prompt length supplied to the TPS regression model when predicting throughput.
# Calculate lambda parameter for exponential distribution to achieve desired median
lambda_param = np.log(2) / median_output_tokens  # For exponential, median = ln(2)/λ

# Define ranges and values
def get_node_power(model_name):
    """
    Node power from table of values.
    """
    return 12.8 if model_name == 'DeepSeek-R1' else 10.2

pu_range = (0.4, 0.9)        # for lognormal, as 0.7Pmax is where it is centred
PUE_range = (1.05, 1.6)       # for lognormal, as PUE ranges between those values

# Power Usage Effectiveness (PUE) accounts for additional data centre
# energy overhead such as cooling and power distribution losses.

# Compute log-normal parameters where needed
mu_pu, sigma_pu = lognorm_params(*pu_range)
mu_pue, sigma_pue = lognorm_params(*PUE_range)

# Create separate distributions for each model using regression
all_model_energies = {}
all_model_tps = {}  # Add this to store TPS predictions
print("\nGenerating Energy Distributions:")
print("=" * 50)

# Combine both model types for processing
all_tps_models = {**tps_regression_models, **interpolation_models}

for model_name in all_tps_models.keys():
    print(f"Processing {model_name}...")
    
    # Get model-specific node power
    node_power = get_node_power(model_name)
    
    # Generate random output token lengths (exponential distribution)
    # Generate random output response lengths from an exponential distribution.
    # The distribution is parameterised so that the median generated response
    # contains 'median_output_tokens' tokens.
    model_token_lengths = np.round(np.random.exponential(1/lambda_param, n_runs)).astype(int)
    
    # Estimate throughput for each simulated query.
    # The input prompt length is held constant (300 tokens), while the
    # generated response length varies between Monte Carlo samples.
    model_tokens_per_sec = np.array([
        predict_tps_for_lengths(model_name, fixed_input_length, token_length, tps_regression_models, interpolation_models, max_tps_values)
        for token_length in model_token_lengths
    ])
    
    # Handle any None values (fallback to mean if needed)
    valid_tps = model_tokens_per_sec[model_tokens_per_sec != None]
    if len(valid_tps) == 0:
        print(f"  Warning: No valid TPS predictions for {model_name}, skipping")
        continue
    
    model_tokens_per_sec = model_tokens_per_sec.astype(float)
    
    # Store TPS predictions for this model
    all_model_tps[model_name] = model_tokens_per_sec
    
    # Calculate energies for this model
    model_node_power_array = np.full(n_runs, node_power)  # Use model-specific power value
    model_pu = np.random.lognormal(mu_pu, sigma_pu, n_runs)
    model_pue = np.random.lognormal(mu_pue, sigma_pue, n_runs)
    
    # Calculate base energies for this model
    model_energies = np.empty(n_runs)
    
    # Energy per query is calculated by dividing server power consumption
    # by the number of queries processed per second (throughput).
    for i in range(n_runs):
        energy_kj = model_pue[i] * (model_node_power_array[i] * model_pu[i] * (model_token_lengths[i])) / model_tokens_per_sec[i]
        model_energies[i] = (energy_kj / 3600) * 1000 # conversion factor from kj to Wh
    
    all_model_energies[model_name] = model_energies
    print(f"  Generated {n_runs} energy samples")

# Rename the key in all_model_energies and all_model_tps to match the renamed model label
if 'DeepSeek-R1' in all_model_energies:
    all_model_energies['DeepSeek-R1 671B'] = all_model_energies.pop('DeepSeek-R1')
if 'DeepSeek-R1' in all_model_tps:
    all_model_tps['DeepSeek-R1 671B'] = all_model_tps.pop('DeepSeek-R1')
if 'Llama-3.1 Nemotron Ultra 253B' in all_model_energies:
    all_model_energies['Llama-3.1 Nemotron\nUltra 253B'] = all_model_energies.pop('Llama-3.1 Nemotron Ultra 253B')
if 'Llama-3.1 Nemotron Ultra 253B' in all_model_tps:
    all_model_tps['Llama-3.1 Nemotron\nUltra 253B'] = all_model_tps.pop('Llama-3.1 Nemotron Ultra 253B')

# Create violin plots for each model
plot_data_list = []
for model_name, energies in all_model_energies.items():
    if not np.isnan(energies).all():  # Skip models with all NaN values
        # Filter outliers (5-95 percentile) - this filtered data will be used for both KDE and boxplot
        p5, p95 = np.percentile(energies, [5, 95])
        filtered_energies = energies[(energies >= p5) & (energies <= p95)]  # Changed to inclusive bounds
        
        df_model = pd.DataFrame({
            'Energy (Wh)': filtered_energies,
            'Model': model_name
        })
        plot_data_list.append(df_model)

# Combine all model data
plot_data_combined = pd.concat(plot_data_list)

# Define the desired order of models
model_order = [
    'DeepSeek-R1 671B',
    'Llama 3.1 405B',
    'Llama-3.1 Nemotron\nUltra 253B',
    'Mixtral 8x22B',
    'Llama 3.1 70B'
]

# Rename label of DeepSeek-R1 to DeepSeek-R1 671B
plot_data_combined['Model'] = plot_data_combined['Model'].replace('DeepSeek-R1', 'DeepSeek-R1 671B')
# Rename label to add line break for better display
plot_data_combined['Model'] = plot_data_combined['Model'].replace('Llama-3.1 Nemotron Ultra 253B', 'Llama-3.1 Nemotron\nUltra 253B')

# Create a custom color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
color_dict = dict(zip(model_order, colors))


# Print regression models first
for model_name in sorted(tps_regression_models.keys()):
    model_info = tps_regression_models[model_name]
    n_points = model_info['n_points']

    # Show regression coefficients (log-linear model interpretation)
    coef = model_info['model'].coef_
    intercept = model_info['model'].intercept_
    
# Print interpolation models
max_tps_models_found = False
for model_name in sorted(interpolation_models.keys()):
    max_tps_models_found = True
    model_info = interpolation_models[model_name]
    n_points = model_info['n_points']
# %%

#%%
# Figure 2: Mixed distribution of top 3 models (by highest median energy consumption)

# Calculate median energy for each model to determine top 3
model_medians = {}
for model_name in model_order:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            median_energy = np.median(energies)
            model_medians[model_name] = median_energy

# Sort models by median energy (descending - most energy intensive first)
sorted_models = sorted(model_medians.items(), key=lambda x: x[1], reverse=True)
top_3_models = [model[0] for model in sorted_models[:3]]

print(f"Top 3 most energy intensive models (by median energy consumption):")
for i, model_name in enumerate(top_3_models, 1):
    median_energy = model_medians[model_name]
    print(f"{i}. {model_name}: {median_energy:.3f} Wh")

# Combine energy distributions from top 3 models
# Reduces noise in tail of violing plot for good visualization, does not change the results
additional_filtering = True  # Set to False to use only standard 5-95 filtering

mixed_energies = []

for model_name in top_3_models:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            
            # Stage 1: Standard 5-95 percentile filtering (consistent with Figure 1)
            p5, p95 = np.percentile(energies, [5, 95])
            stage1_filtered = energies[(energies >= p5) & (energies <= p95)]
            
            # Stage 2: Additional filtering on the already filtered data
            if additional_filtering:
                # Apply 10-90 percentile filtering on the Stage 1 filtered data
                p10_stage2, p90_stage2 = np.percentile(stage1_filtered, [10, 90])
                final_filtered = stage1_filtered[(stage1_filtered >= p10_stage2) & (stage1_filtered <= p90_stage2)]
            else:
                final_filtered = stage1_filtered
            
            # Show the effect of both filtering stages
            original_std = np.std(energies)
            stage1_std = np.std(stage1_filtered)
            final_std = np.std(final_filtered)
            
            stage1_reduction = (1 - stage1_std/original_std) * 100
            total_reduction = (1 - final_std/original_std) * 100
            
            
            mixed_energies.extend(final_filtered)

mixed_energies = np.array(mixed_energies)

# =============================================================================
# DOCUMENT SIZE SENSITIVITY (Baseline only)
# =============================================================================

baseline_model = "DeepSeek-R1"

node_power = get_node_power(baseline_model)

# Reuse uncertainty distributions
pu = np.random.lognormal(mu_pu, sigma_pu, n_runs)
pue = np.random.lognormal(mu_pue, sigma_pue, n_runs)

# =============================================================================
# DOCUMENT SIZE SENSITIVITY (Baseline only)
# =============================================================================

baseline_model = "DeepSeek-R1"

node_power = get_node_power(baseline_model)

# Reuse uncertainty distributions
pu = np.random.lognormal(mu_pu, sigma_pu, n_runs)
pue = np.random.lognormal(mu_pue, sigma_pue, n_runs)

# -------------------------------------------------------------------------
# Assumptions
# -------------------------------------------------------------------------

WORDS_PER_PAGE = 500
TOKENS_PER_WORD = 1.17
SUMMARY_WORDS_PER_PAGE = 100   # 20% compression

num_of_pdf = np.arange(1, 11)
pages = np.arange(1, 31)
median_energy = np.zeros((len(num_of_pdf), len(pages)))

for j, n_pdf in enumerate(num_of_pdf):
    
    for i, n_pages in enumerate(pages):
    
        input_tokens = n_pdf * n_pages * WORDS_PER_PAGE * TOKENS_PER_WORD
        output_tokens = input_tokens * 0.2
    
        tps = predict_tps_for_lengths(
            baseline_model,
            input_tokens,
            output_tokens,
            tps_regression_models,
            interpolation_models,
            max_tps_values
        )
    
        energies = (
            pue
            * node_power
            * pu
            * output_tokens
            / tps
        ) * 1000 / 3600
    
        median_energy[j, i] = np.median(energies)
    
with plt.rc_context({
    "font.size":18,
    "axes.titlesize":20,
    "axes.labelsize":18,
    "xtick.labelsize":10,
    "ytick.labelsize":10
}):

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        median_energy,
        xticklabels=pages.astype(int),
        yticklabels=num_of_pdf.astype(int),
        cmap="viridis",
        cbar_kws={"label":"Median energy (Wh)"}
    )
    
    # Label axes
# %%
# %%
    plt.xlabel("Number of pages")


    plt.ylabel("Number of PDFs summarised")
    plt.title("Parameter Sweep and energy density")
    
    # Improve tick label density
# %%
    plt.xticks(rotation=30)

    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
# Define improvement multipliers (same as fig1.py)
mu_hardware, sigma_hardware = lognorm_params(1.5, 2.5)  # Hardware multiplier
mu_algorithm, sigma_algorithm = lognorm_params(1.5, 10)  # Algorithm/Model multiplier
mu_improved, sigma_improved = lognorm_params(1.5, 5)  # Improved serving multiplier

# Apply improvements by dividing energy by multipliers (since multipliers increase efficiency)
# Note: Apply multipliers to the already-filtered baseline data
n_mixed_samples = len(mixed_energies)
hardware_multiplier = np.random.lognormal(mu_hardware, sigma_hardware, n_mixed_samples)
algorithm_multiplier = np.random.lognormal(mu_algorithm, sigma_algorithm, n_mixed_samples)
improved_multiplier = np.random.lognormal(mu_improved, sigma_improved, n_mixed_samples)

hardware_energies = mixed_energies / hardware_multiplier
algorithm_energies = mixed_energies / algorithm_multiplier
improved_energies = mixed_energies / improved_multiplier
