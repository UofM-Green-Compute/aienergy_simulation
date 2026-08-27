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
fixed_input_length = 500  # Constant prompt length supplied to the TPS regression model when predicting throughput.
# Calculate lambda parameter for exponential distribution to achieve desired median

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

def model_energy_func(median_tokens):
    """
    Takes in token lengths for realistic scenarios.

    Parameters
    ----------
    token_len : integer
        Token length.

    Returns
    -------
    None
    """
    
    lambda_param = np.log(2) / median_tokens
    
    # Combine both model types for processing
    all_tps_models = {**tps_regression_models, **interpolation_models}
    
    for model_name in all_tps_models.keys():
        
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
        
        
        # Print statistics for each model in the specified order
        for model_name in model_order:
            if model_name in all_model_energies:
                energies = all_model_energies[model_name]
                if not np.isnan(energies).all():
                    p5, p25, p50, p75, p95 = np.percentile(energies, [5, 25, 50, 75, 95])
                    mean = np.mean(energies)
        
        for model_name in model_order:
            if model_name in all_model_tps:
                tps_values = all_model_tps[model_name]
                if not np.isnan(tps_values).all():
                    p5, p25, p50, p75, p95 = np.percentile(tps_values, [5, 25, 50, 75, 95])
                    mean = np.mean(tps_values)
                    std = np.std(tps_values)
        
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
                    
             
                    mixed_energies.extend(final_filtered)
        
    return np.asarray(mixed_energies)
  
### ENERGIES FOR DIFF SCENARIOS ###
email_energy = model_energy_func(120)
summary_energy = model_energy_func(350)
report_energy = model_energy_func(900)
meeting_energy = model_energy_func(2000)

### DATAFRAME FOR SCENARIOS ###
plot_data = pd.concat([
    pd.DataFrame({
        "Energy (Wh)": email_energy,
        "Scenario": "Email \ndrafting"}),
    pd.DataFrame({
        "Energy (Wh)": summary_energy,
        "Scenario": "Document \nsummarisation"}),
    pd.DataFrame({
        "Energy (Wh)": report_energy,
        "Scenario": "Long \nreport"}),
    pd.DataFrame({
        "Energy (Wh)": meeting_energy,
        "Scenario": "Meeting"})],
    ignore_index=True)

plt.figure(figsize=(10, 8))

plt.rcParams.update({
     'font.size': 18,
     'axes.titlesize': 25,
     'axes.labelsize': 20,
     'xtick.labelsize': 18,
     'ytick.labelsize': 18,
     'legend.fontsize': 20,
     'figure.titlesize': 25
 })

# Define colors for each category (same as fig1.py)
color_dict = {
    'Email \ndrafting': '#2ecc71',
    'Document \nsummarisation': '#9b59b6',
    'Long \nreport': '#e67e22',
    'Meeting': '#3498db'
}

ax = sns.violinplot(
    data=plot_data,
    x="Energy (Wh)",
    y="Scenario",
    orient="h",
    inner=None,
    cut=0,
    density_norm="width", 
    palette=color_dict
)

energies_list = [email_energy, summary_energy, report_energy, meeting_energy]

scenario_names = ["Email \ndrafting", "Document \nsummarisation", "Long \nreport", "Meeting"]
#### VIOLIN PLOT #####
for i, energies in enumerate(energies_list):

    p25, p50, p75 = np.percentile(energies,[25,50,75])

    ax.hlines(i,p25,p75,color="black",lw=2)
    ax.vlines(p50,i-0.1,i+0.1,color="white",lw=3)
    ax.vlines(p50,i-0.1,i+0.1,color="black",lw=2)

# Create custom legend entries with median values (same logic as fig1.py)
legend_elements = []
for category, energies in [
    ('Email \ndrafting', email_energy),
    ('Document \nsummarisation', summary_energy),
    ('Long \nreport', report_energy),
    ('Meeting', meeting_energy)
]:
    # Use the same energies that are actually plotted (no additional filtering)
    p25, p50, p75 = np.percentile(energies, [25, 50, 75])
    legend_elements.append(plt.Line2D([0], [0], 
                         label=f'{category.replace(chr(10), " ")}: {p50:.2f} Wh (IQR:{p25:.2f}-{p75:.2f})', 
                         color=color_dict[category], 
                         linewidth=2))

# Add legend with same styling as Figure 1
ax.legend(handles=legend_elements, frameon=True, facecolor='white', 
         edgecolor='none', loc='lower right', bbox_to_anchor=(1, 0),
         fontsize=12)

plt.title('Per-Query Energy Consumption (P5-P95)\n Realistic Scenarios', fontsize=18, pad=20)
plt.xlabel('Energy per Query (Wh)', fontsize=12)
plt.grid(True, alpha=0.3)
