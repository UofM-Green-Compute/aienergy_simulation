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
from scipy.stats import lognorm, pareto, t

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

model_throughput = model_throughput[model_throughput['Quantization'] == 'FP8']

#%%
# Helper function to get log-normal parameters from (5th, 95th percentile)
def lognorm_params(min_val, max_val):
    """
    Calculates parameters for a log-normal distribution.
    
    The uncertainty range is defined using the 5th and 95th percentiles.
    The value 1.645 corresponds to the 95% confidence interval of a
    standard normal distribution.
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
            continue
        
        # Store the maximum TPS value for this model (for capping predictions)
        max_tps_values[model_name] = model_subset['TPS_numeric'].max()
        
        if len(model_subset) < 3:  # Use highest TPS for 2 points
            max_tps = model_subset['TPS_numeric'].max()  # Changed from mean to max
            
            interpolation_models[model_name] = {
                'type': 'max_tps',
                'max_tps': max_tps,
                'n_points': len(model_subset),
                'tps_range': (model_subset['TPS_numeric'].min(), model_subset['TPS_numeric'].max())
            }
            
        else:
            # Features: [Input_Length, Output_Length]
            X = model_subset[['Input_Length_numeric', 'Output_Length_numeric']].values
            y = model_subset['TPS_numeric'].values
            
            # Log-linear regression (handle zeros by adding small epsilon)
            X_log = np.log(np.maximum(X, 1e-6))  # Avoid log(0)
            y_log = np.log(np.maximum(y, 1e-6))
            
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
    if model_name in regression_models:
        # Log transform the features
        log_features = np.array([[np.log(max(input_length, 1e-6)), np.log(max(output_length, 1e-6))]])
        
        # Predict log(TPS)
        log_tps_pred = regression_models[model_name]['model'].predict(log_features)[0]

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

# Simulation settings
n_runs = 10000
median_output_tokens = 300
min_tokens = 50
max_tokens = (2 * median_output_tokens) - min_tokens

fixed_input_length = 500
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

# Compute log-normal parameters where needed
mu_pu, sigma_pu = lognorm_params(*pu_range)
mu_pue, sigma_pue = lognorm_params(*PUE_range)

# Create separate distributions for each model using regression
all_model_energies = {}
all_model_tps = {}  # Add this to store TPS predictions

all_tps_models = {**tps_regression_models, **interpolation_models}

# ============================================================
# COMPARE OUTPUT-LENGTH DISTRIBUTIONS
# ============================================================

n_runs = 10000
median_output_tokens = 300

distributions = [
    'Exponential',
    'Log-normal',
    'Pareto',
    'Log-t',
    'Uniform'
]

# Store final energy distribution for each output-length model
all_distribution_energies = {}

# ============================================================
# Generate energy distributions
# ============================================================

for distribution in distributions:

    # Energy generated by each model
    distribution_model_energies = []

    for model_name in all_tps_models.keys():

        node_power = get_node_power(model_name)
        if distribution == 'Exponential':

            # Median = ln(2) / lambda
            lambda_param = np.log(2) / median_output_tokens

            model_token_lengths = np.random.exponential(
                scale=1 / lambda_param,
                size=n_runs
            )

        elif distribution == 'Log-normal':
            # Median = exp(mu)
            mu = np.log(median_output_tokens)

            # Spread parameter
            sigma = 1.0

            model_token_lengths = np.random.lognormal(
                mean=mu,
                sigma=sigma,
                size=n_runs
            )

        elif distribution == 'Pareto':

            # Median = xm * 2^(1/alpha)
            alpha = 2.0
            xm = median_output_tokens / (2 ** (1 / alpha))
            model_token_lengths = (
                xm * (1 + np.random.pareto(alpha, n_runs))
            )

        elif distribution == 'Log-t':
            # Log-t distribution:
            # log(L) = log(median) + t
            degrees_of_freedom = 3

            log_lengths = (
                np.log(median_output_tokens)
                + np.random.standard_t(
                    df=degrees_of_freedom,
                    size=n_runs
                )
            )

            model_token_lengths = np.exp(log_lengths)

        elif distribution == 'Uniform':

            # Symmetric distribution centred on the median
            min_tokens = 1
            max_tokens = (
                2 * median_output_tokens
                - min_tokens
            )
            model_token_lengths = np.random.uniform(
                min_tokens,
                max_tokens,
                n_runs
            )

        # Convert to integer token counts
        model_token_lengths = np.round(
            model_token_lengths
        ).astype(int)

        # Prevent zero-token outputs
        model_token_lengths = np.maximum(
            model_token_lengths,
            1
        )

        model_tokens_per_sec = np.array([

            predict_tps_for_lengths(
                model_name,
                fixed_input_length,
                token_length,
                tps_regression_models,
                interpolation_models,
                max_tps_values
            )

            for token_length in model_token_lengths

        ])

        model_tokens_per_sec = (
            model_tokens_per_sec.astype(float)
        )

        model_pu = np.random.lognormal(
            mu_pu,
            sigma_pu,
            n_runs
        )

        model_pue = np.random.lognormal(
            mu_pue,
            sigma_pue,
            n_runs
        )

        model_energies = (

            model_pue
            * node_power
            * model_pu
            * model_token_lengths
            / model_tokens_per_sec

        ) * 1000 / 3600

        distribution_model_energies.append(
            model_energies
        )

    distribution_model_energies = np.concatenate(
        distribution_model_energies
    )

    p5, p95 = np.percentile(
        distribution_model_energies,
        [5, 95]
    )

    filtered_energies = (
        distribution_model_energies[
            (distribution_model_energies >= p5)
            &
            (distribution_model_energies <= p95)
        ]
    )

    all_distribution_energies[distribution] = (
        filtered_energies
    )

    # Print statistics
    p25, p50, p75 = np.percentile(
        filtered_energies,
        [25, 50, 75]
    )

plot_data = []

for distribution in distributions:

    energies = all_distribution_energies[distribution]

    plot_data.append(
        pd.DataFrame({
            'Energy (Wh)': energies,
            'Distribution': distribution
        })
    )

plot_data = pd.concat(
    plot_data,
    ignore_index=True
)

colors = {
    'Exponential': '#2ecc71',
    'Log-normal': '#3498db',
    'Pareto': '#9b59b6',
    'Log-t': '#e67e22',
    'Uniform': '#95a5a6'
}

fig, ax = plt.subplots(
    figsize=(10, 8)
)

sns.violinplot(
    data=plot_data,
    x='Energy (Wh)',
    y='Distribution',
    order=distributions,
    hue='Distribution',
    palette=colors,
    orient='h',
    inner=None,
    cut=0,
    density_norm='width',
    bw_adjust=1,
    legend=False,
    ax=ax
)

for i, distribution in enumerate(distributions):

    energies = all_distribution_energies[
        distribution
    ]

    p25, p50, p75 = np.percentile(
        energies,
        [25, 50, 75]
    )

    # IQR
    ax.hlines(
        y=i,
        xmin=p25,
        xmax=p75,
        color='black',
        linewidth=2,
        zorder=5
    )

    # Median
    ax.vlines(
        x=p50,
        ymin=i - 0.10,
        ymax=i + 0.10,
        color='white',
        linewidth=4,
        zorder=6
    )

    ax.vlines(
        x=p50,
        ymin=i - 0.10,
        ymax=i + 0.10,
        color='black',
        linewidth=2,
        zorder=7
    )


ax.set_xlabel(
    'Energy per Query (Wh)',
    fontsize=16
)

ax.set_ylabel('')

ax.set_title(
    'Effect of Output-Length Distribution\n'
    'on Per-Query Energy Consumption',
    fontsize=16,
    pad=15
)

ax.grid(
    True,
    alpha=0.3
)

ax.yaxis.grid(
    True,
    linestyle='--',
    alpha=0.3
)

ax.tick_params(
    axis='both',
    labelsize=12
)

plt.tight_layout()