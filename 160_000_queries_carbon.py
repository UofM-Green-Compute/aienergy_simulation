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
    
    # Get numeric TPS values
    model_data['TPS_numeric'] = pd.to_numeric(model_data['Tokens per Second (TPS)'], errors='coerce')
    
    # Convert Input Length and Output Length to numeric (handle non-numeric values)
    model_data['Input_Length_numeric'] = pd.to_numeric(model_data['Input Length'], errors='coerce')
    model_data['Output_Length_numeric'] = pd.to_numeric(model_data['Output Length'], errors='coerce')

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
            
            # Use the highest TPS value from available data
            max_tps = model_subset['TPS_numeric'].max()  # Changed from mean to max
            
            interpolation_models[model_name] = {
                'type': 'max_tps',
                'max_tps': max_tps,
                'n_points': len(model_subset),
                'tps_range': (model_subset['TPS_numeric'].min(), model_subset['TPS_numeric'].max())
            }

        else:  # Use regression for 3+ points

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
n_runs = 10_000

# Fixed assumptions
DEFAULT_PU = 0.70
DEFAULT_PUE = 1.30

# Model being analysed
MODEL_NAME = "DeepSeek-R1"


def get_node_power(model_name):
    """
    Node power from the existing model-power assumptions.
    """

    return 12.8 if model_name == 'DeepSeek-R1' else 10.2

input_tokens = 300
output_tokens = 500

# PUE range
PUE_RANGE = (1.20, 1.56)

# Power utilisation range
PU_RANGE = (0.40, 0.90)

mu_pue, sigma_pue = lognorm_params(*PUE_RANGE)
mu_pu, sigma_pu = lognorm_params(*PU_RANGE)

def calculate_fixed_token_energy(
    input_tokens,
    output_tokens,
    model_name,
    pue,
    pu
):

    # Predict TPS once because token lengths are fixed
    tps = predict_tps_for_lengths(
        model_name,
        input_tokens,
        output_tokens,
        tps_regression_models,
        interpolation_models,
        max_tps_values
    )

    if tps is None:
        raise ValueError(
            f"No TPS prediction available for {model_name}"
        )

    node_power = get_node_power(model_name)

    energy_wh = (
        pue
        * node_power
        * pu
        * output_tokens
        / tps
    ) * 1000 / 3600

    return energy_wh

# ============================================================
# MAXIMUM TOKEN SCENARIOS
# ============================================================

maximum_token_scenarios = {

    "Maximum output tokens": {
        "input_tokens": 1_000,
        "output_tokens": 160_000,
    },

    "Maximum input tokens": {
        "input_tokens": 128_000,
        "output_tokens": 1_000,
    },

    "Maximum combined tokens": {
        "input_tokens": 80_000,
        "output_tokens": 80_000,
    },
}


# ============================================================
# PUE AND POWER UTILISATION UNCERTAINTY
# ============================================================

PUE_RANGE = (1.20, 1.56)
PU_RANGE = (0.40, 0.90)

mu_pue, sigma_pue = lognorm_params(*PUE_RANGE)
mu_pu, sigma_pu = lognorm_params(*PU_RANGE)


maximum_energy_results = {}


for scenario_name, scenario in maximum_token_scenarios.items():

    # Generate uncertain PUE
    pue_samples = np.random.lognormal(
        mean=mu_pue,
        sigma=sigma_pue,
        size=n_runs
    )

    # Generate uncertain power utilisation
    pu_samples = np.random.lognormal(
        mean=mu_pu,
        sigma=sigma_pu,
        size=n_runs
    )

    # Calculate TPS ONCE because input/output tokens are fixed
    tps = predict_tps_for_lengths(
        MODEL_NAME,
        scenario["input_tokens"],
        scenario["output_tokens"],
        tps_regression_models,
        interpolation_models,
        max_tps_values
    )

    if tps is None:
        raise ValueError(
            f"No TPS prediction available for {MODEL_NAME}"
        )

    node_power = get_node_power(MODEL_NAME)

    # Vectorised energy calculation
    energies = (
        pue_samples
        * node_power
        * pu_samples
        * scenario["output_tokens"]
        / tps
    ) * 1000 / 3600

    maximum_energy_results[scenario_name] = energies


# ============================================================
# MAXIMUM-TOKEN STATISTICS
# ============================================================

maximum_statistics = []


for scenario_name, energies in maximum_energy_results.items():

    q25, median, q75 = np.percentile(
        energies,
        [25, 50, 75]
    )

    maximum_statistics.append({

        "Scenario": scenario_name,

        "Input tokens":
            maximum_token_scenarios[
                scenario_name
            ]["input_tokens"],

        "Output tokens":
            maximum_token_scenarios[
                scenario_name
            ]["output_tokens"],

        "Mean energy (Wh)":
            np.mean(energies),

        "Median energy (Wh)":
            median,

        "Q1 (Wh)":
            q25,

        "Q3 (Wh)":
            q75,

        "IQR (Wh)":
            q75 - q25,

        "Maximum simulated energy (Wh)":
            np.max(energies)
    })


maximum_statistics_df = pd.DataFrame(
    maximum_statistics
)

maximum_statistics_df.to_csv(
    "maximum_token_energy_statistics.csv",
    index=False
)

# Input and output token values to test
INPUT_VALUES = np.array([
    100,
    300,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000
])

OUTPUT_VALUES = np.array([
    100,
    300,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    160_000
])


results = []

for input_length in INPUT_VALUES:
    for output_length in OUTPUT_VALUES:

        # Predict throughput for this input/output combination
        tps = predict_tps_for_lengths(
            MODEL_NAME,
            input_length,
            output_length,
            tps_regression_models,
            interpolation_models,
            max_tps_values
        )

        if tps is None:
            continue

        # Central energy estimate using default PUE and PU
        energy_wh = (
            DEFAULT_PUE
            * get_node_power(MODEL_NAME)
            * DEFAULT_PU
            * output_length
            / tps
        ) * 1000 / 3600

        results.append({

            "Input": input_length,

            "Output": output_length,

            "TPS": tps,

            "Median Energy": energy_wh
        })


results = pd.DataFrame(results)


# Save sensitivity results
results.to_csv(
    "token_sensitivity_results.csv",
    index=False
)

def plot_energy_heatmap(
    results,
    value_column,
    colorbar_label,
    title,
    filename
):

    pivot = results.pivot(
        index="Output",
        columns="Input",
        values=value_column
    )

    X = pivot.columns.values
    Y = pivot.index.values
    Z = pivot.values

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="viridis"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Input tokens",
        fontsize=18
    )

    ax.set_ylabel(
        "Output tokens",
        fontsize=18
    )

    ax.set_title(
        title,
        fontsize=20,
        pad=15
    )

    ax.tick_params(
        axis="both",
        labelsize=14
    )

    cbar = fig.colorbar(
        contour,
        ax=ax
    )

    cbar.set_label(
        colorbar_label,
        fontsize=16
    )

    cbar.ax.tick_params(
        labelsize=13
    )

    ax.grid(
        True,
        which="both",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        f"manuscript_figures/updated_figures/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        f"manuscript_figures/updated_figures/{filename}.svg",
        bbox_inches="tight"
    )

    plt.show()

plot_energy_heatmap(
    results,
    "Median Energy",
    "Energy (Wh)",
    "Sensitivity of Energy Consumption to Input and Output Tokens",
    "figure_median_energy_sensitivity"
)

# Representative query
input_tokens = 300
output_tokens = 500

NUMBER_OF_QUERIES = 160_000

# UK electricity carbon conversion factor
# kg CO2e per kWh
DEFRA_conv = 0.177

energy_wh_per_query = calculate_fixed_token_energy(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    model_name=MODEL_NAME,
    pue=DEFAULT_PUE,
    pu=DEFAULT_PU
)

energy_kwh_per_query = (
    energy_wh_per_query / 1000
)

# Total electricity for 160,000 queries
total_energy_kwh = (
    energy_kwh_per_query
    * NUMBER_OF_QUERIES
)

# Carbon emissions
co2_per_query = (
    energy_kwh_per_query
    * DEFRA_conv
)

total_co2 = (
    total_energy_kwh
    * DEFRA_conv
)
print(
    f"Input tokens/query: {input_tokens:,}"
)
print(
    f"Output tokens/query: {output_tokens:,}"
)
print(
    f"Number of queries: {NUMBER_OF_QUERIES:,}"
)
print(
    f"Energy/query: {energy_wh_per_query:.6f} Wh"
)
print(
    f"Total energy: {total_energy_kwh:.3f} kWh"
)
print(
    f"CO2/query: {co2_per_query:.6f} kg CO2e"
)
print(
    f"Total CO2: {total_co2:.3f} kg CO2e"
)

carbon_results = pd.DataFrame(results * DEFRA_conv)

# Carbon heatmap
results.to_csv(
    "token_sensitivity_results.csv",
    index=False
)

def plot_carbon_heatmap(
    results,
    value_column,
    colorbar_label,
    title,
    filename
):

    pivot = results.pivot(
        index="Output",
        columns="Input",
        values=value_column
    )

    X = pivot.columns.values
    Y = pivot.index.values
    Z = pivot.values

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="viridis"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Input tokens",
        fontsize=18
    )

    ax.set_ylabel(
        "Output tokens",
        fontsize=18
    )

    ax.set_title(
        title,
        fontsize=20,
        pad=15
    )

    ax.tick_params(
        axis="both",
        labelsize=14
    )

    cbar = fig.colorbar(
        contour,
        ax=ax
    )

    cbar.set_label(
        colorbar_label,
        fontsize=16
    )

    cbar.ax.tick_params(
        labelsize=13
    )

    ax.grid(
        True,
        which="both",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        f"manuscript_figures/updated_figures/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        f"manuscript_figures/updated_figures/{filename}.svg",
        bbox_inches="tight"
    )

    plt.show()

plot_energy_heatmap(
    carbon_results,
    "Median Energy",
    "Carbon consumption (kg/kWh)",
    "Sensitivity of Carbon Consumption to Input and Output Tokens",
    "figure_median_energy_sensitivity"
)