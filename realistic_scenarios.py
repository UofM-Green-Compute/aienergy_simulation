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

tps_regression_models, interpolation_models, max_tps_values = create_tps_regression_models(model_throughput)

# Central values
n_runs = 10_000
DEFAULT_PUE = 1.30
DEFAULT_PU = 0.70

np.random.seed(42)

def get_node_power(model_name):
    """
    Node power from the existing model-power assumptions.
    """

    return 12.8 if model_name == 'DeepSeek-R1' else 10.2

def calculate_energy(
    input_tokens,
    output_tokens,
    model_name,
    pue=DEFAULT_PUE,
    pu=DEFAULT_PU
):
    """
    Calculate energy consumption for individual simulated queries.

    Parameters
    ----------
    input_tokens : array-like
        Number of input tokens for each query.

    output_tokens : array-like
        Number of output tokens for each query.

    model_name : str
        Model being simulated.

    pue : float
        Power Usage Effectiveness.

    pu : float
        Power utilisation.

    Returns
    -------
    np.ndarray
        Energy consumption in Wh for each query.
    """

    node_power = get_node_power(model_name)
    tps = np.array([
        predict_tps_for_lengths(
            model_name,
            inp,
            out,
            tps_regression_models,
            interpolation_models,
            max_tps_values
        )
        for inp, out in zip(input_tokens, output_tokens)
    ])

    tps = tps.astype(float)

    energy_wh = (
        pue
        * node_power
        * pu
        * output_tokens
        / tps
    ) * 1000 / 3600

    return energy_wh

def generate_scenario_tokens(
    input_tokens,
    output_median,
    n_queries,
    n_runs=n_runs
):
    """
    Generate token lengths for a complete scenario.

    Each Monte Carlo run represents one occurrence of the
    complete scenario.

    Example:
        10 Word summaries means each Monte Carlo run contains
        10 separate queries.
    """

    input_tokens = np.full(
        (n_runs, n_queries),
        int(input_tokens),
        dtype=int
    )

    lambda_param = np.log(2) / output_median
    output_tokens = np.random.exponential(
        scale=1 / lambda_param,
        size=(n_runs, n_queries)
    )

    output_tokens = np.maximum(
        np.round(output_tokens).astype(int),
        1
    )
    return input_tokens, output_tokens

def run_scenario(
    scenario,
    model_name,
    pue=DEFAULT_PUE,
    pu=DEFAULT_PU,
    n_runs=n_runs
):
    """
    Run a Monte Carlo simulation for one scenario.

    Each Monte Carlo sample represents the total energy required
    for one occurrence of that scenario.

    """

    input_tokens, output_tokens = generate_scenario_tokens(
        input_tokens=scenario["input_tokens"],
        output_median=scenario["output_median"],
        n_queries=scenario["n_queries"],
        n_runs=n_runs
    )

    # Store energy for every individual query
    energy_per_query = np.zeros(
        (n_runs, scenario["n_queries"])
    )

    for q in range(scenario["n_queries"]):

        energy_per_query[:, q] = calculate_energy(
            input_tokens=input_tokens[:, q],
            output_tokens=output_tokens[:, q],
            model_name=model_name,
            pue=pue,
            pu=pu
        )
    total_energy = energy_per_query.sum(axis=1)

    return total_energy

scenario_colors = {
    "Negligible input": "#95A5A6",
    "1 meeting": "#3498DB",
    "10 Word summaries": "#9B59B6",
    "2 long PDF summaries": "#E67E22",
    "10 emails": "#2ECC71",
    "5 meetings": "#E74C3C",
    "15 long PDF summaries": "#F1C40F",
}

scenarios = {
    "Negligible input": {
        "n_queries": 1,
        "input_tokens": 1,
        "output_median": 180,
    },
    "1 meeting": {
        "n_queries": 1,
        "input_tokens": 10000,
        "output_median": 5000,
    },
    "10 Word summaries": {
        "n_queries": 10,
        "input_tokens": 3000,
        "output_median": 350,
    },
    "2 long PDF summaries": {
        "n_queries": 2,
        "input_tokens": 15000,
        "output_median": 900,
    },
    "10 emails": {
        "n_queries": 10,
        "input_tokens": 1000,
        "output_median": 180,
    },
    "5 meetings": {
        "n_queries": 5,
        "input_tokens": 10000,
        "output_median": 5000,
    },
    "15 long PDF summaries": {
        "n_queries": 15,
        "input_tokens": 15000,
        "output_median": 900,
    },
}

MODEL_NAME = "Llama 3.1 70B"

scenario_energies = {}

for scenario_name, scenario in scenarios.items():

    scenario_energies[scenario_name] = run_scenario(
        scenario=scenario,
        model_name=MODEL_NAME,
        pue=DEFAULT_PUE,
        pu=DEFAULT_PU,
        n_runs=n_runs
    )

statistics = []
for scenario_name, energies in scenario_energies.items():

    q25, median, q75 = np.percentile(
        energies,
        [25, 50, 75]
    )

    statistics.append({
        "Scenario": scenario_name,
        "Number of queries":
            scenarios[scenario_name]["n_queries"],
        "Input tokens per query":
            scenarios[scenario_name]["input_tokens"],
        "Median output tokens per query":
            scenarios[scenario_name]["output_median"],
        "Mean energy (Wh)":
            np.mean(energies),
        "Median energy (Wh)":
            median,
        "IQR lower (Q1) (Wh)":
            q25,
        "IQR upper (Q3) (Wh)":
            q75,
        "IQR (Wh)":
            q75 - q25,
        "Maximum simulated energy (Wh)":
            np.max(energies),
    })
statistics_df = pd.DataFrame(statistics)

statistics_df.to_csv(
    "realistic_scenario_energy_statistics.csv",
    index=False
)

central_scenarios = [
    "Negligible input",
    "1 meeting",
    "10 Word summaries",
    "2 long PDF summaries",
    "10 emails"
]

additional_scenarios = [
    "5 meetings",
    "15 long PDF summaries"
]

def plot_scenarios(
    scenario_names,
    filename,
    title
):

    plot_data = pd.concat([
        pd.DataFrame({
            "Energy (Wh)": scenario_energies[name],
            "Scenario": name
        })

        for name in scenario_names

    ], ignore_index=True)
    with plt.rc_context({
        "font.size":18,
        "axes.titlesize":20,
        "axes.labelsize":18,
        "xtick.labelsize":10,
        "ytick.labelsize":10
    }):
        plt.figure(figsize=(12, 8))
    
        ax = sns.violinplot(
            data=plot_data,
            x="Energy (Wh)",
            y="Scenario",
            orient="h",
            inner=None,
            cut=0,
            density_norm="width",
            palette={
                name: scenario_colors[name]
                for name in scenario_names
            },
            hue="Scenario",
            legend=False
        )

    
        for i, name in enumerate(scenario_names):
            energies = scenario_energies[name]
            q25, median, q75 = np.percentile(
                energies,
                [25, 50, 75]
            )
    
            # IQR
            ax.hlines(
                i,
                q25,
                q75,
                color="black",
                linewidth=3
            )
    
            # Median
            ax.vlines(
                median,
                i - 0.12,
                i + 0.12,
                color="black",
                linewidth=3
            )
    
    
        plt.title(title)
    
        plt.xlabel(
            "Energy per scenario (Wh)"
        )
    
        plt.ylabel("Scenario")
    
        plt.grid(
            True,
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )
    
        plt.show()

plot_scenarios(
    central_scenarios,
    "central_university_scenarios.png",
    "Central Energy Estimates for University LLM Use"
)

plot_scenarios(
    additional_scenarios,
    "additional_university_scenarios.png",
    "Additional High-Use University Scenarios"
)