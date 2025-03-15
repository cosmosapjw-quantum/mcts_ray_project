# hyperparameter_analysis.py
"""
Analysis utilities for hyperparameter optimization results.
Provides tools to visualize and analyze results from Ray Tune experiments.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ray.tune import Analysis

def load_results(experiment_dir):
    """
    Load results from a Ray Tune experiment directory
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Analysis: Ray Tune Analysis object
    """
    return Analysis(experiment_dir)

def plot_learning_curves(analysis, metric="loss", mode="min", top_n=5, save_path=None):
    """
    Plot learning curves for the top N configurations
    
    Args:
        analysis: Ray Tune Analysis object
        metric: Metric to plot
        mode: "min" or "max"
        top_n: Number of top configurations to plot
        save_path: Optional path to save the plot
    """
    # Get dataframes for all trials
    dfs = analysis.trial_dataframes
    
    # If no dataframes, return
    if not dfs:
        print("No trial data available")
        return
    
    # Get top N configurations
    if mode == "min":
        top_configs = analysis.get_best_configs(metric, mode="min", scope="last", n=top_n)
    else:
        top_configs = analysis.get_best_configs(metric, mode="max", scope="last", n=top_n)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot learning curves for top configurations
    for i, (trial_id, df) in enumerate(dfs.items()):
        config = analysis.get_trial_config(trial_id)
        # Check if this config is in the top N
        in_top = False
        for top_config in top_configs:
            if all(config.get(k) == v for k, v in top_config.items() if k in config):
                in_top = True
                break
        
        if in_top:
            # Plot with thicker line and label
            if 'games_completed' in df.columns and metric in df.columns:
                plt.plot(df['games_completed'], df[metric], linewidth=2, 
                        label=f"Trial {trial_id[-6:]}")
        elif i < 20:  # Plot at most 20 background curves to avoid clutter
            # Plot with transparent line and no label
            if 'games_completed' in df.columns and metric in df.columns:
                plt.plot(df['games_completed'], df[metric], linewidth=1, alpha=0.3, color='gray')
    
    plt.xlabel('Games Completed')
    plt.ylabel(metric.capitalize())
    plt.title(f'Learning Curves for Top {top_n} Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_parameter_importance(analysis, metric="loss", mode="min", save_path=None):
    """
    Plot parameter importance based on correlation with performance
    
    Args:
        analysis: Ray Tune Analysis object
        metric: Metric to use for performance
        mode: "min" or "max"
        save_path: Optional path to save the plot
    """
    # Get dataframes
    dfs = analysis.trial_dataframes
    
    # If no dataframes, return
    if not dfs:
        print("No trial data available")
        return
    
    # Prepare data for correlation analysis
    data = []
    
    for trial_id, df in dfs.items():
        if df.empty or metric not in df.columns:
            continue
            
        config = analysis.get_trial_config(trial_id)
        if not config:
            continue
            
        # Use last reported value of the metric
        last_value = df[metric].iloc[-1] if not df[metric].iloc[-1].isnull() else float('inf')
        
        # Flatten config and add metric value
        row = {k: v for k, v in config.items() if not isinstance(v, dict)}
        row[metric] = last_value
        data.append(row)
    
    # Convert to dataframe
    param_df = pd.DataFrame(data)
    
    # Calculate correlations
    if len(param_df) > 1:
        correlations = param_df.corr()[metric].drop(metric).sort_values(
            ascending=(mode == "min"))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        corr_values = correlations.values
        abs_corr = np.abs(corr_values)
        
        # Create bar plot
        colors = ['red' if x < 0 else 'green' for x in corr_values]
        plt.barh(correlations.index, abs_corr, color=colors)
        
        plt.xlabel('Absolute Correlation')
        plt.ylabel('Parameter')
        plt.title(f'Parameter Importance for {metric}')
        plt.grid(True, alpha=0.3)
        
        # Add correlation values as text
        for i, v in enumerate(abs_corr):
            sign = '-' if corr_values[i] < 0 else '+'
            plt.text(v + 0.01, i, f"{sign}{v:.2f}", va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("Not enough data points for correlation analysis")

def plot_pairwise_relationships(analysis, metric="loss", params=None, top_n=50, save_path=None):
    """
    Plot pairwise relationships between parameters and performance
    
    Args:
        analysis: Ray Tune Analysis object
        metric: Metric to use for performance
        params: List of parameters to include (default: top 5 by correlation)
        top_n: Number of trials to include
        save_path: Optional path to save the plot
    """
    # Get dataframes
    dfs = analysis.trial_dataframes
    
    # If no dataframes, return
    if not dfs:
        print("No trial data available")
        return
    
    # Prepare data for analysis
    data = []
    
    for trial_id, df in dfs.items():
        if df.empty or metric not in df.columns:
            continue
            
        config = analysis.get_trial_config(trial_id)
        if not config:
            continue
            
        # Use last reported value of the metric
        last_value = df[metric].iloc[-1] if not df[metric].iloc[-1].isnull() else float('inf')
        
        # Flatten config and add metric value
        row = {k: v for k, v in config.items() if not isinstance(v, dict)}
        row[metric] = last_value
        data.append(row)
    
    # Convert to dataframe
    param_df = pd.DataFrame(data)
    
    # Select parameters to include
    if params is None:
        if len(param_df) > 1:
            # Use top 5 parameters by correlation
            correlations = param_df.corr()[metric].drop(metric)
            params = correlations.abs().sort_values(ascending=False).head(5).index.tolist()
        else:
            # Use all parameters
            params = [col for col in param_df.columns if col != metric]
    
    # Include the metric
    params = params + [metric]
    
    # Limit to top_n trials to avoid overcrowding
    if len(param_df) > top_n:
        param_df = param_df.sort_values(by=metric).head(top_n)
    
    # Create figure
    if len(params) > 1:
        plt.figure(figsize=(15, 12))
        sns.pairplot(param_df[params], diag_kind='kde', markers='o', 
                    plot_kws=dict(alpha=0.6, edgecolor='k', linewidth=0.5))
        plt.suptitle(f'Pairwise Relationships between Parameters and {metric}', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("Not enough parameters for pairwise analysis")

def print_best_configurations(analysis, metric="loss", mode="min", top_n=5):
    """
    Print the top N configurations based on a metric
    
    Args:
        analysis: Ray Tune Analysis object
        metric: Metric to use for ranking
        mode: "min" or "max"
        top_n: Number of top configurations to print
    """
    if mode == "min":
        best_configs = analysis.get_best_configs(metric, mode="min", scope="last", n=top_n)
    else:
        best_configs = analysis.get_best_configs(metric, mode="max", scope="last", n=top_n)
    
    print(f"\nTop {top_n} configurations based on {metric} ({mode}):")
    for i, config in enumerate(best_configs):
        print(f"\nRank {i+1}:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")

def plot_parallel_coordinates(analysis, metric="loss", mode="min", top_n=10, save_path=None):
    """
    Create a parallel coordinates plot of the top N configurations
    
    Args:
        analysis: Ray Tune Analysis object
        metric: Metric to use for ranking
        mode: "min" or "max"
        top_n: Number of top configurations to include
        save_path: Optional path to save the plot
    """
    # Collect the top configurations and their performance
    configs = []
    metric_values = []
    
    # Get dataframes
    dfs = analysis.trial_dataframes
    
    for trial_id, df in dfs.items():
        if df.empty or metric not in df.columns:
            continue
            
        config = analysis.get_trial_config(trial_id)
        if not config:
            continue
            
        # Use last reported value of the metric
        if not df[metric].iloc[-1].isnull():
            last_value = df[metric].iloc[-1]
            metric_values.append(last_value)
            configs.append(config)
    
    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    if mode == "max":
        sorted_indices = sorted_indices[::-1]
    
    # Take top N
    top_indices = sorted_indices[:top_n]
    top_configs = [configs[i] for i in top_indices]
    top_metrics = [metric_values[i] for i in top_indices]
    
    # Create dataframe for parallel coordinates
    param_df = pd.DataFrame(top_configs)
    param_df[metric] = top_metrics
    
    # Select numeric parameters
    numeric_params = []
    for col in param_df.columns:
        if col == metric:
            numeric_params.append(col)
        elif pd.api.types.is_numeric_dtype(param_df[col]):
            numeric_params.append(col)
    
    if len(numeric_params) < 2:
        print("Not enough numeric parameters for parallel coordinates plot")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Normalize all parameters to [0, 1]
    param_df_norm = param_df[numeric_params].copy()
    for col in numeric_params:
        if param_df_norm[col].nunique() > 1:
            param_df_norm[col] = (param_df_norm[col] - param_df_norm[col].min()) / (param_df_norm[col].max() - param_df_norm[col].min())
    
    # Plot each configuration
    cmap = plt.cm.viridis
    for i in range(len(param_df_norm)):
        # Color based on metric value
        color = cmap(param_df_norm[metric].iloc[i])
        
        # Values for this configuration
        values = param_df_norm.iloc[i]
        
        # Plot lines connecting parameter values
        x = range(len(numeric_params))
        y = [values[p] for p in numeric_params]
        plt.plot(x, y, 'o-', color=color, alpha=0.7)
    
    # Set up x-axis with parameter names
    plt.xticks(range(len(numeric_params)), numeric_params, rotation=45)
    plt.ylabel('Normalized Parameter Value')
    plt.title(f'Parallel Coordinates Plot of Top {top_n} Configurations')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar for metric value
    norm = plt.Normalize(param_df_norm[metric].min(), param_df_norm[metric].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(metric)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def apply_config_to_file(config, output_file="tuned_config.py"):
    """
    Apply the best configuration to a new config file
    
    Args:
        config: Best configuration dictionary
        output_file: Path to the output Python file
    """
    with open(output_file, "w") as f:
        f.write("# Auto-generated configuration from hyperparameter optimization\n")
        f.write("# Generated by hyperparameter_analysis.py\n\n")
        
        for key, value in sorted(config.items()):
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")
        
        # Handle special case for TEMPERATURE_SCHEDULE
        if 'TEMP_INIT' in config and 'TEMP_FINAL' in config and 'TEMP_DECAY_MOVE' in config:
            f.write("\n# Temperature schedule\n")
            f.write("TEMPERATURE_SCHEDULE = {\n")
            f.write(f"    0: {config['TEMP_INIT']},\n")
            f.write(f"    {config['TEMP_DECAY_MOVE']}: {config['TEMP_FINAL']},\n")
            f.write("}\n")
    
    print(f"Applied best configuration to {output_file}")

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Analysis tools for hyperparameter optimization results')
    
    # Basic parameters
    parser.add_argument('experiment_dir', type=str, help='Path to the experiment directory')
    parser.add_argument('--metric', type=str, default='loss', help='Metric to analyze')
    parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'], 
                       help='Whether to minimize or maximize the metric')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Directory to save analysis results')
    
    # Analysis options
    parser.add_argument('--learning-curves', action='store_true', 
                       help='Plot learning curves')
    parser.add_argument('--parameter-importance', action='store_true', 
                       help='Plot parameter importance')
    parser.add_argument('--pairwise', action='store_true', 
                       help='Plot pairwise relationships')
    parser.add_argument('--parallel-coords', action='store_true', 
                       help='Plot parallel coordinates')
    parser.add_argument('--print-best', action='store_true', 
                       help='Print best configurations')
    parser.add_argument('--apply-best', action='store_true', 
                       help='Apply best configuration to a new file')
    parser.add_argument('--all', action='store_true', 
                       help='Run all analyses')
    parser.add_argument('--top-n', type=int, default=5, 
                       help='Number of top configurations to consider')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.experiment_dir}...")
    analysis = load_results(args.experiment_dir)
    
    # Determine which analyses to run
    run_learning_curves = args.learning_curves or args.all
    run_parameter_importance = args.parameter_importance or args.all
    run_pairwise = args.pairwise or args.all
    run_parallel_coords = args.parallel_coords or args.all
    run_print_best = args.print_best or args.all
    run_apply_best = args.apply_best or args.all
    
    # Perform requested analyses
    if run_learning_curves:
        print("Plotting learning curves...")
        save_path = os.path.join(args.output_dir, 'learning_curves.png') if args.output_dir else None
        plot_learning_curves(analysis, args.metric, args.mode, args.top_n, save_path)
    
    if run_parameter_importance:
        print("Plotting parameter importance...")
        save_path = os.path.join(args.output_dir, 'parameter_importance.png') if args.output_dir else None
        plot_parameter_importance(analysis, args.metric, args.mode, save_path)
    
    if run_pairwise:
        print("Plotting pairwise relationships...")
        save_path = os.path.join(args.output_dir, 'pairwise_relationships.png') if args.output_dir else None
        plot_pairwise_relationships(analysis, args.metric, None, args.top_n * 10, save_path)
    
    if run_parallel_coords:
        print("Plotting parallel coordinates...")
        save_path = os.path.join(args.output_dir, 'parallel_coordinates.png') if args.output_dir else None
        plot_parallel_coordinates(analysis, args.metric, args.mode, args.top_n, save_path)
    
    if run_print_best:
        print_best_configurations(analysis, args.metric, args.mode, args.top_n)
    
    if run_apply_best:
        # Get best configuration
        best_config = analysis.get_best_config(args.metric, mode=args.mode)
        output_file = "tuned_config.py"
        if args.output_dir:
            output_file = os.path.join(args.output_dir, output_file)
        apply_config_to_file(best_config, output_file)

if __name__ == "__main__":
    main()