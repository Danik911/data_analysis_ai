import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Any, Union

def calculate_advanced_statistics(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate advanced statistics for numeric columns including skewness, kurtosis, and confidence intervals.
    
    Args:
        df: The DataFrame to analyze
        numeric_columns: Optional list of numeric columns to analyze. If None, all numeric columns are used.
        
    Returns:
        A dictionary with column names as keys and dictionaries of statistics as values
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    
    for column in numeric_columns:
        # Skip if column doesn't exist
        if column not in df.columns:
            continue
            
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        # Get non-null values for the column
        values = df[column].dropna()
        
        if len(values) < 2:
            continue
            
        stats_dict = {}
        
        # Basic statistics
        stats_dict['mean'] = values.mean()
        stats_dict['median'] = values.median()
        stats_dict['std'] = values.std()
        
        # Advanced statistics
        stats_dict['skewness'] = values.skew()
        stats_dict['kurtosis'] = values.kurtosis()
        
        # 95% confidence interval for the mean
        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=values.mean(),
            scale=stats.sem(values)
        )
        stats_dict['ci_95_low'] = ci_low
        stats_dict['ci_95_high'] = ci_high
        
        # Normality test (Shapiro-Wilk)
        shapiro_test = stats.shapiro(values)
        stats_dict['shapiro_stat'] = shapiro_test[0]
        stats_dict['shapiro_p_value'] = shapiro_test[1]
        stats_dict['is_normal'] = shapiro_test[1] > 0.05  # p > 0.05 suggests normality
        
        results[column] = stats_dict
    
    return results

def calculate_mode_statistics(df: pd.DataFrame, group_column: str, 
                              numeric_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate statistics for each group in group_column for the specified numeric columns.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column to group by (e.g., 'Mode')
        numeric_columns: Optional list of numeric columns to analyze. If None, all numeric columns are used.
        
    Returns:
        A nested dictionary with structure {column: {group: {statistic: value}}}
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    
    for column in numeric_columns:
        # Skip if column doesn't exist or is not numeric
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        results[column] = {}
        
        # For each group in the group column
        for group_value, group_df in df.groupby(group_column):
            values = group_df[column].dropna()
            
            if len(values) < 2:
                continue
                
            stats_dict = {}
            
            # Basic statistics
            stats_dict['mean'] = values.mean()
            stats_dict['median'] = values.median()
            stats_dict['std'] = values.std()
            stats_dict['count'] = len(values)
            
            # Advanced statistics
            stats_dict['skewness'] = values.skew()
            stats_dict['kurtosis'] = values.kurtosis()
            
            # 95% confidence interval for the mean
            ci_low, ci_high = stats.t.interval(
                confidence=0.95,
                df=len(values)-1,
                loc=values.mean(),
                scale=stats.sem(values)
            )
            stats_dict['ci_95_low'] = ci_low
            stats_dict['ci_95_high'] = ci_high
            
            results[column][group_value] = stats_dict
    
    return results

def perform_anova(df: pd.DataFrame, group_column: str, numeric_column: str) -> Dict[str, Any]:
    """
    Perform one-way ANOVA test to check if there's a significant difference between groups.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column containing the groups (e.g., 'Mode')
        numeric_column: Numeric column to analyze (e.g., 'Time')
        
    Returns:
        Dictionary with ANOVA results
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_column not in df.columns:
        raise ValueError(f"Numeric column '{numeric_column}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' is not numeric")
    
    # Create lists of data for each group
    groups = []
    group_names = []
    
    for group_value, group_df in df.groupby(group_column):
        values = group_df[numeric_column].dropna()
        if len(values) > 1:  # Need at least 2 values for variance
            groups.append(values)
            group_names.append(group_value)
    
    if len(groups) < 2:
        return {
            "error": "Not enough groups with sufficient data for ANOVA",
            "is_significant": False
        }
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "is_significant": p_value < 0.05,  # p < 0.05 indicates significant difference
        "groups": group_names
    }

def perform_tukey_hsd(df: pd.DataFrame, group_column: str, numeric_column: str) -> Dict[str, Any]:
    """
    Perform Tukey's HSD (Honest Significant Difference) test for pairwise comparisons 
    after ANOVA indicates significant differences.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column containing the groups (e.g., 'Mode')
        numeric_column: Numeric column to analyze (e.g., 'Time')
        
    Returns:
        Dictionary with Tukey HSD results
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_column not in df.columns:
        raise ValueError(f"Numeric column '{numeric_column}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' is not numeric")
    
    # First check if ANOVA shows significant differences
    anova_result = perform_anova(df, group_column, numeric_column)
    
    if not anova_result.get("is_significant", False):
        return {
            "message": "ANOVA did not show significant differences between groups",
            "anova_result": anova_result,
            "pairwise_results": {}
        }
    
    # Prepare data for Tukey's HSD
    # Create a DataFrame with the group as a separate column
    data = []
    for group_value, group_df in df.groupby(group_column):
        for value in group_df[numeric_column].dropna():
            data.append({'group': group_value, 'value': value})
    
    if not data:
        return {
            "error": "No valid data for Tukey HSD test",
            "anova_result": anova_result
        }
    
    data_df = pd.DataFrame(data)
    
    # Perform Tukey's HSD test
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey_result = pairwise_tukeyhsd(data_df['value'], data_df['group'], alpha=0.05)
        
        # Format results
        pairwise_results = []
        for i, (group1, group2, reject, _, _, _) in enumerate(zip(
            tukey_result.groupsunique[tukey_result.pairindices[:, 0]],
            tukey_result.groupsunique[tukey_result.pairindices[:, 1]],
            tukey_result.reject
        )):
            mean1 = df[df[group_column] == group1][numeric_column].mean()
            mean2 = df[df[group_column] == group2][numeric_column].mean()
            
            pairwise_results.append({
                "group1": group1,
                "group2": group2,
                "mean_difference": abs(mean1 - mean2),
                "is_significant": bool(reject),
                "p_value": tukey_result.pvalues[i]
            })
        
        return {
            "anova_result": anova_result,
            "pairwise_results": pairwise_results
        }
    except Exception as e:
        return {
            "error": f"Error performing Tukey HSD test: {str(e)}",
            "anova_result": anova_result
        }

def generate_advanced_plots(df: pd.DataFrame, output_dir: str = "plots/advanced") -> List[str]:
    """
    Generate advanced statistical plots including density plots, Q-Q plots, violin plots,
    correlation heatmaps, and pair plots.
    
    Args:
        df: The DataFrame to visualize
        output_dir: Directory to save plots in
        
    Returns:
        List of saved plot file paths
    """
    plot_paths = []
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    try:
        # 1. Density plots for Time and Distance
        numeric_cols = ['Time', 'Distance']
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(10, 6))
                # Plot the density for the whole dataset
                sns.kdeplot(data=df, x=col, fill=True, color='gray', alpha=0.5, label='Overall')
                
                # Plot density by Mode
                if 'Mode' in df.columns:
                    for mode in df['Mode'].unique():
                        subset = df[df['Mode'] == mode]
                        if len(subset) > 1:  # Need at least 2 points for density
                            sns.kdeplot(data=subset, x=col, fill=True, alpha=0.3, label=mode)
                
                plt.title(f'Density Plot of {col} by Mode')
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.legend()
                
                density_path = os.path.join(output_dir, f"{col.lower()}_density_plot.png")
                plt.savefig(density_path)
                plt.close()
                plot_paths.append(density_path)
                print(f"Saved plot: {density_path}")
        
        # 2. Q-Q plots for Time and Distance
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(8, 8))
                stats.probplot(df[col].dropna(), dist="norm", plot=plt)
                plt.title(f'Q-Q Plot of {col}')
                
                qq_path = os.path.join(output_dir, f"{col.lower()}_qq_plot.png")
                plt.savefig(qq_path)
                plt.close()
                plot_paths.append(qq_path)
                print(f"Saved plot: {qq_path}")
        
        # 3. Violin plots (Time by Mode)
        if 'Time' in df.columns and 'Mode' in df.columns:
            plt.figure(figsize=(12, 7))
            # Update violinplot to use hue instead of palette directly
            sns.violinplot(data=df, x='Mode', y='Time', inner='quart', hue='Mode', legend=False)
            plt.title('Violin Plot of Time by Mode')
            plt.xlabel('Mode of Transport')
            plt.ylabel('Time (minutes)')
            
            violin_path = os.path.join(output_dir, "time_mode_violin_plot.png")
            plt.savefig(violin_path)
            plt.close()
            plot_paths.append(violin_path)
            print(f"Saved plot: {violin_path}")
        
        # 4. Correlation heatmap
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                linewidths=0.5, 
                fmt='.2f',
                center=0
            )
            plt.title('Correlation Heatmap')
            
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            plot_paths.append(heatmap_path)
            print(f"Saved plot: {heatmap_path}")
        
        # 5. Pair plot with regression lines
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            plot_cols = ['Time', 'Distance']
            valid_cols = [col for col in plot_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(valid_cols) >= 2 and 'Mode' in df.columns:
                plt.figure(figsize=(10, 8))
                pair_plot = sns.pairplot(
                    data=df, 
                    vars=valid_cols, 
                    hue='Mode', 
                    kind='scatter',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6},
                    height=2.5
                )
                pair_plot.map_lower(sns.regplot, line_kws={'color': 'black'})
                plt.suptitle('Pair Plot with Regression Lines', y=1.02)
                
                pair_path = os.path.join(output_dir, "pair_plot.png")
                pair_plot.savefig(pair_path)
                plt.close('all')
                plot_paths.append(pair_path)
                print(f"Saved plot: {pair_path}")
        
        return plot_paths
    except Exception as e:
        import traceback
        print(f"Error generating advanced plots: {e}")
        print(traceback.format_exc())
        plt.close('all')  # Ensure all plots are closed
        return [f"Error: {str(e)}"]

def generate_statistical_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive statistical report including advanced statistics 
    and significance testing.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        Dictionary with statistical report data
    """
    report = {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "advanced_statistics": {},
        "group_statistics": {},
        "significance_tests": {}
    }
    
    # Calculate advanced statistics for numeric columns
    report["advanced_statistics"] = calculate_advanced_statistics(df)
    
    # Calculate statistics by Mode (if present)
    if 'Mode' in df.columns:
        report["group_statistics"]["Mode"] = calculate_mode_statistics(df, 'Mode')
        
        # Perform ANOVA and Tukey HSD for Time and Distance by Mode
        for col in ['Time', 'Distance']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                anova_result = perform_anova(df, 'Mode', col)
                
                if anova_result.get("is_significant", False):
                    tukey_result = perform_tukey_hsd(df, 'Mode', col)
                    report["significance_tests"][col] = tukey_result
                else:
                    report["significance_tests"][col] = {
                        "anova_result": anova_result,
                        "message": "No significant differences found between modes"
                    }
    
    return report