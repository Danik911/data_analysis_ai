import os
import json
import pandas as pd
import traceback
from typing import Dict, Any, List, Optional
from pandas_helper import PandasHelper
from llama_index.experimental.query_engine import PandasQueryEngine
from statistical_analysis import generate_statistical_report

async def perform_advanced_analysis(
    df: pd.DataFrame, 
    llm, 
    original_path: str,
    modification_summary: str = None
) -> Dict[str, Any]:
    """
    Perform advanced statistical analysis on the cleaned data.
    
    Args:
        df: The DataFrame to analyze
        llm: The language model to use for the agent
        original_path: Path to the original data file
        modification_summary: Summary of data modifications performed
        
    Returns:
        Dictionary containing statistical analysis results
    """
    print("[ADVANCED ANALYSIS] Starting advanced statistical analysis")
    print(f"[ADVANCED ANALYSIS] DataFrame shape: {df.shape}")
    print(f"[ADVANCED ANALYSIS] DataFrame columns: {df.columns.tolist()}")
    print(f"[ADVANCED ANALYSIS] DataFrame data types: {df.dtypes.to_dict()}")
    
    # Create a helper instance with the cleaned DataFrame
    query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False)
    pandas_helper = PandasHelper(df, query_engine)
    
    # Create directory for reports if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Define the path for the statistical analysis report
    statistical_report_path = "reports/statistical_analysis_report.json"
    
    print("[ADVANCED ANALYSIS] Calling pandas_helper.perform_advanced_analysis...")
    
    # Perform advanced statistical analysis
    statistical_report = await pandas_helper.perform_advanced_analysis(
        save_report=True,
        report_path=statistical_report_path
    )
    
    print(f"[ADVANCED ANALYSIS] Statistical report generated with keys: {list(statistical_report.keys())}")
    
    # Generate a summary of the statistical analysis
    summary = "Advanced Statistical Analysis Complete\n\n"
    
    try:
        # Add advanced statistics summary
        if "advanced_statistics" in statistical_report:
            print(f"[ADVANCED ANALYSIS] Processing advanced statistics for {len(statistical_report['advanced_statistics'])} columns")
            summary += "## Advanced Statistics\n\n"
            for column, stats in statistical_report["advanced_statistics"].items():
                summary += f"### {column}\n"
                summary += f"- Mean: {stats.get('mean', 'N/A'):.2f}\n"
                summary += f"- Median: {stats.get('median', 'N/A'):.2f}\n"
                summary += f"- Standard Deviation: {stats.get('std', 'N/A'):.2f}\n"
                summary += f"- Skewness: {stats.get('skewness', 'N/A'):.2f}\n"
                summary += f"- Kurtosis: {stats.get('kurtosis', 'N/A'):.2f}\n"
                
                # Add confidence intervals if available
                if 'ci_95_low' in stats and 'ci_95_high' in stats:
                    summary += f"- 95% Confidence Interval: ({stats['ci_95_low']:.2f}, {stats['ci_95_high']:.2f})\n"
                
                # Add normality test results if available
                if 'is_normal' in stats:
                    summary += f"- Normality (Shapiro-Wilk): {'Normal' if stats['is_normal'] else 'Not normal'}\n"
                
                summary += "\n"
        
        # Add significance test results
        if "significance_tests" in statistical_report:
            print(f"[ADVANCED ANALYSIS] Processing significance tests for {len(statistical_report['significance_tests'])} columns")
            summary += "## Significance Tests\n\n"
            for column, test_results in statistical_report["significance_tests"].items():
                summary += f"### {column}\n"
                
                # Add ANOVA results
                if "anova_result" in test_results:
                    anova = test_results["anova_result"]
                    if "is_significant" in anova:
                        is_significant = anova["is_significant"]
                        summary += f"- ANOVA: {'Significant differences found' if is_significant else 'No significant differences'}\n"
                        if "f_statistic" in anova and "p_value" in anova:
                            summary += f"  - F-statistic: {anova['f_statistic']:.2f}, p-value: {anova['p_value']:.4f}\n"
                
                # Add Tukey HSD results if available
                if "pairwise_results" in test_results and test_results["pairwise_results"]:
                    print(f"[ADVANCED ANALYSIS] Processing {len(test_results['pairwise_results'])} pairwise comparisons for {column}")
                    summary += "- Tukey HSD Pairwise Comparisons:\n"
                    for pair in test_results["pairwise_results"]:
                        if "group1" in pair and "group2" in pair and "is_significant" in pair:
                            sig_text = "Significant" if pair["is_significant"] else "Not significant"
                            summary += f"  - {pair['group1']} vs {pair['group2']}: {sig_text}\n"
                            if "mean_difference" in pair:
                                summary += f"    Mean difference: {pair['mean_difference']:.2f}\n"
                
                summary += "\n"
        
        # Add group statistics summary if available
        if "group_statistics" in statistical_report and "Mode" in statistical_report["group_statistics"]:
            print("[ADVANCED ANALYSIS] Processing group statistics by Mode")
            mode_stats = statistical_report["group_statistics"]["Mode"]
            summary += "## Statistics by Mode\n\n"
            
            for column, modes in mode_stats.items():
                summary += f"### {column} by Mode\n"
                for mode, stats in modes.items():
                    summary += f"- {mode}:\n"
                    summary += f"  - Mean: {stats.get('mean', 'N/A'):.2f}\n"
                    summary += f"  - Count: {stats.get('count', 'N/A')}\n"
                    if 'ci_95_low' in stats and 'ci_95_high' in stats:
                        summary += f"  - 95% CI: ({stats['ci_95_low']:.2f}, {stats['ci_95_high']:.2f})\n"
                
                summary += "\n"
    except Exception as e:
        print(f"[ADVANCED ANALYSIS ERROR] Error generating statistical summary: {e}")
        print(traceback.format_exc())
        summary += f"Error generating statistical summary: {e}\n"
        summary += "Full statistical report saved to JSON file.\n"
    
    print(f"[ADVANCED ANALYSIS] Statistical summary generated with length {len(summary)}")
    
    # Generate advanced plots using the PandasHelper
    print("[ADVANCED ANALYSIS] Generating advanced visualizations...")
    os.makedirs("plots/advanced", exist_ok=True)
    advanced_plot_paths = await pandas_helper.generate_advanced_plots(output_dir="plots/advanced")
    
    if advanced_plot_paths:
        print(f"[ADVANCED ANALYSIS] Generated {len(advanced_plot_paths)} advanced plots")
        plot_info = "Advanced visualizations generated:\n"
        for path in advanced_plot_paths:
            if isinstance(path, str) and not path.startswith("Error"):
                plot_info += f"- {path}\n"
                print(f"[ADVANCED ANALYSIS] Generated plot: {path}")
        
        print(f"[ADVANCED ANALYSIS] Advanced visualization summary: {len(plot_info)} characters")
    else:
        print("[ADVANCED ANALYSIS WARNING] No advanced plots were generated or an error occurred")
        plot_info = "No advanced plots were generated."
    
    # Prepare path for modified file
    path_parts = os.path.splitext(original_path)
    modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"
    print(f"[ADVANCED ANALYSIS] Modified file path: {modified_file_path}")
    
    print("[ADVANCED ANALYSIS] Advanced statistical analysis completed")
    
    return {
        "statistical_report": statistical_report,
        "summary": summary,
        "plot_info": plot_info,
        "statistical_report_path": statistical_report_path,
        "modified_data_path": modified_file_path
    }

async def summarize_statistical_findings(statistical_report: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of key statistical findings.
    
    Args:
        statistical_report: Dictionary containing statistical analysis results
        
    Returns:
        String containing a formatted summary of key statistical findings
    """
    print("[ADVANCED ANALYSIS] Creating statistical findings summary")
    summary = "Key Statistical Findings:\n\n"
    
    # Extract and summarize basic statistics
    if "advanced_statistics" in statistical_report:
        for column, stats in statistical_report["advanced_statistics"].items():
            summary += f"- {column}: Mean = {stats.get('mean', 'N/A'):.2f}, "
            summary += f"Median = {stats.get('median', 'N/A'):.2f}, "
            summary += f"Std Dev = {stats.get('std', 'N/A'):.2f}\n"
    
    # Extract and summarize significant findings
    if "significance_tests" in statistical_report:
        significant_findings = []
        for column, test_results in statistical_report["significance_tests"].items():
            if "anova_result" in test_results and test_results["anova_result"].get("is_significant", False):
                significant_findings.append(f"Significant differences found in {column} across transport modes")
                
                if "pairwise_results" in test_results:
                    sig_pairs = []
                    for pair in test_results["pairwise_results"]:
                        if pair.get("is_significant", False):
                            sig_pairs.append(f"{pair.get('group1', '')} vs {pair.get('group2', '')}")
                    
                    if sig_pairs:
                        summary += f"- Significant differences in {column} between: {', '.join(sig_pairs)}\n"
    
    # Add any normality findings
    non_normal_vars = []
    if "advanced_statistics" in statistical_report:
        for column, stats in statistical_report["advanced_statistics"].items():
            if "is_normal" in stats and not stats["is_normal"]:
                non_normal_vars.append(column)
        
        if non_normal_vars:
            summary += f"- Non-normally distributed variables: {', '.join(non_normal_vars)}\n"
    
    print(f"[ADVANCED ANALYSIS] Statistical findings summary created with length {len(summary)}")
    return summary