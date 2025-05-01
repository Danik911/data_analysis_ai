# filepath: c:\Users\anteb\Desktop\Courses\Projects\data_analysis_ai\data_analysis_agent\pandas_helper.py
import pandas as pd
import os
import traceback
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional, Any
from llama_index.experimental.query_engine import PandasQueryEngine
from statistical_analysis import (
    calculate_advanced_statistics,
    calculate_mode_statistics,
    perform_anova,
    perform_tukey_hsd,
    generate_advanced_plots,
    generate_statistical_report
)

class PandasHelper:
    """
    Helper class to manage pandas DataFrame operations and interactions 
    with a PandasQueryEngine, independent of the main workflow context.
    """
    def __init__(self, df: pd.DataFrame, query_engine: PandasQueryEngine):
        self.query_engine = query_engine
        # Work on a copy internally to avoid modifying the original df passed during init
        self._current_df = df.copy() 
        # Attempt to keep query engine synced with the internal df state
        try:
            # This assumes the experimental engine uses _df internally
            self.query_engine._df = self._current_df 
        except AttributeError:
            print("Warning: Could not set initial query_engine._df in PandasHelper.")

    

    async def execute_pandas_query(self, query_str: str) -> str:
        """
        Executes a pandas query string against the current DataFrame state.
        Attempts to distinguish between queries returning results and modifications.
        Args:
            query_str (str): The pandas query/command to execute. Must use 'df'.
        """
        try:
            print(f"Helper executing query: {query_str}")
            
            # --- Refined Heuristic for Modification ---
            # More specific check for assignments or inplace operations
            is_modification = (
                re.search(r"\bdf\s*\[.*\]\s*=", query_str) or  # df['col'] = ...
                re.search(r"\bdf\s*=\s*", query_str) or        # df = ...
                'inplace=True' in query_str or
                re.search(r"\.drop\(.*\)", query_str) or      # .drop(...) might be inplace or assignment
                re.search(r"\.fillna\(.*\)", query_str) or    # .fillna(...) might be inplace or assignment
                re.search(r"\.rename\(.*\)", query_str) or    # .rename(...) might be inplace or assignment
                re.search(r"\.replace\(.*\)", query_str)      # .replace(...) might be inplace or assignment
                # Add other modification patterns if needed
            )
            # --- End Refined Heuristic ---

            if is_modification:
                # --- Modification Logic (using exec) ---
                try:
                    local_vars = {'df': self._current_df.copy()} 
                    global_vars = {'pd': pd}
                    
                    exec(query_str, global_vars, local_vars)
                    
                    modified_df = local_vars['df']
                    
                    # Check if the DataFrame object actually changed
                    # This helps differentiate queries like `x = df[...]` from actual modifications
                    if not self._current_df.equals(modified_df):
                        self._current_df = modified_df
                        try:
                            self.query_engine._df = modified_df
                        except AttributeError:
                            print("Warning: Could not directly update query_engine._df after modification.")
                        result = "Executed modification successfully."
                        print(f"Helper modification result: {result}")
                    else:
                        # If df didn't change, it was likely a query assigning to a variable
                        # Try to capture the result if it's simple (e.g., unique_modes = df['Mode'].unique())
                        # This part is tricky and might need more robust handling
                        result_var_name = query_str.split('=')[0].strip()
                        if result_var_name in local_vars and result_var_name != 'df':
                             result = f"Executed query, result stored in '{result_var_name}': {str(local_vars[result_var_name])[:500]}..."
                             print(f"Helper query (via exec) result: {result}")
                        else:
                             result = "Executed command (likely query assignment), DataFrame unchanged."
                             print(f"Helper exec result: {result}")

                    return result
                    
                except Exception as e:
                    print(f"Helper exec error for query '{query_str}': {e}\n{traceback.format_exc()}")
                    error_msg = f"Error executing modification '{query_str}': {e}"
                    # Handle specific FutureWarning for fillna inplace (Example)
                    if "FutureWarning" in str(e) and "fillna" in query_str and "inplace=True" in query_str:
                         print("Note: Detected FutureWarning with inplace fillna. Consider using assignment syntax like 'df[col] = df[col].fillna(value)' instead.")
                    
                    return error_msg
            else: 
                # --- Query Logic (using query_engine) ---
                try:
                    try:
                         self.query_engine._df = self._current_df # Ensure engine has latest df
                    except AttributeError:
                         print("Warning: Could not directly update query_engine._df before query.")
                         
                    response = await self.query_engine.aquery(query_str) 
                    result = str(response)
                    
                    # Check for known error patterns from the engine's response string
                    if "error" in result.lower() or "Traceback" in result.lower() or "invalid syntax" in result.lower():
                         error_msg = f"Query engine failed for '{query_str}': {result}"
                         print(error_msg)
                         return error_msg
                         
                    print(f"Helper query engine result: {result[:500]}...")
                    return result
                except Exception as e:
                     print(f"Helper error during query_engine.aquery('{query_str}'): {e}\n{traceback.format_exc()}")
                     return f"Error during query_engine.aquery('{query_str}'): {e}"

        except Exception as e:
            print(f"Helper general error processing query '{query_str}': {e}\n{traceback.format_exc()}")
            return f"Error processing query '{query_str}': {e}"
    async def save_dataframe(self, file_path: str) -> str:
        """
        Saves the current DataFrame state to a CSV file.
        Args:
            file_path (str): The full path where the CSV should be saved.
        """
        try:
            output_dir = os.path.dirname(file_path)
            if output_dir: # Check if path includes a directory
                 os.makedirs(output_dir, exist_ok=True)

            print(f"Helper attempting to save DataFrame to: {file_path}")
            # Save the current internal DataFrame state
            self._current_df.to_csv(file_path, index=False) 
            result = f"DataFrame successfully saved to {file_path}"
            print(result)
            return result
        except Exception as e:
            error_msg = f"Error saving DataFrame to '{file_path}': {e}"
            print(error_msg)
            return error_msg

    def get_final_dataframe(self) -> pd.DataFrame:
        """Returns the final state of the DataFrame managed by the helper."""
        return self._current_df
    
    async def generate_plots(self, output_dir: str = "plots") -> list[str]:
        """
        Generates standard plots (histogram, countplot, scatterplot, boxplot)
        for the current DataFrame ('Time', 'Mode', 'Distance') and saves them
        to the specified directory. Returns a list of saved file paths.

        Args:
            output_dir (str): The directory to save the plots in. Defaults to 'plots'.
        """
        plot_paths = []
        df = self._current_df # Use the helper's current dataframe

        if df is None:
            return ["Error: DataFrame not available in helper."]

        try:
            os.makedirs(output_dir, exist_ok=True)
            sns.set_theme(style="whitegrid")
            plot_context = "Plotting context active" # Placeholder for actual context management if needed

            # 1. Histogram of Commute Times
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Time'], kde=True, bins=15)
            plt.title('Distribution of Commute Times')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Frequency')
            hist_path = os.path.join(output_dir, "time_histogram.png")
            plt.savefig(hist_path)
            plt.close() # Close plot to free memory
            plot_paths.append(hist_path)
            print(f"Saved plot: {hist_path}")

            # 2. Bar Chart of Commute Modes
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Mode', order=df['Mode'].value_counts().index)
            plt.title('Count of Commute Modes')
            plt.xlabel('Mode of Transport')
            plt.ylabel('Count')
            count_path = os.path.join(output_dir, "mode_countplot.png")
            plt.savefig(count_path)
            plt.close()
            plot_paths.append(count_path)
            print(f"Saved plot: {count_path}")

            # 3. Scatter Plot of Distance vs. Time
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='Distance', y='Time', hue='Mode')
            plt.title('Commute Distance vs. Time by Mode')
            plt.xlabel('Distance (km)')
            plt.ylabel('Time (minutes)')
            plt.legend(title='Mode')
            scatter_path = os.path.join(output_dir, "distance_time_scatter.png")
            plt.savefig(scatter_path)
            plt.close()
            plot_paths.append(scatter_path)
            print(f"Saved plot: {scatter_path}")

            # 4. Box Plot of Time by Mode
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='Mode', y='Time')
            plt.title('Commute Time Distribution by Mode')
            plt.xlabel('Mode of Transport')
            plt.ylabel('Time (minutes)')
            box_path = os.path.join(output_dir, "time_mode_boxplot.png")
            plt.savefig(box_path)
            plt.close()
            plot_paths.append(box_path)
            print(f"Saved plot: {box_path}")

            return plot_paths

        except Exception as e:
            error_msg = f"Error generating plots: {e}\n{traceback.format_exc()}"
            print(error_msg)
            # Ensure plot is closed in case of error during saving
            plt.close()
            return [error_msg]
        finally:
            # Reset matplotlib state if necessary, or manage context
            plt.rcdefaults() # Example reset, might need adjustment

    async def generate_advanced_plots(self, output_dir: str = "plots/advanced") -> List[str]:
        """
        Generates advanced statistical plots including density plots, Q-Q plots, violin plots,
        correlation heatmaps, and pair plots. Uses functions from statistical_analysis module.

        Args:
            output_dir (str): The directory to save the advanced plots in. Defaults to 'plots/advanced'.
            
        Returns:
            List of saved plot file paths
        """
        df = self._current_df # Use the helper's current dataframe
        
        if df is None:
            return ["Error: DataFrame not available in helper."]
            
        try:
            # Call the generate_advanced_plots function from statistical_analysis module
            plot_paths = generate_advanced_plots(df, output_dir)
            return plot_paths
        except Exception as e:
            error_msg = f"Error generating advanced plots: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return [error_msg]
            
    async def perform_advanced_analysis(self, save_report: bool = True, 
                                 report_path: str = "reports/statistical_analysis_report.json") -> Dict[str, Any]:
        """
        Performs advanced statistical analysis on the current DataFrame including:
        - Advanced statistics (skewness, kurtosis, confidence intervals)
        - Group-based statistics for each Mode
        - Statistical significance testing (ANOVA and Tukey's HSD)
        
        Args:
            save_report (bool): Whether to save the analysis report to a JSON file
            report_path (str): Path to save the report if save_report is True
            
        Returns:
            Dictionary containing the statistical analysis report
        """
        df = self._current_df
        
        if df is None:
            return {"error": "DataFrame not available in helper."}
            
        try:
            print("[DEBUG] Starting advanced statistical analysis...")
            print(f"[DEBUG] DataFrame shape: {df.shape}")
            print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
            print(f"[DEBUG] DataFrame data types: {df.dtypes.to_dict()}")
            
            print("[DEBUG] Performing advanced statistical analysis...")
            
            # Generate comprehensive statistical report
            overall_stats = calculate_advanced_statistics(df)
            mode_stats = calculate_mode_statistics(df)
            normality_test_results = perform_anova(df)
            
            # Fix any non-serializable values (convert to string or other serializable types)
            def make_json_serializable(obj):
                if isinstance(obj, bool):
                    return str(obj)  # Convert boolean to string
                elif isinstance(obj, (int, float, str, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                else:
                    return str(obj)  # Convert any other type to string
            
            # Prepare report
            report = {
                'overall_statistics': make_json_serializable(overall_stats),
                'mode_statistics': make_json_serializable(mode_stats),
                'normality_tests': make_json_serializable(normality_test_results)
            }
            
            # Save report if requested
            if save_report:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"[DEBUG] Advanced statistical report saved to {report_path}")
                
            return report
        except Exception as e:
            error_msg = f"Error performing advanced analysis: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return {"error": error_msg}