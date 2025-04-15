# filepath: c:\Users\anteb\Desktop\Courses\Projects\data_analysis_ai\data_analysis_agent\pandas_helper.py
import pandas as pd
import os
import traceback
import re
from llama_index.experimental.query_engine import PandasQueryEngine

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