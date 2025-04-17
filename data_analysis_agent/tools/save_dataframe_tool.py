from llama_index.core.workflow import Context

async def save_dataframe_tool(ctx: Context, file_path: str) -> str:
    """
    Saves the current DataFrame in the context to a CSV file.

    Args:
        file_path (str): The full path (including filename) where the CSV should be saved.
                         Example: 'C:/path/to/modified_data.csv'
    """
    try:
        df: pd.DataFrame = await ctx.get("dataframe")
        if df is None:
            return "Error: DataFrame not found in context."

        # --- Ensure the directory exists ---
        output_dir = os.path.dirname(file_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        print(f"Tool attempting to save DataFrame to: {file_path}")

        # Save the DataFrame
        df.to_csv(file_path, index=False)
        result = f"DataFrame successfully saved to {file_path}"
        print(result)
        return result
    except Exception as e:
        error_msg = f"Error saving DataFrame to '{file_path}': {e}"
        print(error_msg)
        return error_msg