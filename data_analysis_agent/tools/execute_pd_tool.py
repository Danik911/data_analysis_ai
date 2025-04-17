from llama_index.core.workflow import Context

async def execute_pandas_query_tool(ctx: Context, query_str: str) -> str:
    """
    Executes a pandas query string against the DataFrame in the context.
    Handles both queries returning results and assignments/inplace modifications.

    Args:
        ctx (Context): The workflow context containing 'dataframe' and 'query_engine'.
        query_str (str): The pandas query/command to execute. Must use 'df'.
    """
    try:
        query_engine: PandasQueryEngine = await ctx.get("query_engine")
        df: pd.DataFrame = await ctx.get("dataframe") # Get current DataFrame

        if df is None:
             return "Error: DataFrame not found in context."
        if query_engine is None:
             return "Error: PandasQueryEngine not found in context."

        print(f"Tool executing query: {query_str}")

        is_modification = '=' in query_str or 'inplace=True' in query_str

        if is_modification:
            try:
                # Prepare execution environment for modification
                local_vars = {'df': df.copy()} # Use a copy to avoid modifying original during exec
                global_vars = {'pd': pd} # Provide pandas module

                exec(query_str, global_vars, local_vars)

                # Get the modified DataFrame from the local scope of exec
                modified_df = local_vars['df']

                # Update the DataFrame in the workflow context
                await ctx.set("dataframe", modified_df)

                # Attempt to update the query engine's internal DataFrame state
                try:
                    query_engine._df = modified_df
                except AttributeError:
                     print("Warning: Could not directly update query_engine._df. Engine might use stale data for subsequent queries.")

                result = "Executed modification successfully."
                print(f"Tool modification result: {result}")
                return result

            except Exception as e:
                import traceback
                print(f"Tool exec error for query '{query_str}': {e}\n{traceback.format_exc()}")
                error_msg = f"Error executing modification '{query_str}': {e}"
                if "FutureWarning" in str(e) and "fillna" in query_str and "inplace=True" in query_str:
                    alt_query_str = query_str.replace(".fillna(", "['Time'].fillna(").replace(", inplace=True)", "") 
                    alt_query_str = f"df['Time'] = df{alt_query_str}" 
                    print(f"Attempting alternative syntax for fillna: {alt_query_str}")
                    try:
                        local_vars = {'df': df.copy()}
                        global_vars = {'pd': pd}
                        exec(alt_query_str, global_vars, local_vars)
                        modified_df = local_vars['df']
                        await ctx.set("dataframe", modified_df)
                        query_engine._df = modified_df
                        result = "Executed modification successfully (alternative fillna syntax)."
                        print(f"Tool modification result: {result}")
                        return result
                    except Exception as e_alt:
                         print(f"Alternative fillna syntax failed: {e_alt}")
                         

                return error_msg # Return original error if no specific handling worked
        else:
            try:
                response = query_engine.query(query_str)
                result = str(response)
                if "error" in result.lower() and ("syntax" in result.lower() or "invalid" in result.lower() or "Traceback" in result):
                     error_msg = f"Query engine failed for '{query_str}': {result}"
                     print(error_msg)
                     return error_msg
                print(f"Tool query result: {result[:500]}...")
                return result
            except Exception as e:
                 import traceback
                 print(f"Error during query_engine.query('{query_str}'): {e}\n{traceback.format_exc()}")
                 error_msg = f"Error during query_engine.query('{query_str}'): {e}"
                 return error_msg

    except Exception as e:
        import traceback
        print(f"Tool general error processing query '{query_str}': {e}\n{traceback.format_exc()}")
        error_msg = f"Error processing query '{query_str}': {e}"
        return error_msg