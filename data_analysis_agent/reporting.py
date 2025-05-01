import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from pandas_helper import PandasHelper

def create_reporting_agent(df: pd.DataFrame, llm) -> Tuple[FunctionCallingAgent, PandasHelper]:
    """
    Create a reporting agent with tools for pandas operations and saving dataframes.
    
    Args:
        df: The DataFrame to analyze
        llm: The language model to use for the agent
    
    Returns:
        Tuple of (FunctionCallingAgent, PandasHelper)
    """
    print("[REPORTING] Creating reporting agent")
    # Create a query engine and helper for the DataFrame
    query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)
    pandas_helper = PandasHelper(df, query_engine)

    # Create tools for the reporting agent
    pandas_query_tool = FunctionTool.from_defaults(
         async_fn=pandas_helper.execute_pandas_query,
         name="execute_pandas_query_tool",
         description=pandas_helper.execute_pandas_query.__doc__
    )
    save_df_tool = FunctionTool.from_defaults(
         async_fn=pandas_helper.save_dataframe,
         name="save_dataframe_tool",
         description=pandas_helper.save_dataframe.__doc__
    )

    # Create the reporting agent
    reporting_agent = FunctionCallingAgent.from_tools(
        tools=[pandas_query_tool, save_df_tool],
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a data analysis and reporting agent with advanced capabilities. You work with an already modified DataFrame based on user decisions.\n"
            "Your tasks are:\n"
            "1. Perform analysis queries on the current DataFrame using 'execute_pandas_query_tool'.\n"
            "2. Generate a concise Markdown report summarizing key findings from your analysis.\n"
            "3. Incorporate advanced statistical findings into your report, including skewness, kurtosis, confidence intervals, and significance tests.\n"
            "4. Save the current DataFrame using the 'save_dataframe_tool'."
        )
    )
    
    print("[REPORTING] Reporting agent created successfully")
    return reporting_agent, pandas_helper

async def generate_report(
    df: pd.DataFrame, 
    llm, 
    original_path: str, 
    modification_summary: str, 
    statistical_summary: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report and save the modified dataframe.
    
    Args:
        df: The DataFrame to analyze
        llm: The language model to use for the agent
        original_path: Path to the original data file
        modification_summary: Summary of data modifications performed
        statistical_summary: Summary of advanced statistical analysis
        
    Returns:
        Dictionary containing final report and other relevant information
    """
    print("[REPORTING] Starting report generation process")
    print(f"[REPORTING] DataFrame shape: {df.shape}")
    
    # Create the reporting agent
    reporting_agent, pandas_helper = create_reporting_agent(df, llm)
    
    # Prepare the path for the modified file
    path_parts = os.path.splitext(original_path)
    modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"
    print(f"[REPORTING] Modified file path will be: {modified_file_path}")
    
    # Create the analysis request
    analysis_request = (
        f"The DataFrame (originally from {original_path}) has been modified and comprehensive advanced statistical analysis has been performed.\n\n"
        f"Data Cleaning Summary:\n{modification_summary}\n\n"
        f"Advanced Statistical Analysis Summary:\n{statistical_summary}\n\n"
        f"Now, please perform the following actions:\n"
        f"1. Perform additional analysis on the modified data as needed. For example, check the description of the 'Time' column (df['Time'].describe()), the unique values in 'Mode' (df['Mode'].unique()), and the description of 'Distance' (df['Distance'].describe()). Use the 'execute_pandas_query_tool'.\n"
        f"2. Generate a comprehensive Markdown report that includes:\n"
        f"   - A summary of the data preparation steps\n"
        f"   - Key findings from your analysis incorporating the advanced statistical measures (skewness, kurtosis, confidence intervals)\n"
        f"   - Interpretation of the significance tests between different modes of transport\n"
        f"   - Insights about the distribution of commute times and distances\n"
        f"3. Save the current DataFrame to the following path using the 'save_dataframe_tool': '{modified_file_path}'"
    )

    print(f"--- Prompting Analysis & Reporting Agent ---\n{analysis_request[:500]}...\n------------------------------------")
    
    # Get the agent's response
    print("[REPORTING] Waiting for agent response...")
    agent_response = await reporting_agent.achat(analysis_request)
    print("[REPORTING] Received agent response")
    
    # Extract the final report
    final_report = "Agent did not provide a valid report."
    if hasattr(agent_response, 'response') and agent_response.response:
         final_report = agent_response.response
         print(f"[REPORTING] Extracted report of length {len(final_report)}")
    else:
         print(f"[REPORTING WARNING] Agent response might not be the expected report")
         final_report = str(agent_response) 

    print(f"--- Analysis & Reporting Agent Final Response (Report) ---\n{final_report[:500]}...\n------------------------------------------")
    
    # Get the final DataFrame state
    final_df = pandas_helper.get_final_dataframe()
    print(f"[REPORTING] Final DataFrame shape: {final_df.shape}")
    
    print("[REPORTING] Report generation completed")
    return {
        "final_report": final_report,
        "modified_file_path": modified_file_path,
        "final_df": final_df
    }