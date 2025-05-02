import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from pandas_helper import PandasHelper

def create_visualization_agent(df: pd.DataFrame, llm) -> FunctionCallingAgent:
    """
    Create a visualization agent with tools for standard and advanced visualizations.
    
    Args:
        df: The DataFrame to visualize
        llm: The language model to use for the agent
    
    Returns:
        FunctionCallingAgent configured with visualization tools
    """
    print("[VISUALIZATION] Creating visualization agent")
    # Create a helper instance with the DataFrame
    query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False)
    pandas_helper = PandasHelper(df, query_engine)

    # Create the standard visualization tool
    standard_visualization_tool = FunctionTool.from_defaults(
         async_fn=pandas_helper.generate_plots,
         name="generate_standard_visualizations_tool",
         description=pandas_helper.generate_plots.__doc__
    )
    
    # Create the advanced visualization tool
    advanced_visualization_tool = FunctionTool.from_defaults(
         async_fn=pandas_helper.generate_advanced_plots,
         name="generate_advanced_visualizations_tool",
         description="Generates advanced statistical plots including density plots, Q-Q plots, violin plots, correlation heatmaps, and pair plots."
    )

    # Create the visualization agent with both standard and advanced visualization tools
    visualization_agent = FunctionCallingAgent.from_tools(
        tools=[standard_visualization_tool, advanced_visualization_tool],
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are an advanced data visualization agent with expertise in creating both standard and advanced visualizations. "
            "Your task is to generate a comprehensive set of visualizations for the provided dataset using: \n"
            "1. 'generate_standard_visualizations_tool' for creating standard plots (histogram, countplot, scatterplot, boxplot)\n"
            "2. 'generate_advanced_visualizations_tool' for creating advanced plots (density plots, Q-Q plots, violin plots, correlation heatmaps, pair plots)\n"
            "Call both tools to ensure a complete visualization suite. After calling the tools, confirm that the plots have been generated "
            "and mention the directories they were saved in ('plots' for standard and 'plots/advanced' for advanced)."
        )
    )
    
    print("[VISUALIZATION] Visualization agent created successfully")
    return visualization_agent

async def generate_visualizations(df: pd.DataFrame, llm, modified_data_path: str) -> Dict[str, Any]:
    """
    Generate both standard and advanced visualizations for the given DataFrame
    
    Args:
        df: DataFrame containing the data to visualize
        llm: Language model to use for the agent
        modified_data_path: Path to the modified data file (for reference in prompt)
        
    Returns:
        Dictionary containing visualization results including confirmation and plot paths
    """
    print("[VISUALIZATION] Starting visualization generation process")
    print(f"[VISUALIZATION] DataFrame shape: {df.shape}")
    
    if df is None:
        print("[VISUALIZATION ERROR] DataFrame is None")
        return {
            "visualization_info": "Error: DataFrame missing for visualization.",
            "plot_paths": []
        }

    # Create the visualization agent
    visualization_agent = create_visualization_agent(df, llm)
    
    # Create the visualization request
    visualization_request = (
        f"The data analysis report is complete. Now, generate both standard and advanced visualizations "
        f"for the cleaned data (referenced by path: {modified_data_path}). To generate comprehensive visualizations:\n\n"
        f"1. First, use the 'generate_standard_visualizations_tool' to create standard plots (histogram, countplot, scatterplot, boxplot)\n"
        f"2. Then, use the 'generate_advanced_visualizations_tool' to create advanced statistical plots (density plots, Q-Q plots, violin plots, correlation heatmaps, pair plots)\n\n"
        f"Focus on columns 'Time', 'Distance', and 'Mode'. Ensure both visualization tools are called."
    )

    print(f"--- Prompting Enhanced Visualization Agent ---\n{visualization_request}\n---------------------------------")

    # Get the agent's response
    print("[VISUALIZATION] Waiting for agent response...")
    agent_response = await visualization_agent.achat(visualization_request)
    print("[VISUALIZATION] Received agent response")

    # Extract visualization information
    viz_confirmation = "Visualization agent did not provide confirmation."
    standard_plot_paths = []
    advanced_plot_paths = []

    # Extract results from tool calls if available
    if hasattr(agent_response, 'tool_calls'):
        print(f"[VISUALIZATION] Found {len(agent_response.tool_calls)} tool calls")
        for call in agent_response.tool_calls:
            if call.tool_name == "generate_standard_visualizations_tool" and hasattr(call, 'result'):
                standard_plot_paths = call.result
                print(f"[VISUALIZATION] Standard visualization tool returned {len(standard_plot_paths)} plot paths")
            elif call.tool_name == "generate_advanced_visualizations_tool" and hasattr(call, 'result'):
                advanced_plot_paths = call.result
                print(f"[VISUALIZATION] Advanced visualization tool returned {len(advanced_plot_paths)} plot paths")

    if hasattr(agent_response, 'response') and agent_response.response:
        viz_confirmation = agent_response.response
    else:
        print(f"[VISUALIZATION WARNING] Visualization agent response might not be the expected confirmation")
        viz_confirmation = str(agent_response)

    print(f"--- Enhanced Visualization Agent Confirmation ---\n{viz_confirmation}\n------------------------------------")

    # Combine standard and advanced plot paths
    all_plot_paths = standard_plot_paths + advanced_plot_paths
    print(f"[VISUALIZATION] Generated a total of {len(all_plot_paths)} plots")
    
    # Return the visualization results
    return {
        "visualization_info": viz_confirmation,
        "plot_paths": all_plot_paths  # Always return a list, even if empty
    }