from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
import os
from getpass import getpass
from dotenv import load_dotenv
from tools.execute_pd_tool import execute_pandas_query_tool
from tools.save_dataframe_tool import save_dataframe_tool
from data_quality import DataQualityAssessment, DataCleaner, assess_data_quality, clean_data

# Load environment variables and set up LLM
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or getpass("Enter OPENAI_API_KEY: ")
llm = OpenAI(model="o3-mini-2025-01-31", api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=4096)

def create_agents():
    """Create and return the agents needed for our workflow"""

    # Define tool for executing pandas queries
    pandas_query_tool = FunctionTool.from_defaults(
        async_fn=execute_pandas_query_tool,  
        name="execute_pandas_query_tool",
        description=execute_pandas_query_tool.__doc__
    )
    
    # Define tool for saving dataframes
    save_df_tool = FunctionTool.from_defaults(
        async_fn=save_dataframe_tool,       
        name="save_dataframe_tool",
        description=save_dataframe_tool.__doc__
    )

    # Create data preparation agent with enhanced capabilities
    data_prep_agent = FunctionCallingAgent.from_tools(
        tools=[],
        llm=llm,
        verbose=False,
        system_prompt=(
            "You are a data preparation agent with enhanced capabilities. Your job is to describe the necessary steps to clean, transform, and prepare data for analysis based on provided statistics. "
            "You handle tasks like:\n"
            "1. Systematic data type verification for all columns\n"
            "2. Value range checking with statistical justification (e.g., Tukey's method for outliers)\n"
            "3. Uniqueness verification for Case Numbers\n"
            "4. Identification and handling of impossible values (negative distances/times, unreasonable values)\n"
            "5. Missing value analysis with pattern detection\n"
            "6. Outlier identification with Z-scores and IQR method\n"
            "7. Distribution analysis pre-cleaning with normality tests\n"
            "8. Standardization of categorical values with frequency analysis\n"
            "9. Documentation of cleaning decisions with statistical justification\n"
            "10. Before/after comparison metrics for transparency\n\n"
            "When analyzing data, provide detailed recommendations with statistical justification."
        )
    )
    
    # Create data analysis agent with tools for pandas operations and saving dataframes
    data_analysis_agent = FunctionCallingAgent.from_tools(
        tools=[pandas_query_tool, save_df_tool],
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a data analysis agent. Your job is to:\n"
            "1. Receive a data preparation description.\n"
            "2. Generate and execute pandas commands (using 'df') via the 'execute_pandas_query_tool' to perform the described cleaning/modifications (e.g., imputation, outlier handling, typo correction).\n"
            "3. Perform further analysis on the MODIFIED data using the 'execute_pandas_query_tool'.\n"
            "4. Generate a concise Markdown report summarizing:\n"
            "    - The cleaning/modification steps you executed.\n"
            "    - Key findings from your analysis of the modified data.\n"
            "5. Save the MODIFIED DataFrame to a new CSV file using the 'save_dataframe_tool'. Name the file by appending '_modified' to the original filename (e.g., if original was data.csv, save as data_modified.csv)."
        )
    )
    
    return data_prep_agent, data_analysis_agent