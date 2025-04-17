from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
import os
from getpass import getpass
from dotenv import load_dotenv
from tools.execute_pd_tool import execute_pandas_query_tool
from tools.save_dataframe_tool import save_dataframe_tool

# Load environment variables and set up LLM
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or getpass("Enter OPENAI_API_KEY: ")
llm = OpenAI(model="gpt-4o-2024-11-20", api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=4096)

def create_agents():
    """Create and return the agents needed for our workflow"""

   
    
    pandas_query_tool = FunctionTool.from_defaults(
        async_fn=execute_pandas_query_tool,  
        name="execute_pandas_query_tool",
        description=execute_pandas_query_tool.__doc__
    )
    save_df_tool = FunctionTool.from_defaults(
        async_fn=save_dataframe_tool,       
        name="save_dataframe_tool",
        description=save_dataframe_tool.__doc__
    )


    data_prep_agent = FunctionCallingAgent.from_tools(
        tools=[],
        llm=llm,
        verbose=False,
        system_prompt="You are a data preparation agent. Your job is to describe the necessary steps to clean, transform, and prepare data for analysis based on provided statistics. "
                     "You handle tasks like dealing with missing values, normalizing data, feature engineering, and ensuring data quality."
    )
    
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