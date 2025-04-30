import os
import pandas as pd
import re
import functools
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.workflow import Event, Workflow, Context, StopEvent, step
from llama_index.core.workflow import StartEvent
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from pandas_helper import PandasHelper
from events import *
from agents import create_agents, llm

class DataAnalysisFlow(Workflow):
    
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> InitialAssessmentEvent: 
        """Initialize the agents and setup the workflow"""

        # --- Load data and create Pandas Query Engine ---
        try:
            df = pd.read_csv(ev.dataset_path)
            query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)

            # Store the DataFrame and query engine in the context
            await ctx.set("dataframe", df)
            await ctx.set("query_engine", query_engine)
            await ctx.set("original_path", ev.dataset_path)

            print(f"Successfully loaded {ev.dataset_path} and created PandasQueryEngine.")

            self.data_prep_agent, self.data_analysis_agent = create_agents()

            # --- Get initial stats for the next step ---
            initial_info_str = "Could not retrieve initial stats."
            column_info_dict = {}
            try:
                if hasattr(query_engine, 'aquery'):
                     response = await query_engine.aquery("Show the shape of the dataframe (number of rows and columns) and the output of df.describe(include='all')")
                else:
                     response = query_engine.query("Show the shape of the dataframe (number of rows and columns) and the output of df.describe(include='all')")
                initial_info_str = str(response)

                missing_counts = df.isna().sum().to_dict()
                dtypes = df.dtypes.astype(str).to_dict()
                column_info_dict = {"dtypes": dtypes, "missing_counts": missing_counts}
                print(f"--- Initial Info Gathered ---\n{initial_info_str}\nColumn Details:\n{column_info_dict}\n-----------------------------")
                # Store these in context for the consultation step later
                await ctx.set("stats_summary", initial_info_str)
                await ctx.set("column_info", column_info_dict)
            except Exception as e:
                print(f"Warning: Could not query initial info from engine during setup: {e}")
                initial_info_str = f"Columns: {df.columns.tolist()}" 
                column_info_dict = {"columns": df.columns.tolist()} 
                await ctx.set("stats_summary", initial_info_str) 
                await ctx.set("column_info", column_info_dict) 
            

            
            return InitialAssessmentEvent( 
                stats_summary=initial_info_str,
                column_info=column_info_dict,
                original_path=ev.dataset_path,
            )
        except Exception as e:
            print(f"Error during setup: Failed to load {ev.dataset_path} or create engine. Error: {e}")
            import traceback
            traceback.print_exc() 
            raise ValueError(f"Setup failed: {e}")
        
    @step
    async def data_preparation(self, ctx: Context, ev: InitialAssessmentEvent) -> DataAnalysisEvent: 
        """Use the data prep agent to suggest cleaning/preparation based on schema."""


        initial_info = ev.stats_summary # Get stats from the event
        column_info = ev.column_info

        prep_prompt = (
            f"The dataset (from {ev.original_path}) has the following shape and summary statistics:\\n{initial_info}\\nColumn Details:\\n{column_info}\\n\\n"
            f"Based *only* on these statistics, describe the necessary data preparation steps. "
            f"Specifically mention potential issues like outliers (e.g., in 'Distance' max value), missing values (e.g., count mismatch in 'Time'), "
            f"and data quality issues in categorical columns (e.g., unique count vs expected for 'Mode', potential typos like 'Bas', 'Cra', 'Walt'). "
            f"Suggest specific actions like imputation for 'Time', outlier investigation/handling for 'Distance', and checking unique values/correcting typos in 'Mode'. "
            f"Focus on describing *what* needs to be done and *why* based *strictly* on the provided stats. **Do NOT suggest normalization or scaling steps.** If no issues are apparent from the stats, state that clearly. ALWAYS provide a description."
            )
        result = self.data_prep_agent.chat(prep_prompt)

        prepared_data_description = None
        if hasattr(result, 'response'):
            prepared_data_description = result.response
            if not prepared_data_description:
                prepared_data_description = "Agent returned an empty description despite the prompt."
                print("Warning: Agent response attribute was empty.")

        else:
            prepared_data_description = "Could not extract data preparation description from agent response."
            print(f"Warning: Agent response does not have expected 'response' attribute. Full result: {result}")


        print(f"--- Prep Agent Description Output ---\\n{prepared_data_description}\\n------------------------------------")

        # Store the *agent's suggested* description (before human input)
        await ctx.set("agent_prepared_data_description", prepared_data_description)


        return DataAnalysisEvent(
            prepared_data_description=prepared_data_description, # Agent's initial suggestion
            original_path=ev.original_path
        )

    
    @step
    async def human_consultation(self, ctx: Context, ev: DataAnalysisEvent) -> ModificationRequestEvent: 
        """Analyzes initial assessment, asks user for cleaning decisions using numbered options.""" 
        print("--- Running Human Consultation Step ---")
        agent_suggestion = ev.prepared_data_description 
        original_path = ev.original_path
        stats_summary = await ctx.get("stats_summary", "Stats not available.")
        column_info = await ctx.get("column_info", {})

     
        consultation_agent = FunctionCallingAgent.from_tools(
            tools=[], 
            llm=llm,
            verbose=True,
            system_prompt=(
                "You are a data cleaning assistant. You are given an initial analysis and suggested cleaning steps. "
                "Your task is to formulate concise, **numbered options** for the user based *only* on the issues explicitly identified in the analysis (missing values, outliers, duplicates, data quality). "
                "**If no issues were identified for a category (e.g., no missing values found), do NOT ask about it.** "
                "For each identified issue, present the finding and suggest 1-3 common handling strategies as numbered options (e.g., 1. Fill median, 2. Fill mean, 3. Drop rows). "
                "Start numbering options from 1 and continue sequentially across all issues. "
                "Combine these into a single, clear message asking the user to reply with the **numbers** of their chosen options, separated by semicolons. Use the provided analysis as context.\n"
                "Example Output Format (if missing values and outliers were found, but no duplicates or quality issues):\n"
                "Based on the analysis:\n"
                "Missing Values ('Time'): 3 found.\n"
                "  1. Fill median\n"
                "  2. Fill mean\n"
                "  3. Drop rows\n"
                "Outliers ('Distance'): Max 99.0 is high.\n"
                "  4. Keep outliers\n"
                "  5. Remove outlier rows\n"
                "  6. Cap outliers at 95th percentile\n"
                "Please reply with the numbers of your chosen options, separated by semicolons (e.g., '1;5'): "
            )
        )

        consultation_prompt = f"Formulate numbered user questions based on this analysis/suggestion:\\n<analysis>\\n{agent_suggestion}\\n</analysis>\\n\\nAdditional Context:\\nStats Summary:\\n{stats_summary}\\nColumn Info:\\n{column_info}"
        print(f"--- Prompting Consultation Agent ---\\n{consultation_prompt}\\n---------------------------------")
        agent_response = await consultation_agent.achat(consultation_prompt)
        consultation_message = agent_response.response if hasattr(agent_response, 'response') else "Could not generate consultation message."

        print(f"--- Consultation Message ---\\n{consultation_message}\\n----------------------------")

        # --- Emit event to request user input ---
        issues_placeholder = {"message": consultation_message} # Keep original message for context
        print("Human Consultation: Emitting CleaningInputRequiredEvent...")
        ctx.write_event_to_stream(
            CleaningInputRequiredEvent(
                issues=issues_placeholder,
                prompt_message=consultation_message 
            )
        )

        # --- Wait for user response (expecting numbers) ---
        print("Human Consultation: Waiting for CleaningResponseEvent...")
        response_event = await ctx.wait_for_event(CleaningResponseEvent)
        print("Human Consultation: Received CleaningResponseEvent.")
        
        user_input_numbers = response_event.user_choices.get("numbers", "") # Get raw numeric string
        print(f"User chose numbers: {user_input_numbers}")

        # --- Agent to Translate Numbers to Description ---
        translation_agent = FunctionCallingAgent.from_tools(
            tools=[],
            llm=llm,
            verbose=True,
            system_prompt=(
                "You are given a text containing numbered options for data cleaning and a string containing the numbers selected by the user (separated by semicolons). "
                "Your task is to generate a clear, descriptive summary of the actions corresponding to the selected numbers. "
                "This summary will be used as instructions for another agent. "
                "Format the output as a list of actions.\n"
                "Example Input:\n"
                "Options Text: 'Based on the analysis:\\nMissing Values ('Time'): 3 found.\\n  1. Fill median\\n  2. Fill mean\\nOutliers ('Distance'): Max 99.0 is high.\\n  3. Keep outliers\\n  4. Remove outlier rows'\n"
                "Selected Numbers: '1;4'\n"
                "Example Output:\n"
                "Apply the following user-specified cleaning steps:\n"
                "- For missing values in 'Time', apply strategy: Fill median.\n"
                "- For outliers in 'Distance', apply strategy: Remove outlier rows.\n"
            )
        )

        translation_prompt = (
            f"Translate the selected numbers into a descriptive action plan.\n\n"
            f"Options Text:\n'''\n{consultation_message}\n'''\n\n"
            f"Selected Numbers: '{user_input_numbers}'\n\n"
            f"Generate the descriptive action plan:"
        )
        print(f"--- Prompting Translation Agent ---\\n{translation_prompt}\\n---------------------------------")
        translation_response = await translation_agent.achat(translation_prompt)
        user_approved_description = translation_response.response if hasattr(translation_response, 'response') else f"Could not translate choices: {user_input_numbers}"

        # Handle potential empty description from translation agent
        if not user_approved_description.strip() or "Could not translate" in user_approved_description:
             print(f"Warning: Translation agent failed or returned empty description. Using fallback.")
             user_approved_description = f"Apply user choices corresponding to numbers: {user_input_numbers} based on the options provided."


        print(f"--- Generated User-Approved Preparation Description ---\\n{user_approved_description}\\n---------------------------------------")

        # Pass the translated description to the next step
        return ModificationRequestEvent( 
            user_approved_description=user_approved_description,
            original_path=original_path
        )


    @step
    async def data_modification(self, ctx: Context, ev: ModificationRequestEvent) -> ModificationCompleteEvent: # Changed input event type
        """Applies the data modifications using a dedicated agent based on user input."""
        print("--- Running Data Modification Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        query_engine: PandasQueryEngine = await ctx.get("query_engine")
        original_path = ev.original_path # Get path from the event

        # Use a PandasHelper instance to manage modifications
        pandas_helper = PandasHelper(df, query_engine)
        pandas_query_tool_local = FunctionTool.from_defaults(
            async_fn=pandas_helper.execute_pandas_query,
            name="execute_pandas_query_tool",
            description=pandas_helper.execute_pandas_query.__doc__
        )

        modification_agent = FunctionCallingAgent.from_tools(
            tools=[pandas_query_tool_local],
            llm=llm,
            verbose=True,
            system_prompt=(
                "You are a data modification agent. Your task is to accurately execute pandas commands "
                "(using 'df' and the 'execute_pandas_query_tool') described in the provided text "
                "to clean and modify the DataFrame based on USER choices. Focus *only* on executing the modification steps described. "
                "**IMPORTANT: NEVER use `inplace=True` in your pandas commands.** Always use assignment, e.g., `df = df[condition]` or `df['col'] = df['col'].fillna(...)` or `df['col'] = df['col'].replace(...)`. "
                "If the description asks to standardize or correct typos in a categorical column (like 'Mode'): "
                "1. First, use the tool to query the unique values (e.g., `df['Mode'].unique()`). "
                "2. Based on the unique values returned and common sense for the likely categories (e.g., Car, Bus, Walk, Cycle, Bike), generate a `df['Mode'] = df['Mode'].replace({...})` command to correct *all* apparent typos (like 'Bas', 'Cra', 'Walt', 'Wilk', 'Cur', etc.) to their standard forms (e.g., 'Bus', 'Car', 'Walk'). "
                "If asked to remove outlier rows based on a specific column (e.g., 'Distance'), use a command like `df = df[df['Distance'] < threshold]` or follow the specific strategy if provided (e.g., quantile). Adjust the threshold reasonably if only max/min is given. "
                "If asked to fill missing values (e.g., in 'Time') use the specified method (mean or median) like `df['Time'] = df['Time'].fillna(df['Time'].mean())`." # Removed inplace=True example
            )
        )

        modification_request = (
            f"Apply the following USER-APPROVED data preparation steps using pandas commands with the 'execute_pandas_query_tool':\n"
            f"<preparation_description>\n{ev.user_approved_description}\n</preparation_description>" 
        )
        print(f"--- Prompting Data Modification Agent ---\\n{modification_request}\\n------------------------------------")

        await modification_agent.achat(modification_request)

       
        final_df = pandas_helper.get_final_dataframe()
        await ctx.set("dataframe", final_df)
        try:
           
            query_engine._df = final_df
            await ctx.set("query_engine", query_engine) 
        except AttributeError:
            print("Warning: Could not update main query engine's _df in context after modification step.")

        print("--- Data Modification Complete ---")

        return ModificationCompleteEvent(original_path=original_path)


    @step
    async def analysis_reporting(self, ctx: Context, ev: ModificationCompleteEvent) -> VisualizationRequestEvent:
        """Performs analysis on the modified data, generates a report, and saves."""
        print("--- Running Analysis & Reporting Step ---")
        df: pd.DataFrame = await ctx.get("dataframe") 
        original_path: str = ev.original_path 

        print("Analysis & Reporting: Creating new Query Engine with modified DataFrame.")
        query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)
        pandas_helper = PandasHelper(df, query_engine) # Pass the new engine


        pandas_query_tool_local = FunctionTool.from_defaults(
             async_fn=pandas_helper.execute_pandas_query,
             name="execute_pandas_query_tool",
             description=pandas_helper.execute_pandas_query.__doc__
        )
        save_df_tool_local = FunctionTool.from_defaults(
             async_fn=pandas_helper.save_dataframe,
             name="save_dataframe_tool",
             description=pandas_helper.save_dataframe.__doc__
        )

        analysis_reporting_agent = FunctionCallingAgent.from_tools(
            tools=[pandas_query_tool_local, save_df_tool_local],
            llm=llm,
            verbose=True,
            system_prompt=(
                "You are a data analysis and reporting agent. You work with an already modified DataFrame based on user decisions.\\n" # Added user decisions context
                "Your tasks are:\\n"
                "1. Perform analysis queries on the current DataFrame using 'execute_pandas_query_tool'.\\n"
                "2. Generate a concise Markdown report summarizing key findings from your analysis.\\n"
                "3. Save the current DataFrame using the 'save_dataframe_tool'."
            )
        )

        path_parts = os.path.splitext(original_path)
        modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"

        analysis_request = (
            f"The DataFrame (originally from {original_path}) has been modified based on prior user-approved cleaning steps.\\n" # Updated context
            f"Now, please perform the following actions:\\n"
            f"1. Perform a brief analysis on the modified data. For example, check the description of the 'Time' column (df['Time'].describe()), the unique values in 'Mode' (df['Mode'].unique()), and the description of 'Distance' (df['Distance'].describe()). Use the 'execute_pandas_query_tool'.\\n"
            f"2. Generate a Markdown report summarizing the key findings from your analysis of the modified data.\\n"
            f"3. Save the current DataFrame to the following path using the 'save_dataframe_tool': '{modified_file_path}'"
        )

        print(f"--- Prompting Analysis & Reporting Agent ---\\n{analysis_request}\\n------------------------------------")

        
        agent_response = await analysis_reporting_agent.achat(analysis_request)

        
        final_df = pandas_helper.get_final_dataframe() 
        await ctx.set("dataframe", final_df)

       
        final_report = "Agent did not provide a valid report."
        if hasattr(agent_response, 'response') and agent_response.response:
             final_report = agent_response.response
             
        else:
             print(f"Warning: Agent response might not be the expected report. Full result: {agent_response}")
             final_report = str(agent_response) 

        print(f"--- Analysis & Reporting Agent Final Response (Report) ---\\n{final_report}\\n------------------------------------------")
        await ctx.set("final_report", final_report)
        return VisualizationRequestEvent(
            modified_data_path=modified_file_path,
            report=final_report
        )
    
    @step
    async def create_visualizations(self, ctx: Context, ev: VisualizationRequestEvent) -> StopEvent:
        """Generates visualizations for the cleaned data using a dedicated agent."""
        print("--- Running Visualization Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        modified_data_path = ev.modified_data_path
        final_report = ev.report # Report from previous step

        if df is None:
            print("Error: DataFrame not found in context for visualization.")
            # Return previous report and error message
            return StopEvent(result={"final_report": final_report, "visualization_info": "Error: DataFrame missing for visualization."})

        # Create a helper instance with the final DataFrame
        query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False) # LLM might not be strictly needed if only plotting
        pandas_helper = PandasHelper(df, query_engine)

        # Create the visualization tool from the helper method
        visualization_tool_local = FunctionTool.from_defaults(
             async_fn=pandas_helper.generate_plots, # Use the new helper method
             name="generate_visualizations_tool",
             description=pandas_helper.generate_plots.__doc__
        )

        # Create the visualization agent
        visualization_agent = FunctionCallingAgent.from_tools(
            tools=[visualization_tool_local],
            llm=llm, # Ensure llm is accessible
            verbose=True,
            system_prompt=(
                "You are a data visualization agent. Your task is to generate insightful plots "
                "for the provided dataset using the 'generate_visualizations_tool'. "
                "The tool will generate standard plots (histogram, countplot, scatterplot, boxplot) "
                "for 'Time', 'Mode', and 'Distance' columns and save them into a 'plots' directory. "
                "Simply call the 'generate_visualizations_tool' function once to create all standard plots. "
                "After calling the tool, confirm that the plots have been generated and mention the directory they were saved in ('plots')."
            )
        )

        visualization_request = (
            f"The data analysis report is complete. Now, generate standard visualizations "
            f"for the cleaned data (referenced by path: {modified_data_path}) using the 'generate_visualizations_tool'. "
            f"Focus on columns 'Time', 'Distance', and 'Mode'."
        )

        print(f"--- Prompting Visualization Agent ---\n{visualization_request}\n---------------------------------")

        agent_response = await visualization_agent.achat(visualization_request)

        # The tool execution happens within the agent's call. The tool prints saved paths.
        # Get the agent's confirmation message.
        viz_confirmation = "Visualization agent did not provide confirmation."
        plot_paths = [] # Placeholder

        # Check if tool calls exist and extract results if possible (depends on LlamaIndex version/structure)
        if hasattr(agent_response, 'tool_calls'):
             for call in agent_response.tool_calls:
                 if call.tool_name == "generate_visualizations_tool" and hasattr(call, 'result'):
                     plot_paths = call.result # Assuming the result is the list of paths
                     print(f"Tool returned plot paths: {plot_paths}")
                     break # Assuming only one call needed

        if hasattr(agent_response, 'response') and agent_response.response:
             viz_confirmation = agent_response.response
        else:
             print(f"Warning: Visualization agent response might not be the expected confirmation. Full result: {agent_response}")
             viz_confirmation = str(agent_response) # Fallback

        print(f"--- Visualization Agent Confirmation ---\n{viz_confirmation}\n------------------------------------")

        # Include the report, confirmation, and potentially plot paths in the final result
        final_result = {
            "final_report": final_report,
            "visualization_info": viz_confirmation,
            "plot_paths": plot_paths if plot_paths else "Plot paths not retrieved from tool call."
        }
        return StopEvent(result=final_result)