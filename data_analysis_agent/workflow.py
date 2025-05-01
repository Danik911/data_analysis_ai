import os
import pandas as pd
import re
import functools
import json
import traceback
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.workflow import Event, Workflow, Context, StopEvent, step
from llama_index.core.workflow import StartEvent
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from pandas_helper import PandasHelper
from events import *
from agents import create_agents, llm
from data_quality import DataQualityAssessment, DataCleaner, assess_data_quality, clean_data
from statistical_analysis import generate_statistical_report

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

            # --- Perform Enhanced Data Quality Assessment ---
            print("Performing comprehensive data quality assessment...")
            assessment_report = assess_data_quality(df, save_report=True, 
                                      report_path='reports/data_quality_report.json')
            
            await ctx.set("assessment_report", assessment_report)
            print(f"Data quality assessment completed and stored in context with quality score: {assessment_report['dataset_info']['quality_score']}")

            # --- Format Quality Assessment Summary for Agent ---
            issue_summary = assessment_report['issue_summary']
            recommendations = assessment_report['recommendations']
            
            quality_summary = (
                f"Data Quality Assessment Summary:\n"
                f"- Total rows: {issue_summary['total_rows']}, Total columns: {issue_summary['total_columns']}\n"
                f"- Missing values: {issue_summary['missing_value_count']}\n"
                f"- Duplicate rows: {issue_summary['duplicate_row_count']}\n"
                f"- Outliers detected: {issue_summary['outlier_count']}\n"
                f"- Impossible values: {issue_summary['impossible_value_count']}\n"
                f"- Quality score: {assessment_report['dataset_info']['quality_score']}/100\n\n"
                f"Recommendations:\n"
            )
            
            for category, recs in recommendations.items():
                if recs:
                    quality_summary += f"- {category.replace('_', ' ').title()}:\n"
                    for rec in recs:
                        quality_summary += f"  * {rec}\n"

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
            
            # Combine quality assessment with basic stats
            combined_summary = f"{quality_summary}\n\nAdditional Statistics:\n{initial_info_str}"
            
            return InitialAssessmentEvent( 
                stats_summary=combined_summary,
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
        """Use the data prep agent to suggest cleaning/preparation based on schema and quality assessment."""

        initial_info = ev.stats_summary # Get enhanced stats and quality assessment from the event
        column_info = ev.column_info
        assessment_report = await ctx.get("assessment_report", None)

        # Enhanced prompt with quality assessment insights
        prep_prompt = (
            f"The dataset (from {ev.original_path}) has been analyzed with our enhanced data quality assessment tool. Here's the comprehensive summary:\n\n{initial_info}\n\n"
            f"Based on these statistics and quality assessment, describe the necessary data preparation steps. "
            f"Pay special attention to the recommendations from our data quality assessment tool, which has already identified issues using Tukey's method for outliers, systematic data type verification, and uniqueness verification. "
            f"For each issue category (missing values, outliers, duplicates, impossible values, data types), suggest specific actions with statistical justification. "
            f"Focus on describing *what* needs to be done and *why* based on the provided assessment and stats. If the assessment shows a high quality score with minimal issues, acknowledge that minimal cleaning is needed."
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


        print(f"--- Prep Agent Description Output ---\n{prepared_data_description}\n------------------------------------")

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
    async def data_modification(self, ctx: Context, ev: ModificationRequestEvent) -> ModificationCompleteEvent:
        """Applies the data modifications using the DataCleaner class based on user input.""" 
        print("--- Running Enhanced Data Modification Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        assessment_report = await ctx.get("assessment_report")
        original_path = ev.original_path # Get path from the event

        # Create a temporary backup of original data for before/after comparisons
        original_df = df.copy()
        
        print("Applying data cleaning using DataCleaner with quality assessment report...")
        
        # Use our enhanced DataCleaner class
        cleaned_df, cleaning_report = clean_data(
            df=df, 
            assessment_report=assessment_report, 
            save_report=True, 
            report_path='reports/cleaning_report.json',
            generate_plots=True, 
            plots_dir='plots/cleaning_comparisons'
        )
        
        print(f"Data cleaning completed with {len(cleaning_report['cleaning_log'])} steps")
        
        # Update the context with cleaned DataFrame
        await ctx.set("dataframe", cleaned_df)
        await ctx.set("cleaning_report", cleaning_report)
        
        # Update the query engine with the cleaned DataFrame
        query_engine = PandasQueryEngine(df=cleaned_df, llm=llm, verbose=True)
        await ctx.set("query_engine", query_engine)
        
        # Generate a summary of the cleaning performed
        cleaning_summary = "Data cleaning was performed with the following steps:\n"
        for i, step in enumerate(cleaning_report['cleaning_log'], 1):
            cleaning_summary += f"{i}. {step['action']}: "
            if step['action'] == 'standardize_mode_values' and 'changes' in step['details']:
                cleaning_summary += f"Standardized {step['details']['changes']} Mode values\n"
            elif step['action'] == 'handle_missing_values':
                cleaning_summary += f"Addressed missing values in {', '.join(step['details']['strategies'].keys())}\n"
            elif step['action'] == 'handle_outliers':
                cleaning_summary += f"Handled outliers in {', '.join(step['details']['columns'])} using {step['details']['method']} method\n"
            elif step['action'] == 'handle_duplicates':
                cleaning_summary += f"Removed {step['details']['duplicates_removed']} duplicate rows\n"
            elif step['action'] == 'handle_impossible_values':
                cleaning_summary += f"Fixed impossible values in {', '.join(step['details']['constraints'].keys())}\n"
            else:
                cleaning_summary += f"Completed\n"
                
        cleaning_summary += "\nBefore/After Metrics:\n"
        metrics = cleaning_report['metrics_comparison']
        cleaning_summary += f"- Rows: {metrics['row_count']['before']} → {metrics['row_count']['after']} ({metrics['row_count']['change']} change)\n"
        cleaning_summary += f"- Missing values: {metrics['missing_values']['before']} → {metrics['missing_values']['after']} ({metrics['missing_values']['change']} change)\n"
        
        if 'numeric_stats' in metrics:
            for col, stats in metrics['numeric_stats'].items():
                if 'mean' in stats:
                    cleaning_summary += f"- {col} mean: {stats['mean']['before']:.2f} → {stats['mean']['after']:.2f}\n"
        
        await ctx.set("modification_summary", cleaning_summary)
        print(f"--- Cleaning Summary ---\n{cleaning_summary}\n-------------------------")

        return ModificationCompleteEvent(
            original_path=original_path,
            modification_summary=cleaning_summary
        )

    @step
    async def advanced_statistical_analysis(self, ctx: Context, ev: ModificationCompleteEvent) -> AdvancedAnalysisCompleteEvent:
        """Performs advanced statistical analysis on the cleaned data including advanced measures and significance testing.""" 
        print("--- Running Advanced Statistical Analysis Step ---")
        print("[DEBUG] Starting advanced statistical analysis...")
        df: pd.DataFrame = await ctx.get("dataframe")
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
        print(f"[DEBUG] DataFrame data types: {df.dtypes.to_dict()}")
        
        original_path: str = ev.original_path
        modification_summary: str = ev.modification_summary

        # Create a helper instance with the cleaned DataFrame
        query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False)
        pandas_helper = PandasHelper(df, query_engine)
        
        # Create directory for reports if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Define the path for the statistical analysis report
        statistical_report_path = "reports/statistical_analysis_report.json"
        
        print("[DEBUG] Performing advanced statistical analysis...")
        
        # Perform advanced statistical analysis using the helper method
        print("[DEBUG] Calling pandas_helper.perform_advanced_analysis...")
        statistical_report = await pandas_helper.perform_advanced_analysis(
            save_report=True,
            report_path=statistical_report_path
        )
        
        # Store the statistical report in the context
        await ctx.set("statistical_report", statistical_report)
        print(f"[DEBUG] Statistical report generated and stored in context. Report keys: {statistical_report.keys()}")
        
        # Generate a summary of the statistical analysis
        summary = "Advanced Statistical Analysis Complete\n\n"
        
        try:
            # Add advanced statistics summary
            if "advanced_statistics" in statistical_report:
                summary += "## Advanced Statistics\n\n"
                print(f"[DEBUG] Processing advanced statistics for columns: {list(statistical_report['advanced_statistics'].keys())}")
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
                summary += "## Significance Tests\n\n"
                print(f"[DEBUG] Processing significance tests for columns: {list(statistical_report['significance_tests'].keys())}")
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
                mode_stats = statistical_report["group_statistics"]["Mode"]
                summary += "## Statistics by Mode\n\n"
                print(f"[DEBUG] Processing group statistics for Mode with columns: {list(mode_stats.keys())}")
                
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
            print(f"Error generating statistical summary: {e}")
            print(traceback.format_exc())
            summary += f"Error generating statistical summary: {e}\n"
            summary += "Full statistical report saved to JSON file.\n"
        
        print(f"--- Advanced Statistical Analysis Summary ---\n{summary}\n---------------------------------")
        
        # Store the summary in the context
        await ctx.set("statistical_summary", summary)
        
        # Generate advanced plots using the PandasHelper
        print("[DEBUG] Generating advanced visualizations...")
        advanced_plot_paths = await pandas_helper.generate_advanced_plots(output_dir="plots/advanced")
        
        if advanced_plot_paths:
            print(f"[DEBUG] Generated {len(advanced_plot_paths)} advanced plots")
            plot_info = "Advanced visualizations generated:\n"
            for path in advanced_plot_paths:
                if isinstance(path, str) and not path.startswith("Error"):
                    plot_info += f"- {path}\n"
                    print(f"[DEBUG] Generated advanced plot: {path}")
            
            print(f"--- Advanced Visualization Info ---\n{plot_info}\n---------------------------------")
            await ctx.set("advanced_plot_info", plot_info)
        else:
            print("[DEBUG] No advanced plots were generated or an error occurred")
        
        # Prepare path for modified file (same as in analysis_reporting step)
        path_parts = os.path.splitext(original_path)
        modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"
        print(f"[DEBUG] Modified file path: {modified_file_path}")
        
        print("[DEBUG] Completed advanced statistical analysis step")
        return AdvancedAnalysisCompleteEvent(
            modified_data_path=modified_file_path,
            report=summary,
            statistical_report_path=statistical_report_path
        )

    @step
    async def analysis_reporting(self, ctx: Context, ev: AdvancedAnalysisCompleteEvent) -> VisualizationRequestEvent:
        """Performs analysis on the modified data, generates a report, and saves.""" 
        print("--- Running Analysis & Reporting Step ---")
        df: pd.DataFrame = await ctx.get("dataframe") 
        original_path: str = ev.modified_data_path
        statistical_summary = ev.report
        statistical_report_path = ev.statistical_report_path

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
                "You are a data analysis and reporting agent with advanced capabilities. You work with an already modified DataFrame based on user decisions.\\n"
                "Your tasks are:\\n"
                "1. Perform analysis queries on the current DataFrame using 'execute_pandas_query_tool'.\\n"
                "2. Generate a concise Markdown report summarizing key findings from your analysis.\\n"
                "3. Incorporate advanced statistical findings into your report, including skewness, kurtosis, confidence intervals, and significance tests.\\n"
                "4. Save the current DataFrame using the 'save_dataframe_tool'."
            )
        )

        path_parts = os.path.splitext(original_path)
        modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"

        # Get the modification summary from context
        modification_summary = await ctx.get("modification_summary", "Modification summary not available.")

        analysis_request = (
            f"The DataFrame (originally from {original_path}) has been modified and comprehensive advanced statistical analysis has been performed.\\n\\n"
            f"Data Cleaning Summary:\\n{modification_summary}\\n\\n"
            f"Advanced Statistical Analysis Summary:\\n{statistical_summary}\\n\\n"
            f"Now, please perform the following actions:\\n"
            f"1. Perform additional analysis on the modified data as needed. For example, check the description of the 'Time' column (df['Time'].describe()), the unique values in 'Mode' (df['Mode'].unique()), and the description of 'Distance' (df['Distance'].describe()). Use the 'execute_pandas_query_tool'.\\n"
            f"2. Generate a comprehensive Markdown report that includes:\\n"
            f"   - A summary of the data preparation steps\\n"
            f"   - Key findings from your analysis incorporating the advanced statistical measures (skewness, kurtosis, confidence intervals)\\n"
            f"   - Interpretation of the significance tests between different modes of transport\\n"
            f"   - Insights about the distribution of commute times and distances\\n"
            f"3. Save the current DataFrame to the following path using the 'save_dataframe_tool': '{modified_file_path}'"
        )

        print(f"--- Prompting Analysis & Reporting Agent ---\\n{analysis_request[:500]}...\\n------------------------------------")

        agent_response = await analysis_reporting_agent.achat(analysis_request)

        final_df = pandas_helper.get_final_dataframe() 
        await ctx.set("dataframe", final_df)

        final_report = "Agent did not provide a valid report."
        if hasattr(agent_response, 'response') and agent_response.response:
             final_report = agent_response.response
        else:
             print(f"Warning: Agent response might not be the expected report. Full result: {agent_response}")
             final_report = str(agent_response) 

        print(f"--- Analysis & Reporting Agent Final Response (Report) ---\\n{final_report[:500]}...\\n------------------------------------------")
        await ctx.set("final_report", final_report)
        
        # Combine the report with information about the advanced plots
        advanced_plot_info = await ctx.get("advanced_plot_info", "Advanced plot information not available.")
        enhanced_report = final_report + "\n\n## Advanced Visualizations\n\n" + advanced_plot_info
        
        return VisualizationRequestEvent(
            modified_data_path=modified_file_path,
            report=enhanced_report
        )

    @step
    async def create_visualizations(self, ctx: Context, ev: VisualizationRequestEvent) -> StopEvent:
        """Generates standard and advanced visualizations for the cleaned data using a dedicated agent.""" 
        print("--- Running Enhanced Visualization Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        modified_data_path = ev.modified_data_path
        final_report = ev.report # Enhanced report from previous step with advanced analysis

        if df is None:
            print("Error: DataFrame not found in context for visualization.")
            # Return previous report and error message
            return StopEvent(result={"final_report": final_report, "visualization_info": "Error: DataFrame missing for visualization."})

        # Create a helper instance with the final DataFrame
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

        visualization_request = (
            f"The data analysis report is complete. Now, generate both standard and advanced visualizations "
            f"for the cleaned data (referenced by path: {modified_data_path}). To generate comprehensive visualizations:\n\n"
            f"1. First, use the 'generate_standard_visualizations_tool' to create standard plots (histogram, countplot, scatterplot, boxplot)\n"
            f"2. Then, use the 'generate_advanced_visualizations_tool' to create advanced statistical plots (density plots, Q-Q plots, violin plots, correlation heatmaps, pair plots)\n\n"
            f"Focus on columns 'Time', 'Distance', and 'Mode'. Ensure both visualization tools are called."
        )

        print(f"--- Prompting Enhanced Visualization Agent ---\n{visualization_request}\n---------------------------------")

        agent_response = await visualization_agent.achat(visualization_request)

        # Get the agent's confirmation message
        viz_confirmation = "Visualization agent did not provide confirmation."
        standard_plot_paths = []
        advanced_plot_paths = []

        # Extract results from tool calls if available
        if hasattr(agent_response, 'tool_calls'):
            for call in agent_response.tool_calls:
                if call.tool_name == "generate_standard_visualizations_tool" and hasattr(call, 'result'):
                    standard_plot_paths = call.result
                    print(f"Standard visualization tool returned plot paths: {standard_plot_paths}")
                elif call.tool_name == "generate_advanced_visualizations_tool" and hasattr(call, 'result'):
                    advanced_plot_paths = call.result
                    print(f"Advanced visualization tool returned plot paths: {advanced_plot_paths}")

        if hasattr(agent_response, 'response') and agent_response.response:
            viz_confirmation = agent_response.response
        else:
            print(f"Warning: Visualization agent response might not be the expected confirmation. Full result: {agent_response}")
            viz_confirmation = str(agent_response)

        print(f"--- Enhanced Visualization Agent Confirmation ---\n{viz_confirmation}\n------------------------------------")

        # Combine standard and advanced plot paths
        all_plot_paths = standard_plot_paths + advanced_plot_paths
        
        # Include the enhanced report, confirmation, and all plot paths in the final result
        final_result = {
            "final_report": final_report,
            "visualization_info": viz_confirmation,
            "plot_paths": all_plot_paths if all_plot_paths else "Plot paths not retrieved from tool calls."
        }
        
        return StopEvent(result=final_result)