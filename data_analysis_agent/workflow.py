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

# Import refactored modules
from visualization import generate_visualizations
from reporting import generate_report
from consultation import handle_user_consultation
from advanced_analysis import perform_advanced_analysis, summarize_statistical_findings, perform_advanced_modeling

# Import new Phase 3 modules
from regression_analysis import perform_regression_analysis, RegressionModel
from model_validation import validate_regression_model
from predictive_application import generate_prediction_examples

# Add a new event for regression analysis
class RegressionModelingEvent(Event):
    """Event triggered after advanced analysis to perform regression modeling"""
    modified_data_path: str
    statistical_report_path: str


class RegressionCompleteEvent(Event):
    """Event triggered when regression modeling is complete"""
    modified_data_path: str
    regression_summary: str
    model_quality: str

class VisualizationCompleteEvent(Event):
    """Event triggered when visualization is complete"""
    final_report: str
    visualization_info: str
    plot_paths: list

class FinalizeReportsEvent(Event):
    """Event triggered to finalize all reports"""
    final_report: str
    visualization_info: str
    plot_paths: list
    reports_to_verify: list


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
            
            # Add tracking for required reports
            await ctx.set("required_reports", [
                "reports/data_quality_report.json",
                "reports/cleaning_report.json", 
                "reports/statistical_analysis_report.json",
                "reports/regression_models.json",
                "reports/advanced_models.json"
            ])
            
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
            prepared_data_description=prepared_data_description, 
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
        
        # Use the refactored consultation module
        consultation_result = await handle_user_consultation(
            ctx, 
            llm,
            agent_suggestion,
            stats_summary,
            column_info,
            original_path
        )
        
        return ModificationRequestEvent( 
            user_approved_description=consultation_result["user_approved_description"],
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
    async def advanced_statistical_analysis(self, ctx: Context, ev: ModificationCompleteEvent) -> RegressionModelingEvent:
        """Performs advanced statistical analysis on the cleaned data including advanced measures and significance testing.""" 
        print("--- Running Advanced Statistical Analysis Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        original_path: str = ev.original_path
        modification_summary: str = ev.modification_summary
        
        # Use the refactored advanced analysis module
        analysis_results = await perform_advanced_analysis(
            df=df,
            llm=llm,
            original_path=original_path,
            modification_summary=modification_summary
        )
        
        # Store results in context
        await ctx.set("statistical_report", analysis_results["statistical_report"])
        await ctx.set("statistical_summary", analysis_results["summary"])
        await ctx.set("advanced_plot_info", analysis_results["plot_info"])
        
        # Continue to regression modeling step
        return RegressionModelingEvent(
            modified_data_path=analysis_results["modified_data_path"],
            statistical_report_path=analysis_results["statistical_report_path"]
        )

    @step
    async def regression_modeling(self, ctx: Context, ev: RegressionModelingEvent) -> RegressionCompleteEvent:
        """Performs regression analysis including linear regression and advanced models."""
        print("--- Running Regression Modeling Step (Phase 3) ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        target_column = 'Time'
        predictor_column = 'Distance'
        
        # 1. Perform Linear Regression Analysis
        print("[REGRESSION] Starting linear regression analysis")
        regression_results = perform_regression_analysis(
            df=df,
            target_column=target_column, 
            predictor_column=predictor_column,
            save_report=True,
            report_path="reports/regression_models.json",
            generate_plots=True,
            plots_dir="plots/regression"
        )
        
        # Store regression model in context for later use
        regression_model = RegressionModel(df, target_column, predictor_column)
        regression_model.fit_full_dataset_model()
        if 'Mode' in df.columns:
            regression_model.fit_mode_specific_models(mode_column='Mode')
        await ctx.set("regression_model", regression_model)
        
        # Store regression results in context
        await ctx.set("regression_results", regression_results)
        
        # 2. Perform Model Validation
        print("[REGRESSION] Validating regression models")
        full_model = regression_model.models.get('full_dataset')
        if full_model:
            X = df[[predictor_column]]
            y = df[target_column]
            
            validation_results = validate_regression_model(
                model=full_model,
                X=X,
                y=y,
                model_name="Time-Distance Regression",
                feature_names=[predictor_column],
                save_report=True,
                report_path="reports/model_validation.json",
                generate_plots=True,
                plots_dir="plots/validation"
            )
            
            # Store validation results
            await ctx.set("validation_results", validation_results)
            
            # Check if model meets assumptions
            assumptions_met = validation_results.get('assumptions_met', {})
            model_quality = "High" if all(assumptions_met.values()) else "Medium" if any(assumptions_met.values()) else "Low"
            await ctx.set("model_quality", model_quality)
        else:
            validation_results = {"status": "error", "error_message": "Full dataset model not available"}
            model_quality = "Unknown"
            await ctx.set("model_quality", model_quality)
        
        # 3. Perform Advanced Modeling
        print("[REGRESSION] Running advanced modeling analysis")
        advanced_modeling_results = perform_advanced_modeling(
            df=df,
            target_column=target_column,
            predictor_column=predictor_column,
            save_report=True,
            report_path="reports/advanced_models.json",
            generate_plots=True,
            plots_dir="plots/models"
        )
        
        # Store advanced modeling results
        await ctx.set("advanced_modeling_results", advanced_modeling_results)
        
        # 4. Generate Prediction Examples
        print("[REGRESSION] Generating prediction examples")
        prediction_results = generate_prediction_examples(
            regression_model=regression_model,
            save_report=True,
            report_path="reports/prediction_results.json",
            generate_plots=True,
            plots_dir="plots/predictions"
        )
        
        # Store prediction results
        await ctx.set("prediction_results", prediction_results)
        
        # Prepare regression summary
        regression_summary = "## Regression Analysis Summary\n\n"
        
        # Add linear model summary
        if regression_results.get('status') == 'success':
            full_model_info = regression_results.get('full_model', {})
            regression_summary += "### Linear Model\n"
            regression_summary += f"- Formula: {full_model_info.get('formula', 'N/A')}\n"
            metrics = full_model_info.get('metrics', {})
            regression_summary += f"- R-squared: {metrics.get('r_squared', 'N/A'):.4f}\n"
            regression_summary += f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
            
            # Add mode-specific models summary if available
            mode_models = regression_results.get('mode_models', {})
            if mode_models:
                regression_summary += "\n### Mode-Specific Models\n"
                for mode, model_info in mode_models.items():
                    regression_summary += f"- **{mode}**: {model_info.get('formula', 'N/A')}\n"
                    mode_metrics = model_info.get('metrics', {})
                    regression_summary += f"  - R-squared: {mode_metrics.get('r_squared', 'N/A'):.4f}\n"
            
        # Add model validation summary
        if validation_results.get('status') == 'success':
            regression_summary += "\n### Model Validation\n"
            assumptions = validation_results.get('assumptions_met', {})
            regression_summary += f"- Normality of residuals: {'✅ Met' if assumptions.get('normality', False) else '❌ Not met'}\n"
            regression_summary += f"- Homoscedasticity: {'✅ Met' if assumptions.get('homoscedasticity', False) else '❌ Not met'}\n"
            regression_summary += f"- Zero mean residuals: {'✅ Met' if assumptions.get('zero_mean_residuals', False) else '❌ Not met'}\n"
            
            cv_metrics = validation_results.get('cross_validation', {}).get('metrics', {})
            regression_summary += f"- Cross-validation R²: {cv_metrics.get('r2_mean', 'N/A'):.4f} (±{cv_metrics.get('r2_std', 'N/A'):.4f})\n"
        
        # Add advanced modeling summary
        if advanced_modeling_results.get('status') == 'success':
            comparison = advanced_modeling_results.get('model_comparison', {})
            best_model_key = comparison.get('overall_best_model')
            
            regression_summary += "\n### Alternative Models\n"
            
            if best_model_key and best_model_key in comparison:
                best_model = comparison.get(best_model_key, {})
                regression_summary += f"- Best model: {best_model.get('name', 'Unknown')}\n"
                
                best_metrics = best_model.get('metrics', {})
                if best_metrics:
                    regression_summary += f"- AIC: {best_metrics.get('aic', 'N/A'):.2f}\n"
                    regression_summary += f"- BIC: {best_metrics.get('bic', 'N/A'):.2f}\n"
                    regression_summary += f"- R-squared: {best_metrics.get('r2', 'N/A'):.4f}\n"
        
        # Add prediction example summary
        if prediction_results.get('status') == 'success':
            regression_summary += "\n### Predictions\n"
            regression_summary += "- Prediction examples generated for full dataset model"
            if 'mode_bus' in regression_model.models:
                regression_summary += " and mode-specific models"
            regression_summary += "\n"
            regression_summary += "- See prediction plots in plots/predictions/ directory\n"
        
        return RegressionCompleteEvent(
            modified_data_path=ev.modified_data_path,
            regression_summary=regression_summary,
            model_quality=model_quality
        )

    @step
    async def analysis_reporting(self, ctx: Context, ev: RegressionCompleteEvent) -> VisualizationRequestEvent:
        """Performs analysis on the modified data, generates a report, and saves.""" 
        print("--- Running Analysis & Reporting Step ---")
        df: pd.DataFrame = await ctx.get("dataframe") 
        original_path: str = ev.modified_data_path
        
        # Get the modification summary from context
        modification_summary = await ctx.get("modification_summary", "Modification summary not available.")
        
        # Get the statistical summary from context
        statistical_summary = await ctx.get("statistical_summary", "Statistical summary not available.")
        
        # Add regression summary to the report
        combined_summary = statistical_summary + "\n\n" + ev.regression_summary
        
        # Use the refactored reporting module
        reporting_results = await generate_report(
            df=df,
            llm=llm,
            original_path=original_path,
            modification_summary=modification_summary,
            statistical_summary=combined_summary
        )
        
        # Update context with final dataframe
        await ctx.set("dataframe", reporting_results["final_df"])
        await ctx.set("final_report", reporting_results["final_report"])
        
        # Combine the report with information about the advanced plots and regression quality
        advanced_plot_info = await ctx.get("advanced_plot_info", "Advanced plot information not available.")
        
        # Add model quality information
        model_quality_info = f"\n\n## Model Quality Assessment\n\nRegression Model Quality: {ev.model_quality}\n"
        
        enhanced_report = reporting_results["final_report"] + "\n\n## Advanced Visualizations\n\n" + advanced_plot_info + model_quality_info
        
        return VisualizationRequestEvent(
            modified_data_path=reporting_results["modified_file_path"],
            report=enhanced_report
        )

    @step
    async def create_visualizations(self, ctx: Context, ev: VisualizationRequestEvent) -> FinalizeReportsEvent:
        """Generates standard and advanced visualizations for the cleaned data using a dedicated agent.""" 
        print("--- Running Enhanced Visualization Step ---")
        df: pd.DataFrame = await ctx.get("dataframe")
        modified_data_path = ev.modified_data_path
        final_report = ev.report # Enhanced report from previous step with advanced analysis

        if df is None:
            print("Error: DataFrame not found in context for visualization.")
            # Return with error but continue to report finalization
            return FinalizeReportsEvent(
                final_report=final_report,
                visualization_info="Error: DataFrame missing for visualization.",
                plot_paths=[],
                reports_to_verify=await ctx.get("required_reports", [])
            )

        # Use the refactored visualization module
        visualization_results = await generate_visualizations(df, llm, modified_data_path)
        
        # Get the list of required reports that need to be verified
        required_reports = await ctx.get("required_reports", [
            "reports/data_quality_report.json",
            "reports/cleaning_report.json", 
            "reports/statistical_analysis_report.json",
            "reports/regression_models.json",
            "reports/advanced_models.json"
        ])
        
        # Continue to report finalization step
        return FinalizeReportsEvent(
            final_report=final_report,
            visualization_info=visualization_results["visualization_info"],
            plot_paths=visualization_results["plot_paths"],
            reports_to_verify=required_reports
        )

    @step
    async def finalize_reports(self, ctx: Context, ev: FinalizeReportsEvent) -> StopEvent:
        """Verifies and finalizes all reports as the last step in the workflow."""
        print("--- Running Report Finalization Step ---")
        final_report = ev.final_report
        visualization_info = ev.visualization_info
        plot_paths = ev.plot_paths
        reports_to_verify = ev.reports_to_verify
        
        # Verify reports and regenerate any missing or incomplete ones
        reports_status = {}
        
        for report_path in reports_to_verify:
            print(f"Verifying report: {report_path}")
            status = {"exists": False, "complete": False, "error": None}
            
            try:
                # Check if report exists
                if os.path.exists(report_path):
                    status["exists"] = True
                    
                    # Read report and check if it's a valid JSON
                    with open(report_path, 'r') as f:
                        try:
                            report_data = json.load(f)
                            
                            # Check if the report has content
                            if report_data and isinstance(report_data, dict):
                                status["complete"] = True
                            else:
                                status["error"] = "Report exists but contains no valid data"
                                print(f"Error: {report_path} exists but contains no valid data")
                                
                        except json.JSONDecodeError:
                            status["error"] = "Invalid JSON format"
                            print(f"Error: {report_path} is not a valid JSON file")
                else:
                    status["error"] = "Report file does not exist"
                    print(f"Error: {report_path} does not exist")
                    
                # Store status for this report
                reports_status[report_path] = status
                
                # If report is incomplete or missing, attempt to regenerate it
                if not status["complete"]:
                    await self._regenerate_report(ctx, report_path)
            
            except Exception as e:
                status["error"] = str(e)
                reports_status[report_path] = status
                print(f"Error verifying report {report_path}: {e}")
                
                # Attempt to regenerate on error
                await self._regenerate_report(ctx, report_path)
        
        # Update the final report with the status of all reports
        report_status_summary = "\n\n## Reports Status\n\n"
        for report_path, status in reports_status.items():
            report_name = os.path.basename(report_path)
            if status["complete"]:
                report_status_summary += f"- ✅ {report_name}: Successfully generated\n"
            else:
                error = status["error"] or "Unknown error"
                report_status_summary += f"- ⚠️ {report_name}: Issue detected - {error}\n"
        
        final_report_with_status = final_report + report_status_summary
        
        # Create a condensed final result
        final_result = {
            "final_report": final_report_with_status,
            "visualization_info": visualization_info,
            "plot_paths": plot_paths,
            "reports_status": reports_status
        }
        
        print("Report finalization complete. Workflow finished.")
        return StopEvent(result=final_result)
    
    async def _regenerate_report(self, ctx: Context, report_path: str) -> None:
        """Helper method to attempt regenerating a missing or incomplete report."""
        print(f"Attempting to regenerate report: {report_path}")
        
        try:
            df = await ctx.get("dataframe")
            report_name = os.path.basename(report_path)
            
            if "data_quality_report" in report_path:
                # Regenerate data quality report
                print("Regenerating data quality report...")
                assess_data_quality(df, save_report=True, report_path=report_path)
                
            elif "cleaning_report" in report_path:
                # Regenerate cleaning report
                print("Regenerating cleaning report...")
                assessment_report = await ctx.get("assessment_report", None)
                if assessment_report:
                    clean_data(
                        df=df,
                        assessment_report=assessment_report,
                        save_report=True,
                        report_path=report_path,
                        generate_plots=False  # Skip plots during regeneration
                    )
                
            elif "regression_models" in report_path:
                # Regenerate regression models report
                print("Regenerating regression models report...")
                regression_model = await ctx.get("regression_model")
                if regression_model:
                    regression_model.save_model_results(file_path=report_path)
                else:
                    # Try to rebuild the model if not in context
                    perform_regression_analysis(
                        df=df,
                        target_column='Time',
                        predictor_column='Distance',
                        save_report=True,
                        report_path=report_path,
                        generate_plots=False  # Skip plots during regeneration
                    )
                    
            elif "statistical_analysis_report" in report_path:
                # Regenerate statistical analysis report
                print("Regenerating statistical analysis report...")
                generate_statistical_report(
                    df=df,
                    save_report=True,
                    report_path=report_path
                )
                
            elif "advanced_models" in report_path:
                # Regenerate advanced models report
                print("Regenerating advanced models report...")
                perform_advanced_modeling(
                    df=df,
                    target_column='Time',
                    predictor_column='Distance',
                    save_report=True,
                    report_path=report_path,
                    generate_plots=False  # Skip plots during regeneration
                )
                
            print(f"Successfully regenerated report: {report_path}")
            
        except Exception as e:
            print(f"Failed to regenerate report {report_path}: {str(e)}")
            traceback.print_exc()