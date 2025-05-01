import os
import json
import pandas as pd
import traceback
import numpy as np
from typing import Dict, Any, List, Optional
from pandas_helper import PandasHelper
from llama_index.experimental.query_engine import PandasQueryEngine
from statistical_analysis import generate_statistical_report

# New imports for advanced regression models
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class UtilsManager:
    """Utility class for managing common operations like JSON serialization"""
    
    @staticmethod
    def make_json_serializable(obj) -> Any:
        """
        Convert any non-JSON-serializable values to serializable types
        
        Args:
            obj: The object to convert to JSON serializable format
        
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, (str, int, float, type(None))):
            return obj
        elif isinstance(obj, bool):
            return str(obj)  # Convert boolean to string
        elif isinstance(obj, (list, tuple)):
            return [UtilsManager.make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: UtilsManager.make_json_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)  # Convert any other type to string


class StatisticalAnalyzer:
    """Class for advanced statistical analysis functionality"""
    
    def __init__(self, df: pd.DataFrame, llm=None):
        """
        Initialize the statistical analyzer
        
        Args:
            df: The DataFrame to analyze
            llm: The language model to use for the PandasQueryEngine
        """
        self.df = df
        self.llm = llm
        self.query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False) if llm else None
        self.pandas_helper = PandasHelper(df, self.query_engine) if self.query_engine else None
    
    async def perform_analysis(self, original_path: str, modification_summary: str = None) -> Dict[str, Any]:
        """
        Perform advanced statistical analysis on the data
        
        Args:
            original_path: Path to the original data file
            modification_summary: Summary of data modifications performed
            
        Returns:
            Dictionary containing statistical analysis results
        """
        print("[ADVANCED ANALYSIS] Starting advanced statistical analysis")
        print(f"[ADVANCED ANALYSIS] DataFrame shape: {self.df.shape}")
        print(f"[ADVANCED ANALYSIS] DataFrame columns: {self.df.columns.tolist()}")
        print(f"[ADVANCED ANALYSIS] DataFrame data types: {self.df.dtypes.to_dict()}")
        
        # Create directory for reports if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Define the path for the statistical analysis report
        statistical_report_path = "reports/statistical_analysis_report.json"
        
        statistical_report = self._generate_statistical_report(statistical_report_path)
        summary = self._generate_summary(statistical_report)
        plot_info = await self._generate_plots()
        
        # Prepare path for modified file
        path_parts = os.path.splitext(original_path)
        modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"
        print(f"[ADVANCED ANALYSIS] Modified file path: {modified_file_path}")
        
        print("[ADVANCED ANALYSIS] Advanced statistical analysis completed")
        
        return {
            "statistical_report": statistical_report,
            "summary": summary,
            "plot_info": plot_info,
            "statistical_report_path": statistical_report_path,
            "modified_data_path": modified_file_path
        }
    
    def _generate_statistical_report(self, report_path: str) -> Dict[str, Any]:
        """
        Generate and save the statistical report
        
        Args:
            report_path: Path to save the report
            
        Returns:
            Dictionary containing the statistical report
        """
        try:
            from statistical_analysis import (
                calculate_advanced_statistics,
                calculate_mode_statistics,
                perform_anova,
                perform_tukey_hsd
            )
            
            # Generate comprehensive statistical report components
            advanced_statistics = calculate_advanced_statistics(self.df)
            
            # Fix: Specify 'Mode' as the group_column parameter
            group_statistics = {"Mode": calculate_mode_statistics(self.df, group_column='Mode')} if 'Mode' in self.df.columns else {}
            
            significance_tests = {}
            if 'Mode' in self.df.columns:
                for col in ['Time', 'Distance']:
                    if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                        anova_result = perform_anova(self.df, 'Mode', col)
                        
                        if anova_result.get("is_significant", False):
                            tukey_result = perform_tukey_hsd(self.df, 'Mode', col)
                            significance_tests[col] = tukey_result
                        else:
                            significance_tests[col] = {
                                "anova_result": anova_result,
                                "message": "No significant differences found between modes"
                            }
            
            # Prepare report
            statistical_report = {
                "advanced_statistics": advanced_statistics,
                "group_statistics": group_statistics,
                "significance_tests": significance_tests
            }
            
            # Save report - Make all values JSON serializable
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            json_serializable_report = UtilsManager.make_json_serializable(statistical_report)
            with open(report_path, 'w') as f:
                json.dump(json_serializable_report, f, indent=2)
            print(f"[ADVANCED ANALYSIS] Statistical report saved to {report_path}")
            
            return statistical_report
        
        except Exception as e:
            print(f"[ADVANCED ANALYSIS ERROR] Error in statistical analysis: {e}")
            print(traceback.format_exc())
            return {"error": str(e)}
    
    def _generate_summary(self, statistical_report: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the statistical report
        
        Args:
            statistical_report: The statistical report to summarize
            
        Returns:
            String containing a formatted summary
        """
        summary = "Advanced Statistical Analysis Complete\n\n"
        
        try:
            # Add advanced statistics summary
            if "advanced_statistics" in statistical_report:
                print(f"[ADVANCED ANALYSIS] Processing advanced statistics for {len(statistical_report['advanced_statistics'])} columns")
                summary += "## Advanced Statistics\n\n"
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
                        is_normal_value = stats['is_normal']
                        # Handle both string and boolean representations
                        if isinstance(is_normal_value, str):
                            is_normal_text = "Normal" if is_normal_value.lower() == "true" else "Not normal"
                        else:
                            is_normal_text = "Normal" if is_normal_value else "Not normal"
                        summary += f"- Normality (Shapiro-Wilk): {is_normal_text}\n"
                    
                    summary += "\n"
            
            # Add significance test results
            if "significance_tests" in statistical_report:
                print(f"[ADVANCED ANALYSIS] Processing significance tests for {len(statistical_report['significance_tests'])} columns")
                summary += "## Significance Tests\n\n"
                for column, test_results in statistical_report["significance_tests"].items():
                    summary += f"### {column}\n"
                    
                    # Add ANOVA results
                    if "anova_result" in test_results:
                        anova = test_results["anova_result"]
                        if "is_significant" in anova:
                            # Handle both string and boolean representations
                            if isinstance(anova["is_significant"], str):
                                is_significant = anova["is_significant"].lower() == "true"
                            else:
                                is_significant = anova["is_significant"]
                            
                            summary += f"- ANOVA: {'Significant differences found' if is_significant else 'No significant differences'}\n"
                            if "f_statistic" in anova and "p_value" in anova:
                                summary += f"  - F-statistic: {anova['f_statistic']:.2f}, p-value: {anova['p_value']:.4f}\n"
                    
                    # Add Tukey HSD results if available
                    if "pairwise_results" in test_results and test_results["pairwise_results"]:
                        print(f"[ADVANCED ANALYSIS] Processing {len(test_results['pairwise_results'])} pairwise comparisons for {column}")
                        summary += "- Tukey HSD Pairwise Comparisons:\n"
                        for pair in test_results["pairwise_results"]:
                            if "group1" in pair and "group2" in pair and "is_significant" in pair:
                                # Handle both string and boolean representations
                                if isinstance(pair["is_significant"], str):
                                    is_sig_pair = pair["is_significant"].lower() == "true"
                                else:
                                    is_sig_pair = pair["is_significant"]
                                    
                                sig_text = "Significant" if is_sig_pair else "Not significant"
                                summary += f"  - {pair['group1']} vs {pair['group2']}: {sig_text}\n"
                                if "mean_difference" in pair:
                                    summary += f"    Mean difference: {pair['mean_difference']:.2f}\n"
                    
                    summary += "\n"
            
            # Add group statistics summary if available
            if "group_statistics" in statistical_report and "Mode" in statistical_report["group_statistics"]:
                print("[ADVANCED ANALYSIS] Processing group statistics by Mode")
                mode_stats = statistical_report["group_statistics"]["Mode"]
                summary += "## Statistics by Mode\n\n"
                
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
            print(f"[ADVANCED ANALYSIS ERROR] Error generating statistical summary: {e}")
            print(traceback.format_exc())
            summary += f"Error generating statistical summary: {e}\n"
            summary += "Full statistical report saved to JSON file.\n"
        
        print(f"[ADVANCED ANALYSIS] Statistical summary generated with length {len(summary)}")
        return summary
    
    async def _generate_plots(self) -> str:
        """
        Generate advanced visualization plots
        
        Returns:
            String containing information about generated plots
        """
        print("[ADVANCED ANALYSIS] Generating advanced visualizations...")
        os.makedirs("plots/advanced", exist_ok=True)
        
        if self.pandas_helper:
            advanced_plot_paths = await self.pandas_helper.generate_advanced_plots(output_dir="plots/advanced")
            
            if advanced_plot_paths:
                print(f"[ADVANCED ANALYSIS] Generated {len(advanced_plot_paths)} advanced plots")
                plot_info = "Advanced visualizations generated:\n"
                for path in advanced_plot_paths:
                    if isinstance(path, str) and not path.startswith("Error"):
                        plot_info += f"- {path}\n"
                        print(f"[ADVANCED ANALYSIS] Generated plot: {path}")
                
                print(f"[ADVANCED ANALYSIS] Advanced visualization summary: {len(plot_info)} characters")
            else:
                print("[ADVANCED ANALYSIS WARNING] No advanced plots were generated or an error occurred")
                plot_info = "No advanced plots were generated."
        else:
            plot_info = "Unable to generate plots: PandasHelper not initialized."
            
        return plot_info
    
    @staticmethod
    def summarize_findings(statistical_report: Dict[str, Any]) -> str:
        """
        Create a concise summary of key statistical findings
        
        Args:
            statistical_report: Dictionary containing statistical analysis results
            
        Returns:
            String containing a formatted summary of key statistical findings
        """
        print("[ADVANCED ANALYSIS] Creating statistical findings summary")
        summary = "Key Statistical Findings:\n\n"
        
        # Extract and summarize basic statistics
        if "advanced_statistics" in statistical_report:
            for column, stats in statistical_report["advanced_statistics"].items():
                summary += f"- {column}: Mean = {stats.get('mean', 'N/A'):.2f}, "
                summary += f"Median = {stats.get('median', 'N/A'):.2f}, "
                summary += f"Std Dev = {stats.get('std', 'N/A'):.2f}\n"
        
        # Extract and summarize significant findings
        if "significance_tests" in statistical_report:
            significant_findings = []
            for column, test_results in statistical_report["significance_tests"].items():
                if "anova_result" in test_results:
                    is_sig = test_results["anova_result"].get("is_significant", False)
                    # Handle both string and boolean representations
                    if isinstance(is_sig, str):
                        is_sig = is_sig.lower() == "true"
                    
                    if is_sig:
                        significant_findings.append(f"Significant differences found in {column} across transport modes")
                        
                        if "pairwise_results" in test_results:
                            sig_pairs = []
                            for pair in test_results["pairwise_results"]:
                                pair_sig = pair.get("is_significant", False)
                                # Handle both string and boolean representations
                                if isinstance(pair_sig, str):
                                    pair_sig = pair_sig.lower() == "true"
                                    
                                if pair_sig:
                                    sig_pairs.append(f"{pair.get('group1', '')} vs {pair.get('group2', '')}")
                            
                            if sig_pairs:
                                summary += f"- Significant differences in {column} between: {', '.join(sig_pairs)}\n"
        
        # Add any normality findings
        non_normal_vars = []
        if "advanced_statistics" in statistical_report:
            for column, stats in statistical_report["advanced_statistics"].items():
                if "is_normal" in stats:
                    is_normal = stats["is_normal"]
                    # Handle both string and boolean representations
                    if isinstance(is_normal, str):
                        is_normal = is_normal.lower() == "true"
                        
                    if not is_normal:
                        non_normal_vars.append(column)
            
            if non_normal_vars:
                summary += f"- Non-normally distributed variables: {', '.join(non_normal_vars)}\n"
        
        print(f"[ADVANCED ANALYSIS] Statistical findings summary created with length {len(summary)}")
        return summary


class RegressionModeler:
    """Class for implementing and comparing regression models"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the regression modeler
        
        Args:
            df: The DataFrame to use for modeling
        """
        self.df = df
    
    def fit_polynomial_regression(self, 
                                 target_column: str = 'Time', 
                                 predictor_column: str = 'Distance',
                                 degrees: List[int] = [2, 3, 4],
                                 test_size: float = 0.2) -> Dict[str, Any]:
        """
        Implement and evaluate polynomial regression models with different degrees
        
        Args:
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
            degrees: List of polynomial degrees to test
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with results for all polynomial models
        """
        print(f"[ADVANCED MODELS] Starting polynomial regression analysis with degrees {degrees}")
        
        # Check if columns exist
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{predictor_column}' not found in DataFrame")
        
        # Prepare data
        data = self.df.dropna(subset=[target_column, predictor_column]).copy()
        
        # Split data
        X = data[[predictor_column]]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = {}
        
        # Linear model (degree=1) as baseline
        try:
            # Fit linear model using statsmodels for detailed statistics
            X_train_sm = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_sm).fit()
            
            # Make predictions
            X_test_sm = sm.add_constant(X_test)
            y_pred = model.predict(X_test_sm)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Add model information
            results[1] = {
                "degree": 1,
                "formula": f"{target_column} = {model.params[0]:.4f} + {model.params[1]:.4f}*{predictor_column}",
                "coefficients": model.params.tolist(),
                "metrics": {
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "aic": model.aic,
                    "bic": model.bic
                },
                "model_type": "polynomial",
                "statsmodels_summary": model.summary().as_text()
            }
            
            print(f"[ADVANCED MODELS] Linear model (degree=1): R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {model.aic:.4f}")
        
        except Exception as e:
            print(f"[ADVANCED MODELS] Error fitting linear model: {str(e)}")
            results[1] = {"error": str(e), "degree": 1}
        
        # Fit polynomial models for each degree
        for degree in degrees:
            try:
                # Create polynomial features
                poly = PolynomialFeatures(degree=degree)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)
                
                # Fit linear regression on polynomial features
                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_poly_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Create formula string
                feature_names = poly.get_feature_names_out(input_features=[predictor_column])
                formula = f"{target_column} = {model.intercept_:.4f}"
                for i, coef in enumerate(model.coef_[1:], 1):  # Skip intercept term
                    formula += f" + {coef:.4f}*{feature_names[i]}"
                
                # Use statsmodels for AIC/BIC
                X_poly_sm = sm.add_constant(X_poly_train[:, 1:])  # Skip the first column (already ones)
                sm_model = sm.OLS(y_train, X_poly_sm).fit()
                
                results[degree] = {
                    "degree": degree,
                    "formula": formula,
                    "coefficients": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                    "metrics": {
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "aic": sm_model.aic,
                        "bic": sm_model.bic
                    },
                    "model_type": "polynomial",
                    "feature_names": feature_names.tolist()
                }
                
                print(f"[ADVANCED MODELS] Polynomial model (degree={degree}): R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {sm_model.aic:.4f}")
            
            except Exception as e:
                print(f"[ADVANCED MODELS] Error fitting polynomial model with degree {degree}: {str(e)}")
                results[degree] = {"error": str(e), "degree": degree}
        
        # Find best model based on AIC
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_degree = min(valid_models.items(), 
                              key=lambda x: x[1]['metrics']['aic'] if 'metrics' in x[1] else float('inf'))[0]
            results["best_model"] = {
                "degree": best_degree,
                "selection_criterion": "AIC",
                "metrics": results[best_degree].get("metrics", {})
            }
            print(f"[ADVANCED MODELS] Best polynomial model based on AIC: degree={best_degree}")
        
        return results
    
    def fit_log_transformation_models(self,
                                    target_column: str = 'Time', 
                                    predictor_column: str = 'Distance',
                                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Implement and evaluate log-transformed regression models
        
        Args:
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with results for all log-transformed models
        """
        print(f"[ADVANCED MODELS] Starting log transformation model analysis")
        
        # Check if columns exist
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{predictor_column}' not found in DataFrame")
        
        # Prepare data
        data = self.df.dropna(subset=[target_column, predictor_column]).copy()
        
        # Ensure values are positive for log transformation
        if (data[target_column] <= 0).any() or (data[predictor_column] <= 0).any():
            print("[ADVANCED MODELS] Warning: Some values are ≤ 0, adding small constant for log transformation")
            # Add small constant to non-positive values
            min_target = data[data[target_column] > 0][target_column].min()
            min_predictor = data[data[predictor_column] > 0][predictor_column].min()
            
            # Add 1% of minimum value or 0.01, whichever is larger
            target_offset = max(0.01, min_target * 0.01) if not pd.isna(min_target) else 0.01
            predictor_offset = max(0.01, min_predictor * 0.01) if not pd.isna(min_predictor) else 0.01
            
            # Apply offsets to ensure all values are positive
            data.loc[data[target_column] <= 0, target_column] = target_offset
            data.loc[data[predictor_column] <= 0, predictor_column] = predictor_offset
        
        # Create log-transformed variables
        data['log_target'] = np.log(data[target_column])
        data['log_predictor'] = np.log(data[predictor_column])
        
        # Define transformation types to test
        transformations = [
            {'name': 'linear', 'x': predictor_column, 'y': target_column, 'formula': 'y ~ x'},
            {'name': 'log_x', 'x': 'log_predictor', 'y': target_column, 'formula': 'y ~ log(x)'},
            {'name': 'log_y', 'x': predictor_column, 'y': 'log_target', 'formula': 'log(y) ~ x'},
            {'name': 'log_log', 'x': 'log_predictor', 'y': 'log_target', 'formula': 'log(y) ~ log(x)'}
        ]
        
        results = {}
        
        for transform in transformations:
            try:
                # Split data
                X = data[[transform['x']]].values
                y = data[transform['y']].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Fit linear model using statsmodels
                X_train_sm = sm.add_constant(X_train)
                model = sm.OLS(y_train, X_train_sm).fit()
                
                # Make predictions
                X_test_sm = sm.add_constant(X_test)
                y_pred = model.predict(X_test_sm)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Create actual formula with coefficients
                if transform['name'] == 'linear':
                    formula = f"{target_column} = {model.params[0]:.4f} + {model.params[1]:.4f}*{predictor_column}"
                elif transform['name'] == 'log_x':
                    formula = f"{target_column} = {model.params[0]:.4f} + {model.params[1]:.4f}*log({predictor_column})"
                elif transform['name'] == 'log_y':
                    formula = f"log({target_column}) = {model.params[0]:.4f} + {model.params[1]:.4f}*{predictor_column}"
                    formula += f"\n{target_column} = exp({model.params[0]:.4f} + {model.params[1]:.4f}*{predictor_column})"
                else:  # log_log
                    formula = f"log({target_column}) = {model.params[0]:.4f} + {model.params[1]:.4f}*log({predictor_column})"
                    formula += f"\n{target_column} = exp({model.params[0]:.4f})*{predictor_column}^{model.params[1]:.4f}"
                
                results[transform['name']] = {
                    "transformation": transform['name'],
                    "formula_type": transform['formula'],
                    "detailed_formula": formula,
                    "coefficients": model.params.tolist(),
                    "metrics": {
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "aic": model.aic,
                        "bic": model.bic
                    },
                    "model_type": "log_transformation"
                }
                
                print(f"[ADVANCED MODELS] {transform['name']} model: R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {model.aic:.4f}")
            
            except Exception as e:
                print(f"[ADVANCED MODELS] Error fitting {transform['name']} model: {str(e)}")
                results[transform['name']] = {"error": str(e), "transformation": transform['name']}
        
        # Find best model based on AIC
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_transform = min(valid_models.items(), 
                                key=lambda x: x[1]['metrics']['aic'] if 'metrics' in x[1] else float('inf'))[0]
            results["best_model"] = {
                "transformation": best_transform,
                "selection_criterion": "AIC",
                "metrics": results[best_transform].get("metrics", {})
            }
            print(f"[ADVANCED MODELS] Best log transformation model based on AIC: {best_transform}")
        
        return results
    
    def compare_models(self,
                      polynomial_results: Dict[str, Any],
                      log_results: Dict[str, Any],
                      target_column: str = 'Time', 
                      predictor_column: str = 'Distance',
                      output_dir: str = "plots/models") -> Dict[str, Any]:
        """
        Compare alternative regression models and generate comparison visualizations
        
        Args:
            polynomial_results: Results from polynomial regression
            log_results: Results from log transformation models
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
            output_dir: Directory to save comparison plots
            
        Returns:
            Dictionary with comparison results and plot paths
        """
        print("[ADVANCED MODELS] Comparing alternative regression models")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics for each model type
        linear_metrics = polynomial_results.get(1, {}).get('metrics', {})
        
        # Get best polynomial model metrics
        best_poly_degree = polynomial_results.get('best_model', {}).get('degree')
        if best_poly_degree and best_poly_degree in polynomial_results:
            best_poly_metrics = polynomial_results[best_poly_degree].get('metrics', {})
            poly_formula = polynomial_results[best_poly_degree].get('formula', 'Formula not available')
        else:
            best_poly_metrics = {}
            poly_formula = 'Not available'
        
        # Get best log transformation model metrics
        best_log_transform = log_results.get('best_model', {}).get('transformation')
        if best_log_transform and best_log_transform in log_results:
            best_log_metrics = log_results[best_log_transform].get('metrics', {})
            log_formula = log_results[best_log_transform].get('detailed_formula', 'Formula not available')
        else:
            best_log_metrics = {}
            log_formula = 'Not available'
        
        # Compile all metrics for comparison
        comparison = {
            "linear_model": {
                "name": "Linear Regression",
                "metrics": linear_metrics,
                "formula": f"{target_column} = {polynomial_results.get(1, {}).get('coefficients', [0, 0])[0]:.4f} + {polynomial_results.get(1, {}).get('coefficients', [0, 0])[1]:.4f}*{predictor_column}" if 1 in polynomial_results else "Not available"
            },
            "best_polynomial": {
                "name": f"Polynomial (degree={best_poly_degree})" if best_poly_degree else "Not available",
                "metrics": best_poly_metrics,
                "formula": poly_formula,
                "degree": best_poly_degree
            },
            "best_log_transform": {
                "name": f"Log Transformation ({best_log_transform})" if best_log_transform else "Not available",
                "metrics": best_log_metrics,
                "formula": log_formula,
                "transformation": best_log_transform
            }
        }
        
        # Determine overall best model based on AIC
        valid_models = []
        
        if linear_metrics and 'aic' in linear_metrics:
            valid_models.append(('linear_model', linear_metrics['aic']))
        
        if best_poly_metrics and 'aic' in best_poly_metrics:
            valid_models.append(('best_polynomial', best_poly_metrics['aic']))
        
        if best_log_metrics and 'aic' in best_log_metrics:
            valid_models.append(('best_log_transform', best_log_metrics['aic']))
        
        if valid_models:
            best_model_key = min(valid_models, key=lambda x: x[1])[0]
            comparison['overall_best_model'] = best_model_key
            comparison['selection_criterion'] = 'AIC'
            print(f"[ADVANCED MODELS] Overall best model based on AIC: {best_model_key}")
        
        # Generate model visualizations
        plot_paths = self._generate_model_visualizations(
            polynomial_results, log_results, 
            target_column, predictor_column, 
            best_poly_degree, best_log_transform, 
            comparison, output_dir
        )
        
        comparison['plot_paths'] = plot_paths
        return comparison
    
    def _generate_model_visualizations(self,
                                     polynomial_results: Dict[str, Any],
                                     log_results: Dict[str, Any],
                                     target_column: str,
                                     predictor_column: str,
                                     best_poly_degree: int,
                                     best_log_transform: str,
                                     comparison: Dict[str, Any],
                                     output_dir: str) -> List[str]:
        """
        Generate visualizations for model comparison
        
        Args:
            polynomial_results: Results from polynomial regression
            log_results: Results from log transformation models
            target_column: Target variable name
            predictor_column: Predictor variable name
            best_poly_degree: Best polynomial degree
            best_log_transform: Best log transformation
            comparison: Model comparison results
            output_dir: Directory to save plots
            
        Returns:
            List of plot file paths
        """
        plot_paths = []
        
        try:
            # Model comparison plot
            plt.figure(figsize=(12, 8))
            
            # Scatter plot of actual data
            sns.scatterplot(data=self.df, x=predictor_column, y=target_column, alpha=0.4, label='Actual Data')
            
            # Create a range of x values for prediction
            x_min = self.df[predictor_column].min()
            x_max = self.df[predictor_column].max()
            x_range = np.linspace(x_min, x_max, 100)
            
            # Plot linear model
            if 1 in polynomial_results and 'coefficients' in polynomial_results[1]:
                coef = polynomial_results[1]['coefficients']
                y_linear = coef[0] + coef[1] * x_range
                plt.plot(x_range, y_linear, label='Linear', color='red', linewidth=2)
            
            # Plot best polynomial model
            if best_poly_degree and best_poly_degree in polynomial_results:
                poly = PolynomialFeatures(degree=best_poly_degree)
                x_poly = poly.fit_transform(x_range.reshape(-1, 1))
                
                if 'coefficients' in polynomial_results[best_poly_degree] and 'intercept' in polynomial_results[best_poly_degree]:
                    coef = polynomial_results[best_poly_degree]['coefficients']
                    intercept = polynomial_results[best_poly_degree]['intercept']
                    
                    # For sklearn models, intercept is separate
                    y_poly = intercept
                    for i, c in enumerate(coef):
                        y_poly += c * x_poly[:, i]
                    
                    plt.plot(x_range, y_poly, label=f'Polynomial (degree={best_poly_degree})', 
                            color='green', linewidth=2)
            
            # Plot best log transformation model
            if best_log_transform and best_log_transform in log_results and 'coefficients' in log_results[best_log_transform]:
                coef = log_results[best_log_transform]['coefficients']
                
                if best_log_transform == 'linear':
                    y_log = coef[0] + coef[1] * x_range
                elif best_log_transform == 'log_x':
                    y_log = coef[0] + coef[1] * np.log(x_range)
                elif best_log_transform == 'log_y':
                    y_log = np.exp(coef[0] + coef[1] * x_range)
                else:  # log_log
                    y_log = np.exp(coef[0]) * x_range ** coef[1]
                
                plt.plot(x_range, y_log, label=f'Log Transform ({best_log_transform})', 
                        color='blue', linewidth=2)
            
            plt.title('Comparison of Alternative Regression Models')
            plt.xlabel(predictor_column)
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save comparison plot
            comparison_plot_path = os.path.join(output_dir, "model_comparison.png")
            plt.savefig(comparison_plot_path)
            plt.close()
            plot_paths.append(comparison_plot_path)
            print(f"[ADVANCED MODELS] Saved model comparison plot: {comparison_plot_path}")
            
            # Generate AIC/BIC comparison bar chart
            self._generate_criterion_barchart(comparison, best_poly_degree, best_log_transform, output_dir, plot_paths)
            
        except Exception as e:
            print(f"[ADVANCED MODELS] Error generating comparison plots: {str(e)}")
            traceback.print_exc()
        
        return plot_paths
    
    def _generate_criterion_barchart(self,
                                   comparison: Dict[str, Any],
                                   best_poly_degree: int,
                                   best_log_transform: str,
                                   output_dir: str,
                                   plot_paths: List[str]) -> None:
        """
        Generate bar chart comparing AIC/BIC criteria
        
        Args:
            comparison: Model comparison results
            best_poly_degree: Best polynomial degree
            best_log_transform: Best log transformation
            output_dir: Directory to save plots
            plot_paths: List to append plot path to
        """
        valid_models_for_plot = []
        model_names = []
        aic_values = []
        bic_values = []
        
        if 'metrics' in comparison['linear_model'] and 'aic' in comparison['linear_model']['metrics']:
            valid_models_for_plot.append('linear_model')
            model_names.append('Linear')
            aic_values.append(comparison['linear_model']['metrics']['aic'])
            bic_values.append(comparison['linear_model']['metrics']['bic'])
        
        if 'metrics' in comparison['best_polynomial'] and 'aic' in comparison['best_polynomial']['metrics']:
            valid_models_for_plot.append('best_polynomial')
            model_names.append(f'Poly (d={best_poly_degree})')
            aic_values.append(comparison['best_polynomial']['metrics']['aic'])
            bic_values.append(comparison['best_polynomial']['metrics']['bic'])
        
        if 'metrics' in comparison['best_log_transform'] and 'aic' in comparison['best_log_transform']['metrics']:
            valid_models_for_plot.append('best_log_transform')
            model_names.append(f'Log ({best_log_transform})')
            aic_values.append(comparison['best_log_transform']['metrics']['aic'])
            bic_values.append(comparison['best_log_transform']['metrics']['bic'])
        
        if valid_models_for_plot:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(model_names))
            width = 0.35
            
            plt.bar(x - width/2, aic_values, width, label='AIC')
            plt.bar(x + width/2, bic_values, width, label='BIC')
            
            plt.xlabel('Model Type')
            plt.ylabel('Criterion Value (Lower is Better)')
            plt.title('Model Selection Criteria Comparison')
            plt.xticks(x, model_names)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            for i, v in enumerate(aic_values):
                plt.text(i - width/2, v + 5, f'{v:.1f}', ha='center')
            
            for i, v in enumerate(bic_values):
                plt.text(i + width/2, v + 5, f'{v:.1f}', ha='center')
            
            # Save AIC/BIC comparison plot
            criterion_plot_path = os.path.join(output_dir, "model_selection_criteria.png")
            plt.savefig(criterion_plot_path)
            plt.close()
            plot_paths.append(criterion_plot_path)
            print(f"[ADVANCED MODELS] Saved model selection criteria plot: {criterion_plot_path}")


class ModelingManager:
    """Manager class for orchestrating model creation, comparison, and evaluation"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the modeling manager
        
        Args:
            df: The DataFrame to use for modeling
        """
        self.df = df
        self.modeler = RegressionModeler(df)
    
    def perform_advanced_modeling(self,
                                target_column: str = 'Time', 
                                predictor_column: str = 'Distance',
                                save_report: bool = True,
                                report_path: str = "reports/advanced_models.json",
                                generate_plots: bool = True,
                                plots_dir: str = "plots/models") -> Dict[str, Any]:
        """
        Implement and evaluate alternative regression models including polynomial and log transformations
        
        Args:
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
            save_report: Whether to save the report to a file
            report_path: Path to save the report
            generate_plots: Whether to generate model visualization plots
            plots_dir: Directory to save plots in
            
        Returns:
            Dictionary with advanced modeling results
        """
        print(f"[ADVANCED MODELS] Starting advanced modeling analysis: {target_column} ~ {predictor_column}")
        
        try:
            # 1. Run polynomial regression with degrees 2-4
            polynomial_results = self.modeler.fit_polynomial_regression(
                target_column=target_column,
                predictor_column=predictor_column,
                degrees=[2, 3, 4]
            )
            
            # 2. Run log transformation models
            log_results = self.modeler.fit_log_transformation_models(
                target_column=target_column,
                predictor_column=predictor_column
            )
            
            # 3. Compare models and generate visualizations
            comparison = self.modeler.compare_models(
                polynomial_results=polynomial_results,
                log_results=log_results,
                target_column=target_column,
                predictor_column=predictor_column,
                output_dir=plots_dir
            ) if generate_plots else self._get_basic_comparison(polynomial_results, log_results)
            
            # 4. Compile overall results
            overall_results = {
                "polynomial_models": polynomial_results,
                "log_transformation_models": log_results,
                "model_comparison": comparison,
                "status": "success"
            }
            
            # 5. Save results if requested
            if save_report:
                self._save_modeling_report(overall_results, report_path)
                overall_results['report_path'] = report_path
            
            print(f"[ADVANCED MODELS] Advanced modeling analysis completed successfully")
            return overall_results
        
        except Exception as e:
            print(f"[ADVANCED MODELS ERROR] Error in advanced modeling: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _get_basic_comparison(self, polynomial_results: Dict[str, Any], log_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a basic comparison of models without generating plots
        
        Args:
            polynomial_results: Results from polynomial regression
            log_results: Results from log transformation models
            
        Returns:
            Dictionary with basic comparison information
        """
        return {
            "linear_model": {
                "name": "Linear Regression",
                "metrics": polynomial_results.get(1, {}).get('metrics', {})
            },
            "best_polynomial": {
                "name": f"Polynomial (degree={polynomial_results.get('best_model', {}).get('degree')})",
                "metrics": polynomial_results.get(polynomial_results.get('best_model', {}).get('degree', 1), {}).get('metrics', {})
            },
            "best_log_transform": {
                "name": f"Log Transformation ({log_results.get('best_model', {}).get('transformation')})",
                "metrics": log_results.get(log_results.get('best_model', {}).get('transformation', ''), {}).get('metrics', {})
            }
        }
    
    def _save_modeling_report(self, results: Dict[str, Any], report_path: str) -> None:
        """
        Save modeling results to a JSON file
        
        Args:
            results: The modeling results to save
            report_path: Path to save the report to
        """
        # Make values JSON serializable
        serializable_results = UtilsManager.make_json_serializable(results)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Write to file
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[ADVANCED MODELS] Advanced modeling results saved to {report_path}")


# Function interfaces to maintain backward compatibility

# Utility function for JSON serialization
def make_json_serializable(obj):
    """Backward compatibility function for make_json_serializable"""
    return UtilsManager.make_json_serializable(obj)

# Statistical analysis functions
async def perform_advanced_analysis(df: pd.DataFrame, llm, original_path: str, modification_summary: str = None) -> Dict[str, Any]:
    """Backward compatibility function for perform_advanced_analysis"""
    analyzer = StatisticalAnalyzer(df, llm)
    return await analyzer.perform_analysis(original_path, modification_summary)

async def summarize_statistical_findings(statistical_report: Dict[str, Any]) -> str:
    """Backward compatibility function for summarize_statistical_findings"""
    return StatisticalAnalyzer.summarize_findings(statistical_report)

# Regression modeling functions
def fit_polynomial_regression(df: pd.DataFrame, target_column: str = 'Time', predictor_column: str = 'Distance',
                           degrees: List[int] = [2, 3, 4], test_size: float = 0.2) -> Dict[str, Any]:
    """Backward compatibility function for fit_polynomial_regression"""
    modeler = RegressionModeler(df)
    return modeler.fit_polynomial_regression(target_column, predictor_column, degrees, test_size)

def fit_log_transformation_models(df: pd.DataFrame, target_column: str = 'Time', predictor_column: str = 'Distance',
                                test_size: float = 0.2) -> Dict[str, Any]:
    """Backward compatibility function for fit_log_transformation_models"""
    modeler = RegressionModeler(df)
    return modeler.fit_log_transformation_models(target_column, predictor_column, test_size)

def compare_alternative_models(df: pd.DataFrame, polynomial_results: Dict[str, Any], log_results: Dict[str, Any],
                            target_column: str = 'Time', predictor_column: str = 'Distance',
                            output_dir: str = "plots/models") -> Dict[str, Any]:
    """Backward compatibility function for compare_alternative_models"""
    modeler = RegressionModeler(df)
    return modeler.compare_models(polynomial_results, log_results, target_column, predictor_column, output_dir)

def perform_advanced_modeling(df: pd.DataFrame, target_column: str = 'Time', predictor_column: str = 'Distance',
                           save_report: bool = True, report_path: str = "reports/advanced_models.json",
                           generate_plots: bool = True, plots_dir: str = "plots/models") -> Dict[str, Any]:
    """Backward compatibility function for perform_advanced_modeling"""
    manager = ModelingManager(df)
    return manager.perform_advanced_modeling(target_column, predictor_column, save_report, report_path, generate_plots, plots_dir)