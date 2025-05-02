import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class RegressionModel:
    """
    A class to implement and evaluate linear regression models for the commute time data.
    This class handles both full dataset models (Time ~ Distance) and mode-specific models.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'Time', predictor_column: str = 'Distance'):
        """
        Initialize the RegressionModel class.
        
        Args:
            df: The DataFrame containing the data
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.predictor_column = predictor_column
        self.models = {}
        self.model_results = {}
        self.sklearn_models = {}
        
        # Check if the necessary columns exist
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{predictor_column}' not found in DataFrame")
    
    def fit_full_dataset_model(self) -> Dict[str, Any]:
        """
        Fit a linear regression model on the full dataset.
        
        Returns:
            Dictionary with regression metrics and model information
        """
        print(f"[REGRESSION] Fitting full dataset model: {self.target_column} ~ {self.predictor_column}")
        
        # Drop rows with missing values in target or predictor
        clean_df = self.df.dropna(subset=[self.target_column, self.predictor_column])
        
        # Extract X and y
        X = clean_df[[self.predictor_column]]
        y = clean_df[self.target_column]
        
        # Add constant for statsmodels
        X_sm = sm.add_constant(X)
        
        # Fit statsmodels OLS for detailed statistics
        model = sm.OLS(y, X_sm).fit()
        
        # Store the model
        self.models['full_dataset'] = model
        
        # Fit sklearn model for prediction
        skl_model = LinearRegression()
        skl_model.fit(X, y)
        self.sklearn_models['full_dataset'] = skl_model
        
        # Calculate metrics
        r_squared = model.rsquared
        adjusted_r_squared = model.rsquared_adj
        rmse = np.sqrt(model.mse_resid)
        mae = np.mean(np.abs(model.resid))
        
        # Calculate correlation coefficient
        correlation = clean_df[[self.target_column, self.predictor_column]].corr().iloc[0, 1]
        
        # Extract model coefficients and p-values
        coef = model.params[1]
        intercept = model.params[0]
        p_value = model.pvalues[1]
        
        # Get confidence intervals for coefficients
        conf_int = model.conf_int(alpha=0.05)
        coef_ci_low, coef_ci_high = conf_int.iloc[1, 0], conf_int.iloc[1, 1]
        intercept_ci_low, intercept_ci_high = conf_int.iloc[0, 0], conf_int.iloc[0, 1]
        
        # Check statistical significance
        is_significant = p_value < 0.05
        
        results = {
            'model_type': 'full_dataset',
            'metrics': {
                'r': correlation,
                'r_squared': r_squared,
                'adjusted_r_squared': adjusted_r_squared,
                'rmse': rmse,
                'mae': mae
            },
            'coefficients': {
                'intercept': intercept,
                'coefficient': coef,
                'p_value': p_value,
                'is_significant': is_significant
            },
            'confidence_intervals': {
                'intercept_ci': [intercept_ci_low, intercept_ci_high],
                'coefficient_ci': [coef_ci_low, coef_ci_high]
            },
            'formula': f"{self.target_column} = {intercept:.2f} + {coef:.2f} * {self.predictor_column}"
        }
        
        # Store results
        self.model_results['full_dataset'] = results
        
        print(f"[REGRESSION] Full dataset model fitted with R² = {r_squared:.4f}")
        return results
    
    def fit_mode_specific_models(self, mode_column: str = 'Mode') -> Dict[str, Dict[str, Any]]:
        """
        Fit separate linear regression models for each transport mode.
        
        Args:
            mode_column: The column containing transport mode information
            
        Returns:
            Dictionary with model results for each mode
        """
        if mode_column not in self.df.columns:
            raise ValueError(f"Mode column '{mode_column}' not found in DataFrame")
        
        print(f"[REGRESSION] Fitting mode-specific models for each value in '{mode_column}'")
        
        all_mode_results = {}
        
        # For each unique mode value
        for mode in self.df[mode_column].unique():
            # Filter data for this mode
            mode_df = self.df[self.df[mode_column] == mode]
            
            # Only proceed if we have enough data points (at least 3 for regression)
            if len(mode_df) < 3:
                print(f"[REGRESSION] Not enough data points for mode '{mode}' (count: {len(mode_df)})")
                continue
            
            # Drop rows with missing values in target or predictor
            clean_mode_df = mode_df.dropna(subset=[self.target_column, self.predictor_column])
            
            if len(clean_mode_df) < 3:
                print(f"[REGRESSION] Not enough clean data points for mode '{mode}' after dropping NA values")
                continue
            
            print(f"[REGRESSION] Fitting model for mode '{mode}' with {len(clean_mode_df)} data points")
            
            # Extract X and y
            X = clean_mode_df[[self.predictor_column]]
            y = clean_mode_df[self.target_column]
            
            # Add constant for statsmodels
            X_sm = sm.add_constant(X)
            
            try:
                # Fit statsmodels OLS
                model = sm.OLS(y, X_sm).fit()
                
                # Store the model
                model_key = f"mode_{mode}"
                self.models[model_key] = model
                
                # Fit sklearn model for prediction
                skl_model = LinearRegression()
                skl_model.fit(X, y)
                self.sklearn_models[model_key] = skl_model
                
                # Calculate metrics
                r_squared = model.rsquared
                adjusted_r_squared = model.rsquared_adj
                rmse = np.sqrt(model.mse_resid)
                mae = np.mean(np.abs(model.resid))
                
                # Calculate correlation coefficient
                correlation = clean_mode_df[[self.target_column, self.predictor_column]].corr().iloc[0, 1]
                
                # Extract model coefficients and p-values
                coef = model.params[1]
                intercept = model.params[0]
                p_value = model.pvalues[1]
                
                # Get confidence intervals for coefficients
                conf_int = model.conf_int(alpha=0.05)
                coef_ci_low, coef_ci_high = conf_int.iloc[1, 0], conf_int.iloc[1, 1]
                intercept_ci_low, intercept_ci_high = conf_int.iloc[0, 0], conf_int.iloc[0, 1]
                
                # Check statistical significance
                is_significant = p_value < 0.05
                
                results = {
                    'model_type': f"mode_{mode}",
                    'mode': mode,
                    'sample_size': len(clean_mode_df),
                    'metrics': {
                        'r': correlation,
                        'r_squared': r_squared,
                        'adjusted_r_squared': adjusted_r_squared,
                        'rmse': rmse,
                        'mae': mae
                    },
                    'coefficients': {
                        'intercept': intercept,
                        'coefficient': coef,
                        'p_value': p_value,
                        'is_significant': is_significant
                    },
                    'confidence_intervals': {
                        'intercept_ci': [intercept_ci_low, intercept_ci_high],
                        'coefficient_ci': [coef_ci_low, coef_ci_high]
                    },
                    'formula': f"{self.target_column} = {intercept:.2f} + {coef:.2f} * {self.predictor_column}"
                }
                
                # Store results
                self.model_results[model_key] = results
                all_mode_results[mode] = results
                
                print(f"[REGRESSION] Mode '{mode}' model fitted with R² = {r_squared:.4f}")
            
            except Exception as e:
                print(f"[REGRESSION] Error fitting model for mode '{mode}': {str(e)}")
                all_mode_results[mode] = {"error": str(e), "mode": mode, "sample_size": len(clean_mode_df)}
        
        return all_mode_results
    
    def generate_regression_plots(self, output_dir: str = "plots/regression") -> List[str]:
        """
        Generate regression plots for all fitted models.
        
        Args:
            output_dir: Directory to save plots in
            
        Returns:
            List of saved plot file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Plot for full dataset model
        if 'full_dataset' in self.models:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot of actual data
            sns.scatterplot(data=self.df, x=self.predictor_column, y=self.target_column, alpha=0.6, label='Data points')
            
            # Add regression line
            model = self.models['full_dataset']
            intercept = model.params[0]
            coef = model.params[1]
            
            # Create range of x values for prediction line
            x_range = np.linspace(self.df[self.predictor_column].min(), self.df[self.predictor_column].max(), 100)
            y_pred = intercept + coef * x_range
            
            plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression Line: y = {intercept:.2f} + {coef:.2f}x')
            
            # Add R-squared to plot
            r_squared = self.model_results['full_dataset']['metrics']['r_squared']
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(f'Linear Regression: {self.target_column} vs {self.predictor_column} (Full Dataset)')
            plt.xlabel(self.predictor_column)
            plt.ylabel(self.target_column)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            full_plot_path = os.path.join(output_dir, "full_dataset_regression.png")
            plt.savefig(full_plot_path)
            plt.close()
            plot_paths.append(full_plot_path)
            print(f"[REGRESSION] Saved plot: {full_plot_path}")
        
        # 2. Plot for mode-specific models
        if any(key.startswith('mode_') for key in self.models.keys()):
            # Create a single plot with all mode-specific regression lines
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with points colored by mode
            if 'Mode' in self.df.columns:
                sns.scatterplot(data=self.df, x=self.predictor_column, y=self.target_column, 
                               hue='Mode', alpha=0.6)
            else:
                sns.scatterplot(data=self.df, x=self.predictor_column, y=self.target_column, 
                               alpha=0.6, label='Data points')
            
            # Add regression lines for each mode
            for key, model in self.models.items():
                if key.startswith('mode_'):
                    mode = key.replace('mode_', '')
                    intercept = model.params[0]
                    coef = model.params[1]
                    
                    # Create range of x values for prediction line
                    mode_df = self.df[self.df['Mode'] == mode]
                    if len(mode_df) > 0:
                        x_min = mode_df[self.predictor_column].min()
                        x_max = mode_df[self.predictor_column].max()
                        x_range = np.linspace(x_min, x_max, 100)
                        y_pred = intercept + coef * x_range
                        
                        plt.plot(x_range, y_pred, linewidth=2, label=f'{mode}: y = {intercept:.2f} + {coef:.2f}x')
            
            plt.title(f'Linear Regression by Transport Mode: {self.target_column} vs {self.predictor_column}')
            plt.xlabel(self.predictor_column)
            plt.ylabel(self.target_column)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Transport Mode')
            
            modes_plot_path = os.path.join(output_dir, "mode_specific_regression.png")
            plt.savefig(modes_plot_path)
            plt.close()
            plot_paths.append(modes_plot_path)
            print(f"[REGRESSION] Saved plot: {modes_plot_path}")
            
            # Individual plots for each mode
            for key, model in self.models.items():
                if key.startswith('mode_'):
                    mode = key.replace('mode_', '')
                    mode_df = self.df[self.df['Mode'] == mode]
                    
                    plt.figure(figsize=(10, 6))
                    
                    # Scatter plot for this mode
                    sns.scatterplot(data=mode_df, x=self.predictor_column, y=self.target_column, alpha=0.6, label=f'{mode} data points')
                    
                    # Add regression line
                    intercept = model.params[0]
                    coef = model.params[1]
                    
                    # Create range of x values for prediction line
                    x_min = mode_df[self.predictor_column].min()
                    x_max = mode_df[self.predictor_column].max()
                    x_range = np.linspace(x_min, x_max, 100)
                    y_pred = intercept + coef * x_range
                    
                    plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression: y = {intercept:.2f} + {coef:.2f}x')
                    
                    # Add R-squared to plot
                    r_squared = self.model_results[key]['metrics']['r_squared']
                    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
                             bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.title(f'Linear Regression: {self.target_column} vs {self.predictor_column} (Mode: {mode})')
                    plt.xlabel(self.predictor_column)
                    plt.ylabel(self.target_column)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    mode_plot_path = os.path.join(output_dir, f"{mode.lower()}_regression.png")
                    plt.savefig(mode_plot_path)
                    plt.close()
                    plot_paths.append(mode_plot_path)
                    print(f"[REGRESSION] Saved plot: {mode_plot_path}")
        
        return plot_paths
    
    def predict(self, distance_values: List[float], model_type: str = 'full_dataset') -> Dict[str, Any]:
        """
        Generate predictions using the specified model.
        
        Args:
            distance_values: List of distance values to predict time for
            model_type: Type of model to use ('full_dataset' or 'mode_XXX')
            
        Returns:
            Dictionary with predictions and intervals
        """
        if model_type not in self.sklearn_models:
            raise ValueError(f"Model type '{model_type}' not found in fitted models")
        
        # Create DataFrame from distance values
        X_pred = pd.DataFrame({self.predictor_column: distance_values})
        
        # Get the sklearn model for prediction
        model = self.sklearn_models[model_type]
        statsmodel = self.models[model_type]
        
        # Get point predictions
        y_pred = model.predict(X_pred)
        
        # Calculate prediction intervals using statsmodels
        X_sm = sm.add_constant(X_pred)
        
        # Get prediction statistics from statsmodels
        try:
            pred = statsmodel.get_prediction(X_sm)
            pred_intervals = pred.conf_int(alpha=0.05)  # 95% prediction interval
            
            lower_interval = pred_intervals[:, 0]
            upper_interval = pred_intervals[:, 1]
            
            results = {
                'distance_values': distance_values,
                'predicted_times': y_pred.tolist(),
                'lower_intervals': lower_interval.tolist(),
                'upper_intervals': upper_interval.tolist(),
                'model_type': model_type
            }
            
            return results
        except Exception as e:
            print(f"[REGRESSION] Error calculating prediction intervals: {str(e)}")
            # Return basic predictions without intervals
            return {
                'distance_values': distance_values,
                'predicted_times': y_pred.tolist(),
                'model_type': model_type,
                'error': f"Could not calculate prediction intervals: {str(e)}"
            }
    
    def get_model_summary(self, model_type: str = 'full_dataset') -> str:
        """
        Get a detailed summary of the specified model.
        
        Args:
            model_type: Type of model to summarize ('full_dataset' or 'mode_XXX')
            
        Returns:
            String with model summary
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found in fitted models")
        
        return self.models[model_type].summary().as_text()
    
    def save_model_results(self, file_path: str = "reports/regression_models.json") -> None:
        """
        Save all model results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        print(f"[REGRESSION] Attempting to save model results to {file_path}")
        
        # Make all values JSON serializable
        def make_serializable(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.float64) or isinstance(item, np.float32):
                return float(item)
            elif isinstance(item, np.int64) or isinstance(item, np.int32):
                return int(item)
            elif isinstance(item, bool) or isinstance(item, np.bool_):  # Handle both Python and NumPy boolean types
                return bool(item)  # Convert to Python native boolean
            else:
                # Log unusual types that might cause serialization issues
                if not isinstance(item, (str, int, float, list, dict, type(None))):
                    print(f"[REGRESSION WARNING] Potentially non-serializable type encountered: {type(item)}")
                return item
        
        try:
            # Convert results to serializable format
            serializable_results = {}
            for key, result in self.model_results.items():
                serializable_results[key] = {}
                for k, v in result.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {k2: make_serializable(v2) for k2, v2 in v.items()}
                    else:
                        serializable_results[key][k] = make_serializable(v)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"[REGRESSION] Model results successfully saved to {file_path}")
        except Exception as e:
            print(f"[REGRESSION ERROR] Failed to save model results to {file_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())


def perform_regression_analysis(df: pd.DataFrame, 
                               target_column: str = 'Time', 
                               predictor_column: str = 'Distance',
                               save_report: bool = True,
                               report_path: str = "reports/regression_models.json",
                               generate_plots: bool = True,
                               plots_dir: str = "plots/regression") -> Dict[str, Any]:
    """
    Perform comprehensive regression analysis on the dataset.
    
    Args:
        df: The DataFrame containing the data
        target_column: The target variable (dependent variable)
        predictor_column: The predictor variable (independent variable)
        save_report: Whether to save the report to a file
        report_path: Path to save the report
        generate_plots: Whether to generate regression plots
        plots_dir: Directory to save plots in
        
    Returns:
        Dictionary with regression analysis results
    """
    print(f"[REGRESSION] Starting regression analysis: {target_column} ~ {predictor_column}")
    
    try:
        # Create RegressionModel instance
        regression_model = RegressionModel(df, target_column, predictor_column)
        
        # Fit full dataset model
        full_model_results = regression_model.fit_full_dataset_model()
        
        # Fit mode-specific models if 'Mode' column exists
        mode_results = {}
        if 'Mode' in df.columns:
            mode_results = regression_model.fit_mode_specific_models(mode_column='Mode')
        
        # Generate plots if requested
        plot_paths = []
        if generate_plots:
            plot_paths = regression_model.generate_regression_plots(output_dir=plots_dir)
        
        # Save model results if requested
        if save_report:
            regression_model.save_model_results(file_path=report_path)
        
        # Return comprehensive results
        analysis_results = {
            'full_model': full_model_results,
            'mode_models': mode_results,
            'plot_paths': plot_paths,
            'report_path': report_path if save_report else None
        }
        
        # Add success message
        analysis_results['status'] = 'success'
        
        print(f"[REGRESSION] Regression analysis completed successfully")
        return analysis_results
    
    except Exception as e:
        import traceback
        print(f"[REGRESSION ERROR] Error in regression analysis: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }