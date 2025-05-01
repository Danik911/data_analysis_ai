import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from regression_analysis import RegressionModel, perform_regression_analysis

class PredictionGenerator:
    """
    A class to generate and visualize regression-based predictions for commute times.
    """
    
    def __init__(self, regression_model: RegressionModel):
        """
        Initialize the PredictionGenerator class.
        
        Args:
            regression_model: A fitted RegressionModel instance
        """
        self.regression_model = regression_model
        self.predictions = {}
        
    def generate_predictions(self, distance_values: List[float] = None, 
                           num_points: int = 8, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate predictions for specific distance values using multiple models.
        
        Args:
            distance_values: List of distance values to predict time for. If None, will generate evenly spaced values.
            num_points: Number of prediction points to generate if distance_values is None
            model_types: List of model types to use for prediction ('full_dataset' or 'mode_XXX')
            
        Returns:
            Dictionary with predictions for each model
        """
        # If no distance values provided, generate evenly spaced values within the data range
        if distance_values is None:
            distance_min = self.regression_model.df[self.regression_model.predictor_column].min()
            distance_max = self.regression_model.df[self.regression_model.predictor_column].max()
            distance_values = np.linspace(distance_min, distance_max, num_points).tolist()
        
        # If no model types provided, use all available models
        if model_types is None:
            model_types = list(self.regression_model.models.keys())
        
        print(f"[PREDICTION] Generating predictions for {len(distance_values)} distance values using {len(model_types)} models")
        
        all_predictions = {}
        
        for model_type in model_types:
            if model_type in self.regression_model.models:
                try:
                    # Get predictions for this model
                    model_predictions = self.regression_model.predict(distance_values, model_type=model_type)
                    all_predictions[model_type] = model_predictions
                    print(f"[PREDICTION] Generated predictions for model: {model_type}")
                except Exception as e:
                    print(f"[PREDICTION] Error generating predictions for model {model_type}: {str(e)}")
            else:
                print(f"[PREDICTION] Model {model_type} not found in fitted models")
        
        # Store predictions
        self.predictions = all_predictions
        
        return all_predictions
    
    def generate_prediction_plots(self, output_dir: str = "plots/predictions") -> List[str]:
        """
        Generate plots visualizing the predictions and prediction intervals.
        
        Args:
            output_dir: Directory to save plots in
            
        Returns:
            List of saved plot file paths
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run generate_predictions first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Combined plot with predictions from all models
        plt.figure(figsize=(12, 8))
        
        # Get all unique distance values across all predictions
        all_distances = []
        for model_type, pred_data in self.predictions.items():
            all_distances.extend(pred_data.get('distance_values', []))
        all_distances = sorted(set(all_distances))
        
        # Plot the actual data points
        sns.scatterplot(
            data=self.regression_model.df, 
            x=self.regression_model.predictor_column, 
            y=self.regression_model.target_column, 
            alpha=0.4, 
            label=f'Actual {self.regression_model.target_column}'
        )
        
        # Plot predictions for each model
        for model_type, pred_data in self.predictions.items():
            distances = pred_data.get('distance_values', [])
            times = pred_data.get('predicted_times', [])
            
            if 'lower_intervals' in pred_data and 'upper_intervals' in pred_data:
                lower = pred_data.get('lower_intervals', [])
                upper = pred_data.get('upper_intervals', [])
                
                # Plot prediction intervals as shaded areas
                plt.fill_between(
                    distances, lower, upper, 
                    alpha=0.2, 
                    label=f'{model_type} 95% Interval'
                )
            
            # Plot predicted values
            plt.plot(
                distances, times, 
                marker='o', 
                linestyle='-', 
                linewidth=2,
                label=f'{model_type} Predicted'
            )
        
        plt.title(f'Predicted {self.regression_model.target_column} vs {self.regression_model.predictor_column} by Model')
        plt.xlabel(self.regression_model.predictor_column)
        plt.ylabel(f'Predicted {self.regression_model.target_column}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        combined_plot_path = os.path.join(output_dir, "combined_predictions.png")
        plt.savefig(combined_plot_path)
        plt.close()
        plot_paths.append(combined_plot_path)
        print(f"[PREDICTION] Saved plot: {combined_plot_path}")
        
        # 2. Individual plots for each model
        for model_type, pred_data in self.predictions.items():
            plt.figure(figsize=(10, 6))
            
            # Plot actual data
            if model_type.startswith('mode_'):
                # For mode-specific models, only show data for that mode
                mode = model_type.replace('mode_', '')
                mode_df = self.regression_model.df[self.regression_model.df['Mode'] == mode]
                sns.scatterplot(
                    data=mode_df, 
                    x=self.regression_model.predictor_column, 
                    y=self.regression_model.target_column, 
                    alpha=0.4, 
                    label=f'Actual {self.regression_model.target_column} ({mode})'
                )
            else:
                # For full dataset model, show all data
                sns.scatterplot(
                    data=self.regression_model.df, 
                    x=self.regression_model.predictor_column, 
                    y=self.regression_model.target_column, 
                    alpha=0.4, 
                    label=f'Actual {self.regression_model.target_column}'
                )
            
            distances = pred_data.get('distance_values', [])
            times = pred_data.get('predicted_times', [])
            
            # Plot prediction intervals if available
            if 'lower_intervals' in pred_data and 'upper_intervals' in pred_data:
                lower = pred_data.get('lower_intervals', [])
                upper = pred_data.get('upper_intervals', [])
                
                plt.fill_between(
                    distances, lower, upper, 
                    alpha=0.2, 
                    label='95% Prediction Interval'
                )
            
            # Plot predicted values
            plt.plot(
                distances, times, 
                marker='o', 
                linestyle='-', 
                linewidth=2,
                label='Predicted Values'
            )
            
            plt.title(f'Prediction: {self.regression_model.target_column} vs {self.regression_model.predictor_column} ({model_type})')
            plt.xlabel(self.regression_model.predictor_column)
            plt.ylabel(f'Predicted {self.regression_model.target_column}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            model_plot_path = os.path.join(output_dir, f"{model_type.lower().replace(' ', '_')}_predictions.png")
            plt.savefig(model_plot_path)
            plt.close()
            plot_paths.append(model_plot_path)
            print(f"[PREDICTION] Saved plot: {model_plot_path}")
        
        return plot_paths
    
    def prepare_prediction_table(self) -> pd.DataFrame:
        """
        Create a formatted DataFrame comparing predictions from different models.
        
        Returns:
            DataFrame with organized prediction comparisons
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run generate_predictions first.")
        
        # Get all unique distance values and sort them
        all_distances = set()
        for model_type, pred_data in self.predictions.items():
            all_distances.update(pred_data.get('distance_values', []))
        all_distances = sorted(all_distances)
        
        # Create a DataFrame with distance values
        comparison_df = pd.DataFrame({'Distance': list(all_distances)})
        
        # Add predictions for each model
        for model_type, pred_data in self.predictions.items():
            distances = pred_data.get('distance_values', [])
            times = pred_data.get('predicted_times', [])
            
            # Create a mapping of distance to predicted time
            pred_map = dict(zip(distances, times))
            
            # Add predicted times to the DataFrame
            comparison_df[f'{model_type}_Time'] = comparison_df['Distance'].map(
                lambda d: pred_map.get(d, np.nan)
            )
            
            # Add prediction intervals if available
            if 'lower_intervals' in pred_data and 'upper_intervals' in pred_data:
                lower = pred_data.get('lower_intervals', [])
                upper = pred_data.get('upper_intervals', [])
                
                lower_map = dict(zip(distances, lower))
                upper_map = dict(zip(distances, upper))
                
                comparison_df[f'{model_type}_Lower'] = comparison_df['Distance'].map(
                    lambda d: lower_map.get(d, np.nan)
                )
                comparison_df[f'{model_type}_Upper'] = comparison_df['Distance'].map(
                    lambda d: upper_map.get(d, np.nan)
                )
        
        return comparison_df
    
    def save_prediction_results(self, file_path: str = "reports/prediction_results.json") -> None:
        """
        Save prediction results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run generate_predictions first.")
        
        # Make all values JSON serializable
        def make_serializable(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.float64) or isinstance(item, np.float32):
                return float(item)
            elif isinstance(item, np.int64) or isinstance(item, np.int32):
                return int(item)
            else:
                return item
        
        # Convert predictions to serializable format
        serializable_predictions = {}
        for model_type, pred_data in self.predictions.items():
            serializable_predictions[model_type] = {}
            for k, v in pred_data.items():
                serializable_predictions[model_type][k] = make_serializable(v)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
        
        print(f"[PREDICTION] Prediction results saved to {file_path}")
    
    def generate_formatted_report(self) -> str:
        """
        Generate a human-readable report of the prediction results.
        
        Returns:
            String with formatted prediction report
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run generate_predictions first.")
        
        # Format the prediction table
        comparison_df = self.prepare_prediction_table()
        
        # Create the report
        report = "# Prediction Results Report\n\n"
        report += f"## Predictions for {self.regression_model.target_column} based on {self.regression_model.predictor_column}\n\n"
        
        # Add section for each model
        for model_type, pred_data in self.predictions.items():
            report += f"### Model: {model_type}\n\n"
            
            # Add formula from model results if available
            if model_type in self.regression_model.model_results:
                formula = self.regression_model.model_results[model_type].get('formula', 'Formula not available')
                report += f"Formula: {formula}\n\n"
            
            # Add specific data for this model
            model_df = comparison_df[['Distance']].copy()
            if f'{model_type}_Time' in comparison_df.columns:
                model_df['Predicted_Time'] = comparison_df[f'{model_type}_Time']
                
                if f'{model_type}_Lower' in comparison_df.columns and f'{model_type}_Upper' in comparison_df.columns:
                    model_df['Lower_95%_CI'] = comparison_df[f'{model_type}_Lower']
                    model_df['Upper_95%_CI'] = comparison_df[f'{model_type}_Upper']
                    model_df['Interval_Width'] = model_df['Upper_95%_CI'] - model_df['Lower_95%_CI']
            
            # Format the table for display in the report
            report += model_df.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "\n\n"
        
        # Add comparison section
        report += "## Model Comparison\n\n"
        
        # If we have at least two models, create a comparison table
        if len(self.predictions) >= 2:
            time_cols = [col for col in comparison_df.columns if col.endswith('_Time')]
            
            if time_cols:
                comparison_table = comparison_df[['Distance'] + time_cols].copy()
                report += comparison_table.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "\n\n"
                
                # Calculate differences between models
                model_pairs = []
                for i, col1 in enumerate(time_cols):
                    for j, col2 in enumerate(time_cols):
                        if i < j:  # Compare each pair only once
                            model_name1 = col1.replace('_Time', '')
                            model_name2 = col2.replace('_Time', '')
                            diff_col = f"{model_name1}_vs_{model_name2}"
                            comparison_df[diff_col] = comparison_df[col1] - comparison_df[col2]
                            model_pairs.append((model_name1, model_name2, diff_col))
                
                if model_pairs:
                    report += "### Model Differences\n\n"
                    for model1, model2, diff_col in model_pairs:
                        report += f"#### {model1} vs {model2}\n\n"
                        diffs = comparison_df[['Distance', diff_col]].copy()
                        report += diffs.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "\n\n"
                        avg_diff = comparison_df[diff_col].mean()
                        report += f"Average difference: {avg_diff:.2f}\n\n"
        
        return report


def generate_prediction_examples(regression_model: RegressionModel,
                              distance_values: List[float] = None,
                              num_points: int = 8,
                              save_report: bool = True,
                              report_path: str = "reports/prediction_results.json",
                              generate_plots: bool = True,
                              plots_dir: str = "plots/predictions") -> Dict[str, Any]:
    """
    Generate prediction examples for both full dataset and mode-specific models.
    
    Args:
        regression_model: A fitted RegressionModel instance
        distance_values: List of distance values to predict time for (if None, evenly spaced values will be used)
        num_points: Number of prediction points to generate if distance_values is None
        save_report: Whether to save the report to a file
        report_path: Path to save the report
        generate_plots: Whether to generate prediction plots
        plots_dir: Directory to save plots in
        
    Returns:
        Dictionary with prediction examples and visualization information
    """
    print(f"[PREDICTION] Generating prediction examples for regression models")
    
    try:
        # Create PredictionGenerator instance
        predictor = PredictionGenerator(regression_model)
        
        # Generate predictions
        all_predictions = predictor.generate_predictions(
            distance_values=distance_values,
            num_points=num_points
        )
        
        # Generate plots if requested
        plot_paths = []
        if generate_plots:
            plot_paths = predictor.generate_prediction_plots(output_dir=plots_dir)
        
        # Prepare comparison table
        comparison_df = predictor.prepare_prediction_table()
        
        # Generate formatted report
        formatted_report = predictor.generate_formatted_report()
        
        # Save prediction results if requested
        if save_report:
            predictor.save_prediction_results(file_path=report_path)
        
        # Return comprehensive results
        prediction_results = {
            'predictions': all_predictions,
            'comparison_table': comparison_df.to_dict(),
            'formatted_report': formatted_report,
            'plot_paths': plot_paths,
            'report_path': report_path if save_report else None
        }
        
        # Add success message
        prediction_results['status'] = 'success'
        
        print(f"[PREDICTION] Prediction examples generated successfully")
        return prediction_results
    
    except Exception as e:
        import traceback
        print(f"[PREDICTION ERROR] Error generating prediction examples: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }


def predict_commute_time(transport_mode: str = 'Car', distance: float = 5.0,
                        use_saved_models: bool = True, 
                        model_path: str = "reports/regression_models.json",
                        data_path: str = "Commute_Times_V1_modified.csv") -> Dict[str, Any]:
    """
    Predict commute time for a given transport mode and distance.
    
    Args:
        transport_mode: Mode of transport ('Car', 'Bus', 'Cycle', or 'Walk')
        distance: Distance in miles
        use_saved_models: Whether to use saved models from a JSON file
        model_path: Path to the saved regression models JSON file
        data_path: Path to the dataset (used if not using saved models)
        
    Returns:
        Dictionary with prediction results
    """
    print(f"[PREDICTION] Predicting commute time for {distance} miles using {transport_mode}")
    
    try:
        if use_saved_models and os.path.exists(model_path):
            # Load the regression model from saved file
            df = pd.read_csv(data_path)
            regression_model = RegressionModel(df, 'Time', 'Distance')
            regression_model.load_model_results(model_path)
            print(f"[PREDICTION] Loaded models from {model_path}")
        else:
            # Fit a new regression model
            print(f"[PREDICTION] Fitting new regression models...")
            df = pd.read_csv(data_path)
            regression_results = perform_regression_analysis(df)
            
            # Even if there was an error during saving, we can still use the model that was created
            # Get the regression model directly from the function call
            regression_model = RegressionModel(df, 'Time', 'Distance')
            regression_model.fit_full_dataset_model()
            regression_model.fit_mode_specific_models()
            
        # Select the appropriate model for this transport mode
        if transport_mode in ['Car', 'Bus', 'Cycle', 'Walk']:
            model_type = f"mode_{transport_mode}"
        else:
            model_type = "full_dataset"
            print(f"[PREDICTION] Warning: Unknown transport mode '{transport_mode}'. Using full dataset model.")
        
        # Check if the requested model is available
        if model_type not in regression_model.models:
            available_models = list(regression_model.models.keys())
            print(f"[PREDICTION] Warning: Model for {transport_mode} not found. Available models: {available_models}")
            model_type = 'full_dataset'
            print(f"[PREDICTION] Using {model_type} model instead")
        
        # Make the prediction
        prediction_result = regression_model.predict([distance], model_type)
        predicted_time = prediction_result.get('predicted_times', [0])[0]
        
        # Check for prediction intervals
        lower_interval = prediction_result.get('lower_intervals', [None])[0]
        upper_interval = prediction_result.get('upper_intervals', [None])[0]
        
        # Return formatted results
        result = {
            'transport_mode': transport_mode,
            'distance': distance,
            'predicted_time': predicted_time,
            'model_used': model_type,
            'status': 'success'
        }
        
        # Add prediction intervals if available
        if lower_interval is not None and upper_interval is not None:
            result['prediction_interval'] = {
                'lower': lower_interval,
                'upper': upper_interval
            }
            
        # Add model formula if available
        if model_type in regression_model.model_results:
            formula = regression_model.model_results[model_type].get('formula')
            if formula:
                result['formula'] = formula
        
        print(f"[PREDICTION] Predicted time for {distance} miles via {transport_mode}: {predicted_time:.2f} minutes")
        return result
        
    except Exception as e:
        import traceback
        print(f"[PREDICTION ERROR] Error predicting commute time: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }