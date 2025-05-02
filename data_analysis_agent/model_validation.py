import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import json

class ModelValidator:
    """
    A class to validate regression models using cross-validation and residual analysis.
    """
    
    def __init__(self, model_name: str = "default_model"):
        """
        Initialize the ModelValidator class.
        
        Args:
            model_name: A name for the model being validated
        """
        self.model_name = model_name
        self.validation_results = {}
        
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                               k_folds: int = 5, random_state: Optional[int] = 42) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on a linear regression model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"[VALIDATION] Performing {k_folds}-fold cross-validation for {self.model_name}")
        
        # Initialize cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        # Initialize metrics lists
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        # Track predictions for each fold
        all_y_pred = []
        all_y_true = []
        
        # Loop through folds
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # Split data for this fold
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model on training data
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions on test data
            y_pred = model.predict(X_test)
            
            # Store predictions and actual values
            all_y_pred.extend(y_pred)
            all_y_true.extend(y_test)
            
            # Calculate metrics for this fold
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store metrics
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            fold_results.append({
                'fold': fold,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })
            
            print(f"[VALIDATION] Fold {fold}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Calculate average metrics across folds
        avg_r2 = np.mean(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        
        # Calculate standard deviations of metrics
        std_r2 = np.std(r2_scores)
        std_rmse = np.std(rmse_scores)
        std_mae = np.std(mae_scores)
        
        # Store validation results
        cv_results = {
            'model_name': self.model_name,
            'k_folds': k_folds,
            'metrics': {
                'r2_mean': avg_r2,
                'r2_std': std_r2,
                'rmse_mean': avg_rmse,
                'rmse_std': std_rmse,
                'mae_mean': avg_mae,
                'mae_std': std_mae
            },
            'fold_results': fold_results
        }
        
        self.validation_results['cross_validation'] = cv_results
        
        print(f"[VALIDATION] Cross-validation complete: Avg R² = {avg_r2:.4f} (±{std_r2:.4f})")
        
        return cv_results
    
    def analyze_residuals(self, model: sm.regression.linear_model.RegressionResultsWrapper, 
                         X: pd.DataFrame, y: pd.Series, title_prefix: str = "") -> Dict[str, Any]:
        """
        Perform detailed residual analysis including normality and homoscedasticity tests.
        
        Args:
            model: A fitted statsmodels regression model
            X: Feature DataFrame
            y: Target Series
            title_prefix: Prefix for plot titles
            
        Returns:
            Dictionary with residual analysis results
        """
        print(f"[VALIDATION] Performing residual analysis for {self.model_name}")
        
        # Get residuals
        residuals = model.resid
        
        # Calculate predicted values
        if isinstance(X, pd.DataFrame) and 'const' not in X.columns:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X
            
        predicted = model.predict(X_with_const)
        
        # 1. Test normality of residuals using Shapiro-Wilk and Anderson-Darling tests
        # Shapiro-Wilk test
        if len(residuals) <= 5000:  # Shapiro-Wilk has a limit on sample size
            shapiro_test = stats.shapiro(residuals)
            shapiro_stat = shapiro_test[0]
            shapiro_p = shapiro_test[1]
            shapiro_normal = shapiro_p > 0.05
        else:
            shapiro_stat = None
            shapiro_p = None
            shapiro_normal = None
            
        # Anderson-Darling test
        anderson_test = stats.anderson(residuals, 'norm')
        anderson_stat = anderson_test[0]
        anderson_critical_values = anderson_test[1]
        anderson_significance_levels = anderson_test[2]
        
        # Find the appropriate significance level (0.05)
        anderson_idx = np.where(anderson_significance_levels == 5)[0][0]
        anderson_normal = anderson_stat < anderson_critical_values[anderson_idx]
        
        # 2. Test for homoscedasticity using Breusch-Pagan test
        bp_test = het_breuschpagan(residuals, X_with_const)
        bp_stat = bp_test[0]
        bp_p = bp_test[1]
        homoscedastic = bp_p > 0.05
        
        # Store normality and homoscedasticity results
        residual_results = {
            'normality_tests': {
                'shapiro': {
                    'statistic': None if shapiro_stat is None else float(shapiro_stat),
                    'p_value': None if shapiro_p is None else float(shapiro_p),
                    'is_normal': None if shapiro_normal is None else bool(shapiro_normal)
                },
                'anderson_darling': {
                    'statistic': float(anderson_stat),
                    'critical_value_5pct': float(anderson_critical_values[anderson_idx]),
                    'is_normal': bool(anderson_normal)
                }
            },
            'homoscedasticity_test': {
                'breusch_pagan': {
                    'statistic': float(bp_stat),
                    'p_value': float(bp_p),
                    'is_homoscedastic': bool(homoscedastic)
                }
            },
            'residuals_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals))
            }
        }
        
        # Store residual analysis results
        self.validation_results['residual_analysis'] = residual_results
        
        print(f"[VALIDATION] Residual analysis complete:")
        print(f"  - Shapiro-Wilk: {'Normal' if shapiro_normal else 'Not normal'} (p = {shapiro_p:.4f})" if shapiro_p is not None else "  - Shapiro-Wilk: N/A (sample too large)")
        print(f"  - Anderson-Darling: {'Normal' if anderson_normal else 'Not normal'}")
        print(f"  - Breusch-Pagan: {'Homoscedastic' if homoscedastic else 'Heteroscedastic'} (p = {bp_p:.4f})")
        
        return residual_results

    def generate_residual_plots(self, model: sm.regression.linear_model.RegressionResultsWrapper,
                             X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None,
                             output_dir: str = "plots/validation") -> List[str]:
        """
        Generate diagnostic plots for residual analysis.
        
        Args:
            model: A fitted statsmodels regression model
            X: Feature DataFrame
            y: Target Series
            feature_names: List of feature names for plots (if None, will use X.columns)
            output_dir: Directory to save plots in
            
        Returns:
            List of saved plot file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = []
        
        # Get residuals
        residuals = model.resid
        
        # Add constant if needed for prediction
        if isinstance(X, pd.DataFrame) and 'const' not in X.columns:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X
            
        predicted = model.predict(X_with_const)
        
        # If feature names not provided, use X column names
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            # Remove 'const' from feature names if present
            if 'const' in feature_names:
                feature_names.remove('const')
        elif feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        
        # 1. Residuals vs Fitted Values plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{self.model_name}: Residuals vs Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        # Add a smoothed line to help visualize patterns
        try:
            sns.regplot(x=predicted, y=residuals, scatter=False, lowess=True, 
                       line_kws={'color': 'red', 'lw': 1})
        except:
            pass  # If lowess fails, just continue without the smoothed line
        
        residuals_fitted_path = os.path.join(output_dir, f"{self.model_name.lower().replace(' ', '_')}_residuals_vs_fitted.png")
        plt.savefig(residuals_fitted_path)
        plt.close()
        plot_paths.append(residuals_fitted_path)
        print(f"[VALIDATION] Saved plot: {residuals_fitted_path}")
        
        # 2. Q-Q plot for normality check
        plt.figure(figsize=(8, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{self.model_name}: Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        qq_plot_path = os.path.join(output_dir, f"{self.model_name.lower().replace(' ', '_')}_residuals_qq.png")
        plt.savefig(qq_plot_path)
        plt.close()
        plot_paths.append(qq_plot_path)
        print(f"[VALIDATION] Saved plot: {qq_plot_path}")
        
        # 3. Residuals distribution (histogram)
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'{self.model_name}: Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        residuals_hist_path = os.path.join(output_dir, f"{self.model_name.lower().replace(' ', '_')}_residuals_hist.png")
        plt.savefig(residuals_hist_path)
        plt.close()
        plot_paths.append(residuals_hist_path)
        print(f"[VALIDATION] Saved plot: {residuals_hist_path}")
        
        # 4. Residuals vs Predictors plots
        if isinstance(X, pd.DataFrame):
            for i, feature in enumerate(feature_names):
                if feature in X.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X[feature], residuals, alpha=0.6)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.title(f'{self.model_name}: Residuals vs {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Residuals')
                    plt.grid(True, alpha=0.3)
                    
                    # Add a smoothed line
                    try:
                        sns.regplot(x=X[feature], y=residuals, scatter=False, lowess=True, 
                                   line_kws={'color': 'red', 'lw': 1})
                    except:
                        pass
                    
                    residuals_pred_path = os.path.join(output_dir, f"{self.model_name.lower().replace(' ', '_')}_residuals_vs_{feature.lower().replace(' ', '_')}.png")
                    plt.savefig(residuals_pred_path)
                    plt.close()
                    plot_paths.append(residuals_pred_path)
                    print(f"[VALIDATION] Saved plot: {residuals_pred_path}")
        
        # 5. Scale-Location plot (Spread-Location) - sqrt(|standardized residuals|) vs fitted values
        plt.figure(figsize=(10, 6))
        standardized_residuals = residuals / np.sqrt(np.mean(residuals**2))
        plt.scatter(predicted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
        plt.title(f'{self.model_name}: Scale-Location Plot')
        plt.xlabel('Fitted Values')
        plt.ylabel('√|Standardized Residuals|')
        plt.grid(True, alpha=0.3)
        
        # Add a smoothed line
        try:
            sns.regplot(x=predicted, y=np.sqrt(np.abs(standardized_residuals)), scatter=False, 
                       lowess=True, line_kws={'color': 'red', 'lw': 1})
        except:
            pass
        
        scale_location_path = os.path.join(output_dir, f"{self.model_name.lower().replace(' ', '_')}_scale_location.png")
        plt.savefig(scale_location_path)
        plt.close()
        plot_paths.append(scale_location_path)
        print(f"[VALIDATION] Saved plot: {scale_location_path}")
        
        return plot_paths
    
    def save_validation_results(self, file_path: str = "reports/model_validation.json") -> None:
        """
        Save validation results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        # Make all values JSON serializable
        def make_serializable(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.float64) or isinstance(item, np.float32):
                return float(item)
            elif isinstance(item, np.int64) or isinstance(item, np.int32):
                return int(item)
            elif isinstance(item, bool):  # Handle boolean values properly
                return item  # Return as-is, native JSON boolean
            else:
                return item
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in self.validation_results.items():
            if isinstance(result, dict):
                serializable_results[key] = {}
                for k, v in result.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {k2: make_serializable(v2) for k2, v2 in v.items()}
                    else:
                        serializable_results[key][k] = make_serializable(v)
            else:
                serializable_results[key] = make_serializable(result)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[VALIDATION] Validation results saved to {file_path}")


def validate_regression_model(model: sm.regression.linear_model.RegressionResultsWrapper,
                             X: pd.DataFrame, y: pd.Series,
                             model_name: str = "Regression Model",
                             feature_names: Optional[List[str]] = None,
                             save_report: bool = True, 
                             report_path: str = "reports/model_validation.json",
                             generate_plots: bool = True,
                             plots_dir: str = "plots/validation") -> Dict[str, Any]:
    """
    Perform comprehensive validation of a regression model including cross-validation and residual analysis.
    
    Args:
        model: A fitted statsmodels regression model
        X: Feature DataFrame
        y: Target Series
        model_name: Name for the model being validated
        feature_names: List of feature names for plots
        save_report: Whether to save the report to a file
        report_path: Path to save the report
        generate_plots: Whether to generate validation plots
        plots_dir: Directory to save plots in
        
    Returns:
        Dictionary with validation results
    """
    print(f"[VALIDATION] Starting validation for model: {model_name}")
    
    try:
        # Create ModelValidator instance
        validator = ModelValidator(model_name=model_name)
        
        # Perform cross-validation
        cv_results = validator.perform_cross_validation(X, y, k_folds=5)
        
        # Perform residual analysis
        residual_results = validator.analyze_residuals(model, X, y)
        
        # Generate plots if requested
        plot_paths = []
        if generate_plots:
            plot_paths = validator.generate_residual_plots(model, X, y, feature_names, output_dir=plots_dir)
        
        # Save validation results if requested
        if save_report:
            validator.save_validation_results(file_path=report_path)
        
        # Return comprehensive results
        validation_results = {
            'cross_validation': cv_results,
            'residual_analysis': residual_results,
            'plot_paths': plot_paths,
            'report_path': report_path if save_report else None
        }
        
        # Add summary of compliance with assumptions
        assumptions_met = {
            'normality': residual_results['normality_tests']['anderson_darling']['is_normal'],
            'homoscedasticity': residual_results['homoscedasticity_test']['breusch_pagan']['is_homoscedastic'],
            'zero_mean_residuals': abs(residual_results['residuals_stats']['mean']) < 0.001
        }
        validation_results['assumptions_met'] = assumptions_met
        
        # Add success message
        validation_results['status'] = 'success'
        
        print(f"[VALIDATION] Model validation completed successfully")
        return validation_results
    
    except Exception as e:
        import traceback
        print(f"[VALIDATION ERROR] Error in model validation: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }