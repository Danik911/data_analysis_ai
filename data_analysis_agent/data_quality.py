"""
Data Quality Assessment and Cleaning Module

This module provides functionality for comprehensive data quality assessment and cleaning,
implementing the requirements from the project plan:
- Systematic data type verification
- Value range checking with Tukey's method for outliers
- Uniqueness verification for Case Numbers
- Impossible value detection
- Data quality reporting
- Mode value standardization
- Cleaning documentation
- Before/after comparison metrics
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class DataQualityAssessment:
    """
    Class for comprehensive data quality assessment, including data type verification,
    outlier detection using Tukey's method, duplicate detection, and more.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The dataset to assess
        """
        self.df = df.copy()
        self.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.results = {}
        
    def verify_data_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Performs systematic data type verification for all columns.
        
        Returns:
            Dict: Results of data type verification for each column
        """
        type_verification = {}
        
        for col in self.df.columns:
            column_data = self.df[col]
            current_type = str(column_data.dtype)
            
            # Check if numeric columns contain non-numeric values that were coerced
            non_numeric_count = 0
            if pd.api.types.is_numeric_dtype(column_data):
                # For numeric columns, check if there were strings coerced
                temp_series = pd.to_numeric(self.df[col], errors='coerce')
                non_numeric_count = temp_series.isna().sum() - self.df[col].isna().sum()
            
            # Check if the column contains mixed types
            inferred_types = set()
            for val in column_data.dropna().unique():
                if isinstance(val, (int, np.integer)):
                    inferred_types.add('integer')
                elif isinstance(val, (float, np.floating)):
                    inferred_types.add('float')
                elif isinstance(val, str):
                    inferred_types.add('string')
                else:
                    inferred_types.add(str(type(val)))
            
            # Determine if current type is appropriate
            suggested_type = current_type
            if current_type == 'object' and len(inferred_types) == 1 and 'integer' in inferred_types:
                suggested_type = 'int64'
            elif current_type == 'object' and len(inferred_types) == 1 and 'float' in inferred_types:
                suggested_type = 'float64'
                
            type_verification[col] = {
                'current_type': current_type,
                'inferred_types': list(inferred_types),
                'non_numeric_count': non_numeric_count,
                'suggested_type': suggested_type,
                'mixed_types': len(inferred_types) > 1
            }
        
        self.results['type_verification'] = type_verification
        return type_verification
    
    def check_missing_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Checks for missing values in each column and identifies patterns.
        
        Returns:
            Dict: Missing value counts and patterns for each column
        """
        missing_values = {}
        
        # Missing value counts by column
        missing_counts = self.df.isna().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        # Check for patterns in missing values
        rows_with_missing = self.df[self.df.isna().any(axis=1)]
        missing_combinations = {}
        
        if not rows_with_missing.empty:
            # Count occurrences of each missing value pattern
            missing_patterns = rows_with_missing.isna().apply(lambda x: tuple(x), axis=1)
            pattern_counts = missing_patterns.value_counts().to_dict()
            
            # Convert tuple keys to string representations for JSON compatibility
            for pattern, count in pattern_counts.items():
                pattern_str = ', '.join([self.df.columns[i] for i, is_missing in enumerate(pattern) if is_missing])
                missing_combinations[pattern_str] = count
        
        for col in self.df.columns:
            missing_values[col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_percentages[col]),
                'is_significant': missing_percentages[col] > 5
            }
        
        self.results['missing_values'] = {
            'column_details': missing_values,
            'total_missing_rows': len(rows_with_missing),
            'missing_patterns': missing_combinations
        }
        return self.results['missing_values']
    
    def detect_outliers_tukey(self, numeric_only=True) -> Dict[str, Dict[str, Any]]:
        """
        Detects outliers using Tukey's method (1.5 * IQR).
        
        Args:
            numeric_only (bool): If True, only analyze numeric columns
            
        Returns:
            Dict: Outlier information for each applicable column
        """
        outliers = {}
        
        # Select columns for analysis
        cols_to_analyze = self.df.select_dtypes(include=['number']).columns if numeric_only else self.df.columns
        
        for col in cols_to_analyze:
            # Skip columns with all missing values
            if self.df[col].isna().all():
                continue
                
            series = self.df[col].dropna()
            
            # Calculate quartiles and IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index.tolist()
            outlier_values = series[outlier_mask].tolist()
            
            # Calculate z-scores for comparison
            z_scores = stats.zscore(series, nan_policy='omit')
            z_score_mask = np.abs(z_scores) > 3
            z_outlier_indices = series.index[z_score_mask].tolist()
            
            # Store results
            outliers[col] = {
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': int(outlier_mask.sum()),
                'outlier_percentage': float((outlier_mask.sum() / len(series)) * 100),
                'outlier_indices': outlier_indices[:10],  # Limit to first 10 for readability
                'outlier_values': outlier_values[:10],    # Limit to first 10 for readability
                'z_score_outlier_count': int(z_score_mask.sum()),
                'method_agreement_percentage': float((sum(idx in z_outlier_indices for idx in outlier_indices) / 
                                                     max(len(outlier_indices), 1)) * 100)
            }
        
        self.results['outliers_tukey'] = outliers
        return outliers
    
    def check_duplicates(self, subset=None) -> Dict[str, Any]:
        """
        Checks for duplicate rows in the dataset, optionally based on specific columns.
        
        Args:
            subset (list, optional): List of columns to check for duplicates. If None, use all columns.
            
        Returns:
            Dict: Duplicate detection results
        """
        if subset is None:
            duplicates = self.df.duplicated()
            duplicate_indices = self.df[duplicates].index.tolist()
        else:
            duplicates = self.df.duplicated(subset=subset)
            duplicate_indices = self.df[duplicates].index.tolist()
        
        duplicate_rows = self.df.loc[duplicate_indices].to_dict('records') if duplicate_indices else []
        
        results = {
            'count': int(duplicates.sum()),
            'percentage': float((duplicates.sum() / len(self.df)) * 100),
            'indices': duplicate_indices[:10],  # Limit to first 10 for readability
            'examples': duplicate_rows[:5]      # Limit to first 5 for readability
        }
        
        if subset:
            self.results[f'duplicates_{"-".join(subset)}'] = results
        else:
            self.results['duplicates'] = results
            
        return results
    
    def identify_impossible_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Identifies impossible or unreasonable values based on domain knowledge.
        
        Returns:
            Dict: Impossible values for each column
        """
        impossible_values = {}
        
        # Define domain-specific rules
        rules = {
            'Distance': {'min': 0, 'max': 50},  # Distances should be positive and reasonable (e.g., < 50km)
            'Time': {'min': 0, 'max': 120}      # Time should be positive and reasonable (e.g., < 120 minutes)
        }
        
        for col, constraints in rules.items():
            if col in self.df.columns:
                min_val = constraints.get('min')
                max_val = constraints.get('max')
                
                # Check for values outside acceptable range
                if min_val is not None:
                    too_small_mask = self.df[col] < min_val
                    too_small_indices = self.df[too_small_mask].index.tolist()
                    too_small_values = self.df.loc[too_small_indices, col].tolist()
                else:
                    too_small_mask = pd.Series(False, index=self.df.index)
                    too_small_indices = []
                    too_small_values = []
                
                if max_val is not None:
                    too_large_mask = self.df[col] > max_val
                    too_large_indices = self.df[too_large_mask].index.tolist()
                    too_large_values = self.df.loc[too_large_indices, col].tolist()
                else:
                    too_large_mask = pd.Series(False, index=self.df.index)
                    too_large_indices = []
                    too_large_values = []
                
                combined_mask = too_small_mask | too_large_mask
                
                impossible_values[col] = {
                    'min_constraint': min_val,
                    'max_constraint': max_val,
                    'total_violations': int(combined_mask.sum()),
                    'too_small_count': int(too_small_mask.sum()),
                    'too_small_indices': too_small_indices[:10],
                    'too_small_values': too_small_values[:10],
                    'too_large_count': int(too_large_mask.sum()),
                    'too_large_indices': too_large_indices[:10],
                    'too_large_values': too_large_values[:10]
                }
        
        # Check for impossible mode values (typos or invalid categories)
        if 'Mode' in self.df.columns:
            # Standard mode values we expect
            valid_modes = {'Car', 'Bus', 'Cycle', 'Walk', 'Train', 'Tram', 'Subway'}
            
            # Find values that are likely typos
            mode_values = self.df['Mode'].dropna().unique()
            invalid_modes = [mode for mode in mode_values if mode not in valid_modes]
            
            # Create mask for invalid modes
            invalid_mask = self.df['Mode'].isin(invalid_modes)
            invalid_indices = self.df[invalid_mask].index.tolist()
            
            impossible_values['Mode'] = {
                'valid_values': list(valid_modes),
                'invalid_values': invalid_modes,
                'invalid_count': int(invalid_mask.sum()),
                'invalid_indices': invalid_indices[:10],
                'likely_corrections': {mode: self._suggest_correction(mode, valid_modes) for mode in invalid_modes}
            }
        
        self.results['impossible_values'] = impossible_values
        return impossible_values
    
    def _suggest_correction(self, invalid_mode: str, valid_modes: set) -> str:
        """
        Suggests a correction for an invalid mode based on string similarity.
        
        Args:
            invalid_mode (str): The invalid mode value
            valid_modes (set): Set of valid mode values
            
        Returns:
            str: Suggested correction
        """
        # Simple correction based on first letter
        if not invalid_mode:
            return "Unknown"
            
        first_letter = invalid_mode[0].upper()
        
        for mode in valid_modes:
            if mode.startswith(first_letter):
                return mode
                
        # If no match by first letter, could use more sophisticated method like Levenshtein distance
        return "Unknown"
    
    def check_distribution(self, numeric_only=True) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes the distribution of values in each applicable column,
        including normality tests.
        
        Args:
            numeric_only (bool): If True, only analyze numeric columns
            
        Returns:
            Dict: Distribution statistics for each applicable column
        """
        distribution_stats = {}
        
        # Select columns for analysis
        cols_to_analyze = self.df.select_dtypes(include=['number']).columns if numeric_only else self.df.columns
        
        for col in cols_to_analyze:
            # Skip columns with too few values or all missing
            series = self.df[col].dropna()
            if len(series) < 3:
                continue
                
            # Basic statistics
            mean = series.mean()
            median = series.median()
            std = series.std()
            skewness = series.skew()
            kurtosis = series.kurt()
            
            # Normality tests
            if len(series) >= 8:  # Shapiro-Wilk needs at least 3 values
                shapiro_test = stats.shapiro(series)
                shapiro_p_value = shapiro_test[1]
            else:
                shapiro_p_value = None
                
            distribution_stats[col] = {
                'mean': float(mean),
                'median': float(median),
                'std': float(std),
                'min': float(series.min()),
                'max': float(series.max()),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'shapiro_p_value': float(shapiro_p_value) if shapiro_p_value is not None else None,
                'is_normal': shapiro_p_value > 0.05 if shapiro_p_value is not None else None,
                'is_skewed': abs(skewness) > 1,
                'skew_direction': 'right' if skewness > 0 else 'left' if skewness < 0 else 'none'
            }
        
        self.results['distribution_stats'] = distribution_stats
        return distribution_stats
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive data quality report including all assessments.
        
        Returns:
            Dict: Complete data quality report
        """
        # Ensure all assessments have been run
        if 'type_verification' not in self.results:
            self.verify_data_types()
        if 'missing_values' not in self.results:
            self.check_missing_values()
        if 'outliers_tukey' not in self.results:
            self.detect_outliers_tukey()
        if 'duplicates' not in self.results:
            self.check_duplicates()
        if 'duplicates_Case' not in self.results and 'Case' in self.df.columns:
            self.check_duplicates(subset=['Case'])
        if 'impossible_values' not in self.results:
            self.identify_impossible_values()
        if 'distribution_stats' not in self.results:
            self.check_distribution()
        
        # Count issues by category
        issue_summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_value_count': sum(self.results['missing_values']['column_details'][col]['count'] for col in self.df.columns),
            'duplicate_row_count': self.results['duplicates']['count'],
            'outlier_count': sum(stats['outlier_count'] for col, stats in self.results['outliers_tukey'].items() if col in self.df.columns),
            'impossible_value_count': sum(stats['total_violations'] 
                                         for col, stats in self.results['impossible_values'].items() 
                                         if col in self.df.columns and 'total_violations' in stats)
        }
        
        # Include Case duplicates if available
        if 'duplicates_Case' in self.results:
            issue_summary['duplicate_case_count'] = self.results['duplicates_Case']['count']
        
        # Add mode issues if available
        if 'Mode' in self.df.columns and 'Mode' in self.results['impossible_values']:
            issue_summary['invalid_mode_count'] = self.results['impossible_values']['Mode']['invalid_count']
        
        # Create summary of data types
        type_summary = {}
        for col, info in self.results['type_verification'].items():
            if info['mixed_types'] or info['current_type'] != info['suggested_type'] or info['non_numeric_count'] > 0:
                type_summary[col] = {
                    'has_issues': True,
                    **info
                }
            else:
                type_summary[col] = {
                    'has_issues': False,
                    'current_type': info['current_type']
                }
        
        # Create overall quality score (simple version)
        max_score = 100
        deductions = 0
        
        # Deduct for missing values (up to 20 points)
        missing_percentage = issue_summary['missing_value_count'] / (issue_summary['total_rows'] * issue_summary['total_columns']) * 100
        deductions += min(20, missing_percentage * 2)
        
        # Deduct for duplicates (up to 15 points)
        if issue_summary['total_rows'] > 0:
            duplicate_percentage = issue_summary['duplicate_row_count'] / issue_summary['total_rows'] * 100
            deductions += min(15, duplicate_percentage * 3)
        
        # Deduct for outliers (up to 10 points)
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0 and issue_summary['total_rows'] > 0:
            outlier_percentage = issue_summary['outlier_count'] / (issue_summary['total_rows'] * len(numeric_columns)) * 100
            deductions += min(10, outlier_percentage * 2)
        
        # Deduct for impossible values (up to 15 points)
        constrained_columns = [col for col in self.df.columns if col in self.results['impossible_values']]
        if len(constrained_columns) > 0 and issue_summary['total_rows'] > 0:
            impossible_percentage = issue_summary['impossible_value_count'] / (issue_summary['total_rows'] * len(constrained_columns)) * 100
            deductions += min(15, impossible_percentage * 3)
        
        # Deduct for type issues (up to 10 points)
        type_issue_count = sum(1 for col, info in type_summary.items() if info['has_issues'])
        if issue_summary['total_columns'] > 0:
            type_issue_percentage = type_issue_count / issue_summary['total_columns'] * 100
            deductions += min(10, type_issue_percentage * 2)
        
        quality_score = max(0, max_score - deductions)
        
        # Compile the report
        report = {
            'dataset_info': {
                'total_rows': issue_summary['total_rows'],
                'total_columns': issue_summary['total_columns'],
                'column_names': list(self.df.columns),
                'quality_score': quality_score
            },
            'issue_summary': issue_summary,
            'type_verification': type_summary,
            'missing_values': self.results['missing_values'],
            'duplicates': self.results['duplicates'],
            'outliers': self.results['outliers_tukey'],
            'impossible_values': self.results['impossible_values'],
            'distribution_stats': self.results['distribution_stats'],
            'recommendations': self._generate_recommendations()
        }
        
        if 'duplicates_Case' in self.results:
            report['case_duplicates'] = self.results['duplicates_Case']
        
        return report
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """
        Generates recommendations for cleaning based on assessment results.
        
        Returns:
            Dict: Recommendations by category
        """
        recommendations = {
            'missing_values': [],
            'duplicates': [],
            'outliers': [],
            'impossible_values': [],
            'data_types': []
        }
        
        # Missing value recommendations
        missing_significant = [col for col in self.df.columns 
                              if self.results['missing_values']['column_details'][col]['is_significant']]
        if missing_significant:
            recommendations['missing_values'].append(
                f"Consider imputation strategies for columns with significant missing values: {', '.join(missing_significant)}"
            )
            
            if 'Time' in missing_significant:
                recommendations['missing_values'].append(
                    "For 'Time' column, consider imputation based on 'Distance' and 'Mode' using regression models"
                )
        
        # Duplicate recommendations
        if self.results['duplicates']['count'] > 0:
            recommendations['duplicates'].append(
                f"Remove {self.results['duplicates']['count']} duplicate rows from the dataset"
            )
            
        if 'duplicates_Case' in self.results and self.results['duplicates_Case']['count'] > 0:
            recommendations['duplicates'].append(
                f"Investigate and resolve {self.results['duplicates_Case']['count']} duplicate Case numbers"
            )
        
        # Outlier recommendations
        outlier_cols = [col for col, stats in self.results['outliers_tukey'].items() 
                        if stats['outlier_count'] > 0]
        if outlier_cols:
            recommendations['outliers'].append(
                f"Review outliers in columns: {', '.join(outlier_cols)}"
            )
            
            if 'Distance' in outlier_cols:
                recommendations['outliers'].append(
                    "For 'Distance' outliers, consider capping at 3 standard deviations from the mean"
                )
                
            if 'Time' in outlier_cols:
                recommendations['outliers'].append(
                    "For 'Time' outliers, consider domain-specific rules (e.g., cap at 120 minutes)"
                )
        
        # Impossible value recommendations
        for col, stats in self.results['impossible_values'].items():
            if col in ['Distance', 'Time'] and stats['total_violations'] > 0:
                recommendations['impossible_values'].append(
                    f"Replace {stats['total_violations']} impossible values in '{col}' with appropriate bounds"
                )
                
            elif col == 'Mode' and 'invalid_count' in stats and stats['invalid_count'] > 0:
                recommendations['impossible_values'].append(
                    f"Standardize {stats['invalid_count']} invalid Mode values using the suggested corrections"
                )
        
        # Data type recommendations
        for col, info in self.results['type_verification'].items():
            if info['current_type'] != info['suggested_type']:
                recommendations['data_types'].append(
                    f"Convert '{col}' from {info['current_type']} to {info['suggested_type']}"
                )
            
            if info['mixed_types']:
                recommendations['data_types'].append(
                    f"Resolve mixed types in '{col}': {', '.join(info['inferred_types'])}"
                )
        
        return recommendations
    
    def save_report(self, file_path: str) -> str:
        """
        Saves the data quality report to a JSON file.
        
        Args:
            file_path (str): Path to save the report
            
        Returns:
            str: Success message
        """
        report = self.generate_report()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            return f"Data quality report saved to {file_path}"
        except Exception as e:
            return f"Error saving report: {e}"


class DataCleaner:
    """
    Class for cleaning and transforming data based on quality assessment results.
    Includes functionality for standardizing values, handling missing data,
    removing outliers, and tracking before/after metrics.
    """
    
    def __init__(self, df: pd.DataFrame, assessment_report: Dict[str, Any] = None):
        """
        Initialize with a pandas DataFrame and optionally an assessment report.
        
        Args:
            df (pd.DataFrame): The dataset to clean
            assessment_report (Dict, optional): Data quality assessment report
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.assessment_report = assessment_report
        self.cleaning_log = []
        self.before_metrics = self._calculate_metrics(self.original_df)
        self.after_metrics = None
        
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate key metrics for the DataFrame to enable before/after comparison.
        
        Args:
            df (pd.DataFrame): DataFrame to calculate metrics for
            
        Returns:
            Dict: Metrics including row count, missing values, etc.
        """
        metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_value_counts': df.isna().sum().to_dict(),
            'total_missing_values': df.isna().sum().sum(),
            'unique_counts': {col: df[col].nunique() for col in df.columns},
        }
        
        # Add descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            metrics['numeric_stats'] = {}
            for col in numeric_cols:
                metrics['numeric_stats'][col] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None
                }
        
        # Mode value counts if 'Mode' column exists
        if 'Mode' in df.columns:
            metrics['mode_value_counts'] = df['Mode'].value_counts().to_dict()
        
        return metrics
    
    def log_cleaning_step(self, action: str, details: Dict[str, Any]) -> None:
        """
        Log a cleaning action for documentation.
        
        Args:
            action (str): Description of the cleaning action
            details (Dict): Details about the action (e.g., affected rows, columns)
        """
        self.cleaning_log.append({
            'action': action,
            'details': details,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    def standardize_mode_values(self) -> pd.DataFrame:
        """
        Standardize Mode values using frequency analysis to identify and correct typos.
        
        Returns:
            pd.DataFrame: DataFrame with standardized Mode values
        """
        if 'Mode' not in self.df.columns:
            self.log_cleaning_step('standardize_mode_values', {
                'status': 'skipped',
                'reason': 'Mode column not found'
            })
            return self.df
        
        # Get current Mode values and their counts
        mode_counts = self.df['Mode'].value_counts()
        
        # Define standardization mapping based on assessment if available
        mode_mapping = {}
        
        if (self.assessment_report and 'impossible_values' in self.assessment_report and 
            'Mode' in self.assessment_report['impossible_values']):
            mode_info = self.assessment_report['impossible_values']['Mode']
            mode_mapping = mode_info.get('likely_corrections', {})
        else:
            # Simple mapping based on common typos
            mode_mapping = {
                'Cra': 'Car',
                'Wilk': 'Walk',
                'Walt': 'Walk',
                'Bas': 'Bus',
                'Buss': 'Bus',
                'Cyc': 'Cycle'
            }
        
        # Apply standardization
        original_values = self.df['Mode'].copy()
        self.df['Mode'] = self.df['Mode'].replace(mode_mapping)
        
        # Count changes
        changes = (original_values != self.df['Mode']).sum()
        
        # Log the action
        self.log_cleaning_step('standardize_mode_values', {
            'status': 'completed',
            'changes': int(changes),
            'mapping': mode_mapping,
            'value_counts_before': original_values.value_counts().to_dict(),
            'value_counts_after': self.df['Mode'].value_counts().to_dict()
        })
        
        return self.df
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values using specified strategies.
        
        Args:
            strategy (Dict, optional): Mapping of column names to imputation strategies
                (e.g., {'Time': 'median', 'Distance': 'mean'})
                
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if strategy is None:
            # Default strategies based on data types
            strategy = {}
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            
            for col in numeric_cols:
                strategy[col] = 'median'  # More robust to outliers than mean
            
            for col in categorical_cols:
                strategy[col] = 'mode'
        
        before_missing = self.df.isna().sum().to_dict()
        imputation_details = {}
        
        for col, method in strategy.items():
            if col not in self.df.columns:
                continue
                
            if method == 'mean':
                fill_value = self.df[col].mean()
                self.df[col] = self.df[col].fillna(fill_value)
                imputation_details[col] = {'method': 'mean', 'value': float(fill_value)}
                
            elif method == 'median':
                fill_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_value)
                imputation_details[col] = {'method': 'median', 'value': float(fill_value)}
                
            elif method == 'mode':
                fill_value = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(fill_value)
                imputation_details[col] = {'method': 'mode', 'value': fill_value}
                
            elif method == 'zero':
                self.df[col] = self.df[col].fillna(0)
                imputation_details[col] = {'method': 'zero', 'value': 0}
                
            elif method == 'drop':
                self.df = self.df.dropna(subset=[col])
                imputation_details[col] = {'method': 'drop', 'rows_removed': before_missing[col]}
        
        after_missing = self.df.isna().sum().to_dict()
        
        # Log the action
        self.log_cleaning_step('handle_missing_values', {
            'status': 'completed',
            'strategies': strategy,
            'missing_before': before_missing,
            'missing_after': after_missing,
            'imputation_details': imputation_details
        })
        
        return self.df
    
    def handle_outliers(self, method: str = 'tukey', columns: List[str] = None) -> pd.DataFrame:
        """
        Handle outliers using specified method.
        
        Args:
            method (str): Method to use ('tukey', 'zscore', 'cap', or 'drop')
            columns (List[str], optional): Columns to process. If None, use all numeric columns.
                
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        outlier_details = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            series = self.df[col].dropna()
            
            if method == 'tukey':
                # Tukey's method (1.5 * IQR)
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                outlier_details[col] = {
                    'method': 'tukey_capping',
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outliers_capped': int(outlier_count)
                }
                
            elif method == 'zscore':
                # Z-score method (cap at 3 standard deviations)
                mean = series.mean()
                std = series.std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                outlier_details[col] = {
                    'method': 'zscore_capping',
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outliers_capped': int(outlier_count)
                }
                
            elif method == 'cap':
                # Simple percentile capping (e.g., at 1% and 99%)
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                outlier_details[col] = {
                    'method': 'percentile_capping',
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outliers_capped': int(outlier_count)
                }
                
            elif method == 'drop':
                # Drop outlier rows
                if 'tukey' in self.assessment_report['outliers']:
                    q1 = self.assessment_report['outliers']['tukey'][col]['q1']
                    q3 = self.assessment_report['outliers']['tukey'][col]['q3']
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                else:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    before_rows = len(self.df)
                    self.df = self.df[~outlier_mask]
                    after_rows = len(self.df)
                
                outlier_details[col] = {
                    'method': 'outlier_removal',
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'rows_removed': int(outlier_count),
                    'rows_before': before_rows,
                    'rows_after': after_rows
                }
        
        # Log the action
        self.log_cleaning_step('handle_outliers', {
            'status': 'completed',
            'method': method,
            'columns': columns,
            'outlier_details': outlier_details
        })
        
        return self.df
    
    def handle_duplicates(self, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Handle duplicate rows in the dataset.
        
        Args:
            subset (List[str], optional): Columns to consider. If None, use all columns.
            keep (str): Which duplicates to keep ('first', 'last', or False for none)
                
        Returns:
            pd.DataFrame: DataFrame with handled duplicates
        """
        before_rows = len(self.df)
        
        if subset is None:
            duplicates = self.df.duplicated(keep=keep)
        else:
            duplicates = self.df.duplicated(subset=subset, keep=keep)
        
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            if keep is False:
                self.df = self.df[~self.df.duplicated(subset=subset, keep=False)]
            else:
                self.df = self.df[~duplicates]
        
        after_rows = len(self.df)
        
        # Log the action
        self.log_cleaning_step('handle_duplicates', {
            'status': 'completed',
            'subset': subset,
            'keep': keep,
            'duplicates_removed': duplicate_count,
            'rows_before': before_rows,
            'rows_after': after_rows
        })
        
        return self.df
    
    def handle_impossible_values(self, constraints: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Handle impossible values based on domain constraints.
        
        Args:
            constraints (Dict, optional): Mapping of column names to constraints
                (e.g., {'Distance': {'min': 0, 'max': 50}})
                
        Returns:
            pd.DataFrame: DataFrame with handled impossible values
        """
        if constraints is None:
            # Default constraints based on domain knowledge
            constraints = {
                'Distance': {'min': 0, 'max': 50},  # Distances should be positive and reasonable
                'Time': {'min': 0, 'max': 120}      # Time should be positive and reasonable
            }
        
        impossible_details = {}
        
        for col, rules in constraints.items():
            if col not in self.df.columns:
                continue
                
            changes = 0
            
            # Handle minimum constraint
            if 'min' in rules:
                min_val = rules['min']
                too_small_mask = self.df[col] < min_val
                too_small_count = too_small_mask.sum()
                
                if too_small_count > 0:
                    self.df.loc[too_small_mask, col] = min_val
                    changes += too_small_count
            
            # Handle maximum constraint
            if 'max' in rules:
                max_val = rules['max']
                too_large_mask = self.df[col] > max_val
                too_large_count = too_large_mask.sum()
                
                if too_large_count > 0:
                    self.df.loc[too_large_mask, col] = max_val
                    changes += too_large_count
            
            impossible_details[col] = {
                'constraints': rules,
                'values_modified': int(changes)
            }
        
        # Log the action
        self.log_cleaning_step('handle_impossible_values', {
            'status': 'completed',
            'constraints': constraints,
            'impossible_details': impossible_details
        })
        
        return self.df
    
    def apply_recommended_cleaning(self) -> pd.DataFrame:
        """
        Apply recommended cleaning actions based on the assessment report.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not self.assessment_report:
            return self.df
        
        # Apply cleaning steps based on recommendations
        
        # 1. Handle data types
        # Skip for now as it's more complex and might require careful handling
        
        # 2. Standardize mode values
        self.standardize_mode_values()
        
        # 3. Handle duplicates
        if self.assessment_report['duplicates']['count'] > 0:
            self.handle_duplicates(keep='first')
        
        if 'case_duplicates' in self.assessment_report and self.assessment_report['case_duplicates']['count'] > 0:
            self.handle_duplicates(subset=['Case'], keep='first')
        
        # 4. Handle missing values
        missing_significant = [col for col in self.df.columns 
                               if self.assessment_report['missing_values']['column_details'][col]['is_significant']]
        if missing_significant:
            strategies = {col: 'median' if col in ['Distance', 'Time'] else 'mode' for col in missing_significant}
            self.handle_missing_values(strategies)
        
        # 5. Handle impossible values
        self.handle_impossible_values()
        
        # 6. Handle outliers
        outlier_cols = [col for col, stats in self.assessment_report['outliers'].items() 
                       if 'outlier_count' in stats and stats['outlier_count'] > 0]
        if outlier_cols:
            self.handle_outliers(method='tukey', columns=outlier_cols)
        
        # Calculate final metrics
        self.after_metrics = self._calculate_metrics(self.df)
        
        return self.df
    
    def get_before_after_comparison(self) -> Dict[str, Any]:
        """
        Generate a comparison of metrics before and after cleaning.
        
        Returns:
            Dict: Comparison of before and after metrics
        """
        if self.after_metrics is None:
            self.after_metrics = self._calculate_metrics(self.df)
        
        # Calculate changes
        changes = {
            'row_count': {
                'before': self.before_metrics['row_count'],
                'after': self.after_metrics['row_count'],
                'change': self.after_metrics['row_count'] - self.before_metrics['row_count'],
                'percent_change': ((self.after_metrics['row_count'] - self.before_metrics['row_count']) / 
                                   self.before_metrics['row_count'] * 100) if self.before_metrics['row_count'] > 0 else 0
            },
            'missing_values': {
                'before': self.before_metrics['total_missing_values'],
                'after': self.after_metrics['total_missing_values'],
                'change': self.after_metrics['total_missing_values'] - self.before_metrics['total_missing_values'],
                'percent_change': ((self.after_metrics['total_missing_values'] - self.before_metrics['total_missing_values']) / 
                                   max(1, self.before_metrics['total_missing_values']) * 100)
            },
            'column_missing_values': {}
        }
        
        # Column-specific missing value changes
        for col in self.before_metrics['missing_value_counts']:
            if col in self.after_metrics['missing_value_counts']:
                before = self.before_metrics['missing_value_counts'][col]
                after = self.after_metrics['missing_value_counts'][col]
                changes['column_missing_values'][col] = {
                    'before': before,
                    'after': after,
                    'change': after - before,
                    'percent_change': ((after - before) / max(1, before) * 100) if before > 0 else 0
                }
        
        # Numeric column statistic changes
        if 'numeric_stats' in self.before_metrics:
            changes['numeric_stats'] = {}
            
            for col in self.before_metrics['numeric_stats']:
                if col in self.after_metrics.get('numeric_stats', {}):
                    changes['numeric_stats'][col] = {}
                    
                    for stat in ['mean', 'median', 'std', 'min', 'max']:
                        before = self.before_metrics['numeric_stats'][col].get(stat)
                        after = self.after_metrics['numeric_stats'][col].get(stat)
                        
                        if before is not None and after is not None:
                            changes['numeric_stats'][col][stat] = {
                                'before': before,
                                'after': after,
                                'change': after - before,
                                'percent_change': ((after - before) / abs(before) * 100) if before != 0 else 0
                            }
        
        # Mode value changes
        if 'mode_value_counts' in self.before_metrics and 'mode_value_counts' in self.after_metrics:
            changes['mode_value_counts'] = {
                'before': self.before_metrics['mode_value_counts'],
                'after': self.after_metrics['mode_value_counts']
            }
            
            # Combined set of all mode values (before and after)
            all_modes = set(self.before_metrics['mode_value_counts'].keys()).union(
                set(self.after_metrics['mode_value_counts'].keys())
            )
            
            mode_changes = {}
            for mode in all_modes:
                before = self.before_metrics['mode_value_counts'].get(mode, 0)
                after = self.after_metrics['mode_value_counts'].get(mode, 0)
                
                mode_changes[mode] = {
                    'before': before,
                    'after': after,
                    'change': after - before,
                    'percent_change': ((after - before) / max(1, before) * 100) if before > 0 else 0
                }
            
            changes['mode_value_counts']['changes'] = mode_changes
        
        # Add cleaning log
        comparison = {
            'metrics_comparison': changes,
            'cleaning_log': self.cleaning_log,
            'summary': {
                'cleaning_steps': len(self.cleaning_log),
                'rows_removed': self.before_metrics['row_count'] - self.after_metrics['row_count'],
                'missing_values_addressed': self.before_metrics['total_missing_values'] - self.after_metrics['total_missing_values']
            }
        }
        
        return comparison
    
    def save_cleaning_report(self, file_path: str) -> str:
        """
        Saves the cleaning report to a JSON file.
        
        Args:
            file_path (str): Path to save the report
            
        Returns:
            str: Success message
        """
        comparison = self.get_before_after_comparison()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(comparison, f, indent=2)
                
            return f"Cleaning report saved to {file_path}"
        except Exception as e:
            return f"Error saving cleaning report: {e}"
    
    def plot_before_after(self, output_dir: str = "plots", include_plots: List[str] = None) -> List[str]:
        """
        Generate before/after comparison plots.
        
        Args:
            output_dir (str): Directory to save plots
            include_plots (List[str], optional): Specific plots to include
                Options: 'missing', 'distribution', 'outliers', 'mode_counts'
                
        Returns:
            List[str]: Paths to saved plots
        """
        if include_plots is None:
            include_plots = ['missing', 'distribution', 'outliers', 'mode_counts']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        saved_plots = []
        
        # 1. Missing values comparison
        if 'missing' in include_plots:
            cols_with_missing = [col for col in self.original_df.columns 
                                if self.original_df[col].isna().sum() > 0]
            
            if cols_with_missing:
                plt.figure(figsize=(10, 6))
                
                before_missing = [self.original_df[col].isna().sum() for col in cols_with_missing]
                after_missing = [self.df[col].isna().sum() for col in cols_with_missing]
                
                x = np.arange(len(cols_with_missing))
                width = 0.35
                
                plt.bar(x - width/2, before_missing, width, label='Before Cleaning')
                plt.bar(x + width/2, after_missing, width, label='After Cleaning')
                
                plt.xlabel('Columns')
                plt.ylabel('Missing Value Count')
                plt.title('Missing Values Before and After Cleaning')
                plt.xticks(x, cols_with_missing, rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                missing_plot_path = os.path.join(output_dir, 'missing_values_comparison.png')
                plt.savefig(missing_plot_path)
                plt.close()
                saved_plots.append(missing_plot_path)
        
        # 2. Distribution comparison for Time and Distance
        if 'distribution' in include_plots:
            numeric_cols = ['Time', 'Distance']
            
            for col in numeric_cols:
                if col in self.original_df.columns and col in self.df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    plt.subplot(1, 2, 1)
                    sns.histplot(self.original_df[col].dropna(), kde=True)
                    plt.title(f'{col} Distribution - Before')
                    
                    plt.subplot(1, 2, 2)
                    sns.histplot(self.df[col].dropna(), kde=True)
                    plt.title(f'{col} Distribution - After')
                    
                    plt.tight_layout()
                    
                    dist_plot_path = os.path.join(output_dir, f'{col.lower()}_distribution_comparison.png')
                    plt.savefig(dist_plot_path)
                    plt.close()
                    saved_plots.append(dist_plot_path)
        
        # 3. Mode value counts comparison
        if 'mode_counts' in include_plots and 'Mode' in self.original_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Get mode counts
            before_modes = self.original_df['Mode'].value_counts().sort_index()
            after_modes = self.df['Mode'].value_counts().sort_index()
            
            # Combine all unique modes
            all_modes = sorted(set(before_modes.index) | set(after_modes.index))
            
            # Create DataFrame for plotting
            mode_counts = pd.DataFrame(index=all_modes, columns=['Before', 'After'])
            
            for mode in all_modes:
                mode_counts.loc[mode, 'Before'] = before_modes.get(mode, 0)
                mode_counts.loc[mode, 'After'] = after_modes.get(mode, 0)
            
            # Plot
            mode_counts.plot(kind='bar', figsize=(10, 6))
            plt.xlabel('Mode')
            plt.ylabel('Count')
            plt.title('Mode Value Counts Before and After Cleaning')
            plt.tight_layout()
            
            mode_plot_path = os.path.join(output_dir, 'mode_counts_comparison.png')
            plt.savefig(mode_plot_path)
            plt.close()
            saved_plots.append(mode_plot_path)
        
        # 4. Outlier comparison (boxplots)
        if 'outliers' in include_plots:
            numeric_cols = ['Time', 'Distance']
            
            for col in numeric_cols:
                if col in self.original_df.columns and col in self.df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Prepare data for plotting
                    before_data = self.original_df[col].dropna()
                    after_data = self.df[col].dropna()
                    
                    # Create DataFrame for boxplot
                    combined_data = pd.DataFrame({
                        'Value': pd.concat([before_data, after_data]),
                        'Stage': ['Before'] * len(before_data) + ['After'] * len(after_data),
                        'Column': [col] * (len(before_data) + len(after_data))
                    })
                    
                    # Plot
                    sns.boxplot(x='Column', y='Value', hue='Stage', data=combined_data)
                    plt.title(f'{col} Outlier Comparison')
                    plt.tight_layout()
                    
                    outlier_plot_path = os.path.join(output_dir, f'{col.lower()}_outlier_comparison.png')
                    plt.savefig(outlier_plot_path)
                    plt.close()
                    saved_plots.append(outlier_plot_path)
        
        return saved_plots


# Functions to easily use the classes

def assess_data_quality(df: pd.DataFrame, save_report: bool = True, 
                        report_path: str = 'reports/data_quality_report.json') -> Dict[str, Any]:
    """
    Perform comprehensive data quality assessment on a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to assess
        save_report (bool): Whether to save the report to a file
        report_path (str): Path to save the report (if save_report is True)
        
    Returns:
        Dict: Data quality assessment report
    """
    assessment = DataQualityAssessment(df)
    
    # Run all assessments
    assessment.verify_data_types()
    assessment.check_missing_values()
    assessment.detect_outliers_tukey()
    assessment.check_duplicates()
    if 'Case' in df.columns:
        assessment.check_duplicates(subset=['Case'])
    assessment.identify_impossible_values()
    assessment.check_distribution()
    
    # Generate report
    report = assessment.generate_report()
    
    if save_report:
        assessment.save_report(report_path)
    
    return report

def clean_data(df: pd.DataFrame, assessment_report: Dict[str, Any] = None, 
               save_report: bool = True, report_path: str = 'reports/cleaning_report.json',
               generate_plots: bool = True, plots_dir: str = 'plots') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a DataFrame based on assessment results.
    
    Args:
        df (pd.DataFrame): The DataFrame to clean
        assessment_report (Dict, optional): Data quality assessment report
        save_report (bool): Whether to save the cleaning report
        report_path (str): Path to save the cleaning report
        generate_plots (bool): Whether to generate before/after plots
        plots_dir (str): Directory to save plots
        
    Returns:
        Tuple: (Cleaned DataFrame, Cleaning report)
    """
    cleaner = DataCleaner(df, assessment_report)
    
    # Apply recommended cleaning
    cleaned_df = cleaner.apply_recommended_cleaning()
    
    # Generate comparison
    comparison = cleaner.get_before_after_comparison()
    
    if save_report:
        cleaner.save_cleaning_report(report_path)
    
    if generate_plots:
        cleaner.plot_before_after(output_dir=plots_dir)
    
    return cleaned_df, comparison