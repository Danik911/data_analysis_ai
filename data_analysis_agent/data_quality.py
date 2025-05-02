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
        Saves the data quality report to a JSON file using atomic write operation.
        
        Args:
            file_path (str): Path to save the report
            
        Returns:
            str: Success message
        """
        report = self.generate_report()
        
        try:
            # Import our utilities module with proper path handling
            import sys
            import os
            module_path = os.path.abspath(os.path.dirname(__file__))
            if module_path not in sys.path:
                sys.path.append(module_path)
            
            from utils import save_json_atomic
            
            # Use atomic JSON saving to prevent corruption
            success = save_json_atomic(report, file_path)
            
            if success:
                return f"Data quality report saved to {file_path}"
            else:
                return f"Error saving data quality report to {file_path}"
        except Exception as e:
            import traceback
            error_msg = f"Error saving data quality report: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg


class DataCleaner:
    """
    Class for comprehensive data cleaning, implementing the recommendations
    from the data quality assessment.
    """
    
    def __init__(self, df: pd.DataFrame, assessment_report: Optional[Dict[str, Any]] = None):
        """
        Initialize with a pandas DataFrame and optionally an assessment report.
        
        Args:
            df (pd.DataFrame): The dataset to clean
            assessment_report (Dict): Optional quality assessment report
        """
        self.df = df.copy()
        self.original_df = df.copy()  # Keep a copy of the original data for before/after comparison
        self.assessment_report = assessment_report
        self.cleaning_log = []
        self.report = {}
        
    def standardize_mode_values(self) -> None:
        """
        Standardizes mode values according to valid mode categories.
        """
        if 'Mode' not in self.df.columns:
            return
            
        # Standard mode values we expect
        valid_modes = {'Car', 'Bus', 'Cycle', 'Walk', 'Train', 'Tram', 'Subway'}
        
        # Track original values for the report
        original_values = self.df['Mode'].copy()
        
        # Count changes made
        changes = 0
        value_map = {}
        
        # Function to map invalid modes to valid ones based on first letter
        def get_correction(invalid_mode):
            if not isinstance(invalid_mode, str) or not invalid_mode:
                return "Unknown"
                
            first_letter = invalid_mode[0].upper()
            
            for mode in valid_modes:
                if mode.startswith(first_letter):
                    return mode
            
            return "Unknown"
        
        # Apply corrections
        for idx, value in enumerate(self.df['Mode']):
            if value not in valid_modes:
                corrected_value = get_correction(value)
                self.df.at[idx, 'Mode'] = corrected_value
                changes += 1
                value_map[value] = corrected_value
        
        # Log the cleaning action
        self.cleaning_log.append({
            'action': 'standardize_mode_values',
            'details': {
                'changes': changes,
                'value_map': value_map
            }
        })
    
    def handle_missing_values(self) -> None:
        """
        Handles missing values based on recommended strategies from the assessment.
        """
        # Track what was done to each column
        strategies = {}
        
        # Get columns with significant missing values if assessment is available
        missing_significant = []
        if self.assessment_report and 'recommendations' in self.assessment_report:
            missing_value_recs = self.assessment_report['recommendations'].get('missing_values', [])
            for rec in missing_value_recs:
                if 'columns with significant missing values' in rec:
                    # Extract column names from recommendation text
                    start_idx = rec.find(':') + 1
                    columns_str = rec[start_idx:].strip()
                    missing_significant = [col.strip() for col in columns_str.split(',')]
        else:
            # If no assessment, find columns with > 5% missing values
            for col in self.df.columns:
                pct_missing = (self.df[col].isna().sum() / len(self.df)) * 100
                if pct_missing > 5:
                    missing_significant.append(col)
        
        # Handle missing values by column
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count == 0:
                continue
                
            # Choose strategy based on column type and domain knowledge
            if col in ['Distance', 'Time']:
                # For numeric columns, use median imputation
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                strategies[col] = {'strategy': 'median_imputation', 'value': float(median_val)}
            
            elif col == 'Mode':
                # For Mode, use most common value
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                strategies[col] = {'strategy': 'mode_imputation', 'value': mode_val}
            
            else:
                # For other columns, use most common value
                mode_val = self.df[col].mode()[0] if not self.df[col].empty else "Unknown"
                self.df[col].fillna(mode_val, inplace=True)
                strategies[col] = {'strategy': 'mode_imputation', 'value': mode_val}
        
        # Log the cleaning action
        self.cleaning_log.append({
            'action': 'handle_missing_values',
            'details': {
                'strategies': strategies
            }
        })
    
    def handle_outliers(self, method='cap') -> None:
        """
        Handles outliers in numeric columns using specified method.
        
        Args:
            method (str): Method to use ('cap' or 'remove')
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        columns_processed = []
        
        for col in numeric_cols:
            # Skip ID columns or columns without outliers
            if col.lower() in ['id', 'case', 'index']:
                continue
                
            # Get outlier information from assessment if available
            outliers_info = {}
            if self.assessment_report and 'outliers_tukey' in self.assessment_report:
                outliers_info = self.assessment_report['outliers_tukey'].get(col, {})
            
            # If no outlier information, calculate it
            if not outliers_info:
                series = self.df[col].dropna()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            else:
                lower_bound = outliers_info.get('lower_bound')
                upper_bound = outliers_info.get('upper_bound')
            
            # Count outliers
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                if method == 'cap':
                    # Cap outliers at the bounds
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    columns_processed.append(col)
                elif method == 'remove':
                    # Remove rows with outliers
                    self.df = self.df[~outliers]
                    columns_processed.append(col)
        
        # Log the cleaning action
        self.cleaning_log.append({
            'action': 'handle_outliers',
            'details': {
                'method': method,
                'columns': columns_processed
            }
        })
    
    def handle_duplicates(self, subset=None) -> None:
        """
        Removes duplicate rows, optionally based on a subset of columns.
        
        Args:
            subset (List[str]): Optional list of columns to consider for duplicates
        """
        # Count duplicates before removal
        if subset:
            duplicates = self.df.duplicated(subset=subset)
        else:
            duplicates = self.df.duplicated()
            
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            # Remove duplicates
            self.df.drop_duplicates(subset=subset, keep='first', inplace=True)
            
            # Log the cleaning action
            self.cleaning_log.append({
                'action': 'handle_duplicates',
                'details': {
                    'duplicates_removed': int(duplicate_count),
                    'subset': subset
                }
            })
    
    def handle_impossible_values(self) -> None:
        """
        Fixes impossible values based on domain constraints.
        """
        # Define domain-specific rules
        constraints = {
            'Distance': {'min': 0, 'max': 50},  # Distances should be positive and reasonable (e.g., < 50km)
            'Time': {'min': 0, 'max': 120}      # Time should be positive and reasonable (e.g., < 120 minutes)
        }
        
        constraints_applied = {}
        
        for col, limits in constraints.items():
            if col not in self.df.columns:
                continue
                
            min_val = limits.get('min')
            max_val = limits.get('max')
            violations = 0
            
            if min_val is not None:
                too_small = self.df[col] < min_val
                violations += too_small.sum()
                if too_small.any():
                    self.df.loc[too_small, col] = min_val
            
            if max_val is not None:
                too_large = self.df[col] > max_val
                violations += too_large.sum()
                if too_large.any():
                    self.df.loc[too_large, col] = max_val
            
            if violations > 0:
                constraints_applied[col] = {'min': min_val, 'max': max_val, 'violations_fixed': int(violations)}
        
        if constraints_applied:
            # Log the cleaning action
            self.cleaning_log.append({
                'action': 'handle_impossible_values',
                'details': {
                    'constraints': constraints_applied
                }
            })
    
    def fix_data_types(self) -> None:
        """
        Converts columns to their appropriate data types based on assessment.
        """
        # Get type recommendations from assessment if available
        type_conversions = {}
        
        if self.assessment_report and 'recommendations' in self.assessment_report:
            type_recs = self.assessment_report['recommendations'].get('data_types', [])
            for rec in type_recs:
                if 'Convert' in rec:
                    parts = rec.split("'")
                    if len(parts) >= 3:
                        col_name = parts[1]
                        from_type = parts[3]
                        to_type = parts[5]
                        type_conversions[col_name] = {'from': from_type, 'to': to_type}
        
        # Apply conversions
        for col, conversion in type_conversions.items():
            if col not in self.df.columns:
                continue
                
            to_type = conversion.get('to')
            
            try:
                if to_type == 'int64':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
                elif to_type == 'float64':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Add more type conversions as needed
            except Exception as e:
                print(f"Error converting {col} to {to_type}: {str(e)}")
        
        if type_conversions:
            # Log the cleaning action
            self.cleaning_log.append({
                'action': 'fix_data_types',
                'details': {
                    'conversions': type_conversions
                }
            })
    
    def clean(self) -> pd.DataFrame:
        """
        Performs a complete cleaning of the dataset.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        # 1. Fix data types first to ensure proper handling
        self.fix_data_types()
        
        # 2. Handle duplicates
        self.handle_duplicates()
        
        # 3. Standardize categorical values (e.g. Mode)
        self.standardize_mode_values()
        
        # 4. Handle impossible values
        self.handle_impossible_values()
        
        # 5. Handle outliers
        self.handle_outliers(method='cap')
        
        # 6. Handle missing values last
        self.handle_missing_values()
        
        return self.df
    
    def generate_metrics_comparison(self) -> Dict[str, Any]:
        """
        Generates before/after metrics for the cleaning process.
        
        Returns:
            Dict: Metrics comparison
        """
        metrics = {}
        
        # Row count comparison
        metrics['row_count'] = {
            'before': len(self.original_df),
            'after': len(self.df),
            'change': len(self.df) - len(self.original_df)
        }
        
        # Missing values comparison
        original_missing = self.original_df.isna().sum().sum()
        new_missing = self.df.isna().sum().sum()
        metrics['missing_values'] = {
            'before': int(original_missing),
            'after': int(new_missing),
            'change': int(new_missing - original_missing)
        }
        
        # Statistics for numeric columns
        metrics['numeric_stats'] = {}
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col not in self.original_df.columns:
                continue
                
            col_metrics = {}
            
            # Mean comparison
            original_mean = self.original_df[col].mean()
            new_mean = self.df[col].mean()
            col_metrics['mean'] = {
                'before': float(original_mean),
                'after': float(new_mean),
                'change': float(new_mean - original_mean)
            }
            
            # Standard deviation comparison
            original_std = self.original_df[col].std()
            new_std = self.df[col].std()
            col_metrics['std'] = {
                'before': float(original_std),
                'after': float(new_std),
                'change': float(new_std - original_std)
            }
            
            # Min/Max comparison
            col_metrics['min'] = {
                'before': float(self.original_df[col].min()),
                'after': float(self.df[col].min())
            }
            col_metrics['max'] = {
                'before': float(self.original_df[col].max()),
                'after': float(self.df[col].max())
            }
            
            metrics['numeric_stats'][col] = col_metrics
        
        return metrics
    
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive report of the cleaning process.
        
        Returns:
            Dict: Cleaning report
        """
        metrics_comparison = self.generate_metrics_comparison()
        
        self.report = {
            'cleaning_log': self.cleaning_log,
            'metrics_comparison': metrics_comparison,
            'original_rows': len(self.original_df),
            'cleaned_rows': len(self.df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return self.report
    
    def save_report(self, file_path: str) -> bool:
        """
        Saves the cleaning report to a JSON file using atomic write operation.
        
        Args:
            file_path (str): Path to save the report
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.report:
            self.generate_cleaning_report()
            
        try:
            # Import our utilities module with proper path handling
            import sys
            import os
            module_path = os.path.abspath(os.path.dirname(__file__))
            if module_path not in sys.path:
                sys.path.append(module_path)
            
            from utils import save_json_atomic
            
            # Use atomic JSON saving to prevent corruption
            success = save_json_atomic(self.report, file_path)
            
            return success
        except Exception as e:
            import traceback
            print(f"Error saving cleaning report: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_comparison_plots(self, plots_dir: str = 'plots/cleaning_comparisons') -> List[str]:
        """
        Generates before/after comparison plots for the cleaning process.
        
        Args:
            plots_dir (str): Directory to save the plots
            
        Returns:
            List[str]: Paths to the generated plots
        """
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = []
        
        # Generate distplots for numerical columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col not in self.original_df.columns:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot before histogram
            sns.histplot(self.original_df[col].dropna(), 
                         color='red', alpha=0.5, label='Before', ax=ax, kde=True)
            
            # Plot after histogram
            sns.histplot(self.df[col].dropna(), 
                         color='blue', alpha=0.5, label='After', ax=ax, kde=True)
            
            ax.set_title(f'Before/After: {col} Distribution')
            ax.legend()
            
            # Save the plot
            plot_path = os.path.join(plots_dir, f'{col}_comparison.png')
            plt.savefig(plot_path)
            plt.close(fig)
            
            plot_paths.append(plot_path)
            
        # For Mode column, generate countplot
        if 'Mode' in self.df.columns and 'Mode' in self.original_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot before counts
            sns.countplot(x='Mode', data=self.original_df, ax=axes[0])
            axes[0].set_title('Mode Values: Before')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Plot after counts
            sns.countplot(x='Mode', data=self.df, ax=axes[1])
            axes[1].set_title('Mode Values: After')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(plots_dir, 'mode_comparison.png')
            plt.savefig(plot_path)
            plt.close(fig)
            
            plot_paths.append(plot_path)
            
        return plot_paths


def assess_data_quality(df: pd.DataFrame, save_report: bool = False, 
                       report_path: str = 'reports/data_quality_report.json') -> Dict[str, Any]:
    """
    Performs comprehensive data quality assessment on a DataFrame.
    
    Args:
        df (pd.DataFrame): The dataset to assess
        save_report (bool): Whether to save the report to a file
        report_path (str): Path to save the report
        
    Returns:
        Dict: Data quality assessment report
    """
    assessor = DataQualityAssessment(df)
    assessment_report = assessor.generate_report()
    
    if save_report:
        assessor.save_report(report_path)
        
    return assessment_report


def clean_data(df: pd.DataFrame, assessment_report: Optional[Dict[str, Any]] = None, 
              save_report: bool = False, report_path: str = 'reports/cleaning_report.json',
              generate_plots: bool = False, plots_dir: str = 'plots/cleaning_comparisons') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive cleaning on a DataFrame based on quality assessment.
    
    Args:
        df (pd.DataFrame): The dataset to clean
        assessment_report (Dict): Optional quality assessment report
        save_report (bool): Whether to save the cleaning report to a file
        report_path (str): Path to save the report
        generate_plots (bool): Whether to generate before/after comparison plots
        plots_dir (str): Directory to save the plots
        
    Returns:
        Tuple[pd.DataFrame, Dict]: The cleaned DataFrame and cleaning report
    """
    cleaner = DataCleaner(df, assessment_report)
    cleaned_df = cleaner.clean()
    cleaning_report = cleaner.generate_cleaning_report()
    
    if generate_plots:
        cleaner.generate_comparison_plots(plots_dir)
    
    if save_report:
        cleaner.save_report(report_path)
        
    return cleaned_df, cleaning_report
