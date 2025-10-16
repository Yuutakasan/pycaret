"""
Data Quality Validation System

This module provides comprehensive data quality validation including:
- Missing data detection and imputation strategies
- Outlier identification and handling
- Data consistency checks across stores
- Temporal consistency validation
- Business rule validation
- Data completeness reports
- Quality score calculation per store
- Interactive quality dashboard

Author: PyCaret Contributors
License: MIT
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


class ImputationStrategy(Enum):
    """Available imputation strategies for missing data."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    ZERO = "zero"
    DROP = "drop"
    KNN = "knn"
    SEASONAL = "seasonal"


class OutlierMethod(Enum):
    """Available outlier detection methods."""
    IQR = "iqr"
    Z_SCORE = "zscore"
    MODIFIED_Z_SCORE = "modified_zscore"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    LOCAL_OUTLIER_FACTOR = "lof"


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationRule:
    """Business rule for data validation."""
    name: str
    description: str
    rule_fn: Callable
    severity: str = "error"  # error, warning, info
    enabled: bool = True


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset or store."""
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    validity_score: float = 0.0
    accuracy_score: float = 0.0
    timeliness_score: float = 0.0
    uniqueness_score: float = 0.0
    overall_score: float = 0.0

    missing_count: int = 0
    missing_percentage: float = 0.0
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    duplicate_count: int = 0
    invalid_count: int = 0

    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'scores': {
                'completeness': self.completeness_score,
                'consistency': self.consistency_score,
                'validity': self.validity_score,
                'accuracy': self.accuracy_score,
                'timeliness': self.timeliness_score,
                'uniqueness': self.uniqueness_score,
                'overall': self.overall_score
            },
            'metrics': {
                'missing_count': self.missing_count,
                'missing_percentage': self.missing_percentage,
                'outlier_count': self.outlier_count,
                'outlier_percentage': self.outlier_percentage,
                'duplicate_count': self.duplicate_count,
                'invalid_count': self.invalid_count
            },
            'issues': self.issues,
            'warnings': self.warnings
        }


class DataQualityValidator:
    """
    Comprehensive data quality validation system.

    Features:
    - Missing data detection and imputation
    - Outlier identification and handling
    - Cross-store consistency checks
    - Temporal consistency validation
    - Business rule validation
    - Quality scoring and reporting

    Examples
    --------
    >>> validator = DataQualityValidator()
    >>>
    >>> # Add business rules
    >>> validator.add_rule(
    ...     name="no_negative_sales",
    ...     description="Sales should not be negative",
    ...     rule_fn=lambda df: df['sales'] >= 0
    ... )
    >>>
    >>> # Validate data
    >>> metrics = validator.validate(df, store_id='store_001')
    >>> print(f"Quality Score: {metrics.overall_score:.2f}")
    >>>
    >>> # Generate report
    >>> report = validator.generate_report()
    >>> validator.save_report(report, 'data_quality_report.html')
    """

    def __init__(
        self,
        missing_threshold: float = 0.3,
        outlier_threshold: float = 3.0,
        duplicate_subset: Optional[List[str]] = None,
        temporal_column: str = 'date',
        store_column: str = 'store',
        verbose: bool = True
    ):
        """
        Initialize the data quality validator.

        Parameters
        ----------
        missing_threshold : float, default=0.3
            Maximum acceptable proportion of missing values (0-1)
        outlier_threshold : float, default=3.0
            Z-score threshold for outlier detection
        duplicate_subset : list, optional
            Column subset to check for duplicates
        temporal_column : str, default='date'
            Name of the temporal column
        store_column : str, default='store'
            Name of the store identifier column
        verbose : bool, default=True
            Print validation progress
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.duplicate_subset = duplicate_subset
        self.temporal_column = temporal_column
        self.store_column = store_column
        self.verbose = verbose

        self.validation_rules: List[ValidationRule] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.store_metrics: Dict[str, QualityMetrics] = {}

        # Initialize default business rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default business validation rules."""
        # Rule: No negative sales
        self.add_rule(
            name="no_negative_sales",
            description="Sales values should not be negative",
            rule_fn=lambda df: df.get('sales', pd.Series([True])) >= 0,
            severity="error"
        )

        # Rule: No negative inventory
        self.add_rule(
            name="no_negative_inventory",
            description="Inventory values should not be negative",
            rule_fn=lambda df: df.get('inventory', pd.Series([True])) >= 0,
            severity="error"
        )

        # Rule: No negative prices
        self.add_rule(
            name="no_negative_prices",
            description="Price values should not be negative",
            rule_fn=lambda df: df.get('price', pd.Series([True])) >= 0,
            severity="error"
        )

        # Rule: Reasonable dates
        self.add_rule(
            name="reasonable_dates",
            description="Dates should be within reasonable range",
            rule_fn=lambda df: self._validate_date_range(df),
            severity="warning"
        )

    def _validate_date_range(self, df: pd.DataFrame) -> pd.Series:
        """Validate that dates are within reasonable range."""
        if self.temporal_column not in df.columns:
            return pd.Series([True] * len(df))

        date_col = pd.to_datetime(df[self.temporal_column], errors='coerce')
        current_date = pd.Timestamp.now()
        min_valid_date = current_date - pd.Timedelta(days=3650)  # 10 years ago
        max_valid_date = current_date + pd.Timedelta(days=365)   # 1 year future

        return (date_col >= min_valid_date) & (date_col <= max_valid_date)

    def add_rule(
        self,
        name: str,
        description: str,
        rule_fn: Callable,
        severity: str = "error",
        enabled: bool = True
    ):
        """
        Add a custom validation rule.

        Parameters
        ----------
        name : str
            Rule name
        description : str
            Rule description
        rule_fn : callable
            Function that takes DataFrame and returns boolean Series
        severity : str, default='error'
            Severity level: 'error', 'warning', or 'info'
        enabled : bool, default=True
            Whether rule is enabled
        """
        rule = ValidationRule(
            name=name,
            description=description,
            rule_fn=rule_fn,
            severity=severity,
            enabled=enabled
        )
        self.validation_rules.append(rule)

        if self.verbose:
            print(f"✓ Added validation rule: {name}")

    def detect_missing_data(
        self,
        df: pd.DataFrame,
        store_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect missing data and analyze patterns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        store_id : str, optional
            Store identifier for tracking

        Returns
        -------
        dict
            Missing data analysis results
        """
        results = {
            'total_cells': df.size,
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns': {},
            'patterns': []
        }

        # Per-column analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            results['columns'][col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct),
                'data_type': str(df[col].dtype),
                'exceeds_threshold': missing_pct > (self.missing_threshold * 100)
            }

        # Detect missing patterns
        if results['missing_cells'] > 0:
            # Check for systematic missingness
            missing_matrix = df.isnull()

            # Rows with all values missing
            all_missing_rows = missing_matrix.all(axis=1).sum()
            if all_missing_rows > 0:
                results['patterns'].append({
                    'type': 'all_missing_rows',
                    'count': int(all_missing_rows),
                    'description': f'{all_missing_rows} rows with all values missing'
                })

            # Columns with >50% missing
            high_missing_cols = [
                col for col, info in results['columns'].items()
                if info['missing_percentage'] > 50
            ]
            if high_missing_cols:
                results['patterns'].append({
                    'type': 'high_missing_columns',
                    'columns': high_missing_cols,
                    'description': f'{len(high_missing_cols)} columns with >50% missing values'
                })

        return results

    def impute_missing_data(
        self,
        df: pd.DataFrame,
        strategy: Union[str, ImputationStrategy] = ImputationStrategy.MEDIAN,
        column_strategies: Optional[Dict[str, ImputationStrategy]] = None
    ) -> pd.DataFrame:
        """
        Impute missing data using specified strategies.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with missing values
        strategy : str or ImputationStrategy, default='median'
            Default imputation strategy
        column_strategies : dict, optional
            Column-specific imputation strategies

        Returns
        -------
        pd.DataFrame
            Data with imputed values
        """
        df_imputed = df.copy()

        if isinstance(strategy, str):
            strategy = ImputationStrategy(strategy)

        column_strategies = column_strategies or {}

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            col_strategy = column_strategies.get(col, strategy)
            if isinstance(col_strategy, str):
                col_strategy = ImputationStrategy(col_strategy)

            if self.verbose:
                print(f"Imputing {col} using {col_strategy.value} strategy")

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if col_strategy == ImputationStrategy.MEAN:
                    df_imputed[col].fillna(df[col].mean(), inplace=True)
                elif col_strategy == ImputationStrategy.MEDIAN:
                    df_imputed[col].fillna(df[col].median(), inplace=True)
                elif col_strategy == ImputationStrategy.ZERO:
                    df_imputed[col].fillna(0, inplace=True)
                elif col_strategy == ImputationStrategy.FORWARD_FILL:
                    df_imputed[col].fillna(method='ffill', inplace=True)
                elif col_strategy == ImputationStrategy.BACKWARD_FILL:
                    df_imputed[col].fillna(method='bfill', inplace=True)
                elif col_strategy == ImputationStrategy.INTERPOLATE:
                    df_imputed[col] = df_imputed[col].interpolate(method='linear')
                elif col_strategy == ImputationStrategy.SEASONAL:
                    # Seasonal imputation (use median of same period)
                    if self.temporal_column in df.columns:
                        df_imputed[col] = self._seasonal_impute(df, col)

            # Categorical columns
            else:
                if col_strategy == ImputationStrategy.MODE:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df_imputed[col].fillna(mode_val[0], inplace=True)
                elif col_strategy == ImputationStrategy.FORWARD_FILL:
                    df_imputed[col].fillna(method='ffill', inplace=True)
                elif col_strategy == ImputationStrategy.BACKWARD_FILL:
                    df_imputed[col].fillna(method='bfill', inplace=True)

        return df_imputed

    def _seasonal_impute(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Impute using seasonal patterns."""
        result = df[col].copy()

        if self.temporal_column not in df.columns:
            return result.fillna(result.median())

        df_temp = df.copy()
        df_temp['_date'] = pd.to_datetime(df_temp[self.temporal_column])
        df_temp['_dayofweek'] = df_temp['_date'].dt.dayofweek
        df_temp['_month'] = df_temp['_date'].dt.month

        # Fill missing values with median of same day of week
        for idx in df_temp[df_temp[col].isnull()].index:
            day = df_temp.loc[idx, '_dayofweek']
            month = df_temp.loc[idx, '_month']

            # Try same day of week and month
            similar = df_temp[
                (df_temp['_dayofweek'] == day) &
                (df_temp['_month'] == month) &
                df_temp[col].notna()
            ]

            if len(similar) > 0:
                result.loc[idx] = similar[col].median()
            else:
                # Fall back to same day of week
                similar = df_temp[(df_temp['_dayofweek'] == day) & df_temp[col].notna()]
                if len(similar) > 0:
                    result.loc[idx] = similar[col].median()

        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: Union[str, OutlierMethod] = OutlierMethod.IQR,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        method : str or OutlierMethod, default='iqr'
            Outlier detection method
        columns : list, optional
            Columns to check (default: all numeric)

        Returns
        -------
        dict
            Outlier detection results
        """
        if isinstance(method, str):
            method = OutlierMethod(method)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            'method': method.value,
            'total_outliers': 0,
            'columns': {},
            'outlier_indices': set()
        }

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            data = df[col].dropna()

            if method == OutlierMethod.IQR:
                outliers = self._detect_outliers_iqr(data)
            elif method == OutlierMethod.Z_SCORE:
                outliers = self._detect_outliers_zscore(data, threshold=self.outlier_threshold)
            elif method == OutlierMethod.MODIFIED_Z_SCORE:
                outliers = self._detect_outliers_modified_zscore(data)
            else:
                outliers = pd.Series([False] * len(data), index=data.index)

            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(data)) * 100

            results['columns'][col] = {
                'outlier_count': int(outlier_count),
                'outlier_percentage': float(outlier_pct),
                'outlier_indices': outliers[outliers].index.tolist()
            }

            results['total_outliers'] += outlier_count
            results['outlier_indices'].update(outliers[outliers].index.tolist())

        results['outlier_indices'] = list(results['outlier_indices'])

        return results

    def _detect_outliers_iqr(self, data: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (data < lower_bound) | (data > upper_bound)

    def _detect_outliers_zscore(
        self,
        data: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=data.index)

    def _detect_outliers_modified_zscore(
        self,
        data: pd.Series,
        threshold: float = 3.5
    ) -> pd.Series:
        """Detect outliers using Modified Z-score method."""
        median = data.median()
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return pd.Series([False] * len(data), index=data.index)

        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'cap',
        detection_method: Union[str, OutlierMethod] = OutlierMethod.IQR,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle outliers using specified method.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        method : str, default='cap'
            Handling method: 'cap', 'remove', 'transform'
        detection_method : str or OutlierMethod, default='iqr'
            Detection method
        columns : list, optional
            Columns to process

        Returns
        -------
        pd.DataFrame
            Data with handled outliers
        """
        df_clean = df.copy()
        outlier_info = self.detect_outliers(df, method=detection_method, columns=columns)

        for col, info in outlier_info['columns'].items():
            if info['outlier_count'] == 0:
                continue

            if method == 'cap':
                # Cap outliers at boundaries
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == 'remove':
                # Remove outlier rows
                df_clean = df_clean.drop(info['outlier_indices'])

            elif method == 'transform':
                # Log transform to reduce outlier impact
                if (df[col] > 0).all():
                    df_clean[col] = np.log1p(df[col])

            if self.verbose:
                print(f"Handled {info['outlier_count']} outliers in {col} using {method}")

        return df_clean

    def check_consistency(
        self,
        df: pd.DataFrame,
        store_id: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Check data consistency within and across stores.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        store_id : str, optional
            Store identifier
        reference_data : pd.DataFrame, optional
            Reference data for cross-store comparison

        Returns
        -------
        dict
            Consistency check results
        """
        results = {
            'store_id': store_id,
            'checks': [],
            'passed': 0,
            'failed': 0
        }

        # Check 1: Data type consistency
        type_check = self._check_dtype_consistency(df)
        results['checks'].append(type_check)
        if type_check['passed']:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Check 2: Value range consistency
        range_check = self._check_value_ranges(df)
        results['checks'].append(range_check)
        if range_check['passed']:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Check 3: Temporal consistency
        if self.temporal_column in df.columns:
            temporal_check = self._check_temporal_consistency(df)
            results['checks'].append(temporal_check)
            if temporal_check['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1

        # Check 4: Cross-store consistency
        if reference_data is not None:
            cross_check = self._check_cross_store_consistency(df, reference_data)
            results['checks'].append(cross_check)
            if cross_check['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def _check_dtype_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check that data types are consistent and appropriate."""
        issues = []

        for col in df.columns:
            dtype = df[col].dtype

            # Check for object columns that could be numeric
            if dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    issues.append(f"Column '{col}' is object but could be numeric")
                except (ValueError, TypeError):
                    pass

            # Check for mixed types
            if dtype == 'object':
                types = df[col].apply(type).unique()
                if len(types) > 1:
                    issues.append(f"Column '{col}' has mixed types: {types}")

        return {
            'check': 'data_type_consistency',
            'passed': len(issues) == 0,
            'issues': issues
        }

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check that values are within expected ranges."""
        issues = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Column '{col}' has {inf_count} infinite values")

            # Check for extremely large values
            if df[col].max() > 1e10:
                issues.append(f"Column '{col}' has suspiciously large values (max: {df[col].max():.2e})")

        return {
            'check': 'value_range_consistency',
            'passed': len(issues) == 0,
            'issues': issues
        }

    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal data consistency."""
        issues = []

        try:
            dates = pd.to_datetime(df[self.temporal_column], errors='coerce')

            # Check for invalid dates
            invalid_dates = dates.isnull().sum()
            if invalid_dates > 0:
                issues.append(f"{invalid_dates} invalid dates in '{self.temporal_column}'")

            # Check for date order
            if not dates.is_monotonic_increasing:
                issues.append(f"Dates in '{self.temporal_column}' are not in chronological order")

            # Check for duplicate dates (if store column exists)
            if self.store_column in df.columns:
                duplicates = df.groupby([self.store_column, self.temporal_column]).size()
                dup_count = (duplicates > 1).sum()
                if dup_count > 0:
                    issues.append(f"{dup_count} duplicate date-store combinations")

            # Check for gaps in time series
            date_diffs = dates.diff().dropna()
            if len(date_diffs) > 0:
                mode_diff = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else pd.Timedelta(days=1)
                gaps = (date_diffs > mode_diff * 2).sum()
                if gaps > 0:
                    issues.append(f"{gaps} potential gaps in time series")

        except Exception as e:
            issues.append(f"Error checking temporal consistency: {str(e)}")

        return {
            'check': 'temporal_consistency',
            'passed': len(issues) == 0,
            'issues': issues
        }

    def _check_cross_store_consistency(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check consistency across stores."""
        issues = []

        # Check column alignment
        if set(df.columns) != set(reference_df.columns):
            missing = set(reference_df.columns) - set(df.columns)
            extra = set(df.columns) - set(reference_df.columns)
            if missing:
                issues.append(f"Missing columns: {missing}")
            if extra:
                issues.append(f"Extra columns: {extra}")

        # Check data type alignment
        common_cols = set(df.columns) & set(reference_df.columns)
        for col in common_cols:
            if df[col].dtype != reference_df[col].dtype:
                issues.append(
                    f"Column '{col}' dtype mismatch: "
                    f"{df[col].dtype} vs {reference_df[col].dtype}"
                )

        return {
            'check': 'cross_store_consistency',
            'passed': len(issues) == 0,
            'issues': issues
        }

    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate business rules.

        Parameters
        ----------
        df : pd.DataFrame
            Input data

        Returns
        -------
        dict
            Business rule validation results
        """
        results = {
            'total_rules': len(self.validation_rules),
            'enabled_rules': sum(1 for r in self.validation_rules if r.enabled),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'rules': []
        }

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            try:
                validation_result = rule.rule_fn(df)
                violations = (~validation_result).sum()

                rule_result = {
                    'name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity,
                    'passed': violations == 0,
                    'violations': int(violations),
                    'violation_percentage': float((violations / len(df)) * 100)
                }

                if violations == 0:
                    results['passed'] += 1
                elif rule.severity == 'error':
                    results['failed'] += 1
                else:
                    results['warnings'] += 1

                results['rules'].append(rule_result)

                if self.verbose and violations > 0:
                    print(f"⚠ Rule '{rule.name}' failed: {violations} violations ({rule.severity})")

            except Exception as e:
                results['rules'].append({
                    'name': rule.name,
                    'description': rule.description,
                    'severity': 'error',
                    'passed': False,
                    'error': str(e)
                })
                results['failed'] += 1

        return results

    def calculate_quality_score(
        self,
        df: pd.DataFrame,
        store_id: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> QualityMetrics:
        """
        Calculate comprehensive quality score.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        store_id : str, optional
            Store identifier
        weights : dict, optional
            Custom dimension weights

        Returns
        -------
        QualityMetrics
            Comprehensive quality metrics
        """
        if weights is None:
            weights = {
                'completeness': 0.25,
                'consistency': 0.20,
                'validity': 0.20,
                'accuracy': 0.15,
                'timeliness': 0.10,
                'uniqueness': 0.10
            }

        metrics = QualityMetrics()

        # 1. Completeness Score
        missing_info = self.detect_missing_data(df, store_id)
        metrics.missing_count = missing_info['missing_cells']
        metrics.missing_percentage = missing_info['missing_percentage']
        metrics.completeness_score = max(0, 100 - missing_info['missing_percentage'])

        # 2. Consistency Score
        consistency_info = self.check_consistency(df, store_id)
        total_checks = consistency_info['passed'] + consistency_info['failed']
        metrics.consistency_score = (
            (consistency_info['passed'] / total_checks * 100)
            if total_checks > 0 else 100
        )

        # 3. Validity Score (Business Rules)
        validity_info = self.validate_business_rules(df)
        total_rules = validity_info['passed'] + validity_info['failed']
        metrics.validity_score = (
            (validity_info['passed'] / total_rules * 100)
            if total_rules > 0 else 100
        )
        metrics.invalid_count = validity_info['failed']

        # 4. Accuracy Score (Outlier-based)
        outlier_info = self.detect_outliers(df)
        metrics.outlier_count = outlier_info['total_outliers']
        metrics.outlier_percentage = (
            (outlier_info['total_outliers'] / len(df)) * 100
            if len(df) > 0 else 0
        )
        metrics.accuracy_score = max(0, 100 - metrics.outlier_percentage)

        # 5. Timeliness Score
        if self.temporal_column in df.columns:
            try:
                dates = pd.to_datetime(df[self.temporal_column], errors='coerce')
                latest_date = dates.max()
                current_date = pd.Timestamp.now()
                days_old = (current_date - latest_date).days

                # Penalize old data
                metrics.timeliness_score = max(0, 100 - (days_old / 30) * 10)
            except Exception:
                metrics.timeliness_score = 50
        else:
            metrics.timeliness_score = 100

        # 6. Uniqueness Score
        duplicate_subset = self.duplicate_subset or df.columns.tolist()
        duplicates = df.duplicated(subset=duplicate_subset).sum()
        metrics.duplicate_count = duplicates
        metrics.uniqueness_score = (
            max(0, 100 - (duplicates / len(df)) * 100)
            if len(df) > 0 else 100
        )

        # Overall Score (weighted average)
        metrics.overall_score = (
            weights['completeness'] * metrics.completeness_score +
            weights['consistency'] * metrics.consistency_score +
            weights['validity'] * metrics.validity_score +
            weights['accuracy'] * metrics.accuracy_score +
            weights['timeliness'] * metrics.timeliness_score +
            weights['uniqueness'] * metrics.uniqueness_score
        )

        # Collect issues
        if metrics.completeness_score < 80:
            metrics.issues.append({
                'dimension': 'completeness',
                'severity': 'high',
                'message': f'High missing data: {metrics.missing_percentage:.1f}%'
            })

        if metrics.validity_score < 80:
            metrics.issues.append({
                'dimension': 'validity',
                'severity': 'high',
                'message': f'Business rule violations: {metrics.invalid_count}'
            })

        if metrics.uniqueness_score < 90:
            metrics.issues.append({
                'dimension': 'uniqueness',
                'severity': 'medium',
                'message': f'Duplicate records: {metrics.duplicate_count}'
            })

        return metrics

    def validate(
        self,
        df: pd.DataFrame,
        store_id: Optional[str] = None,
        save_metrics: bool = True
    ) -> QualityMetrics:
        """
        Run comprehensive validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        store_id : str, optional
            Store identifier
        save_metrics : bool, default=True
            Save metrics to history

        Returns
        -------
        QualityMetrics
            Quality metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting data quality validation{' for ' + store_id if store_id else ''}")
            print(f"{'='*60}\n")

        metrics = self.calculate_quality_score(df, store_id)

        if save_metrics:
            if store_id:
                self.store_metrics[store_id] = metrics

            self.validation_history.append({
                'timestamp': datetime.now(),
                'store_id': store_id,
                'metrics': metrics.to_dict(),
                'record_count': len(df)
            })

        if self.verbose:
            self._print_quality_summary(metrics, store_id)

        return metrics

    def _print_quality_summary(self, metrics: QualityMetrics, store_id: Optional[str] = None):
        """Print quality validation summary."""
        print(f"\n{'='*60}")
        print(f"Quality Validation Summary{' - ' + store_id if store_id else ''}")
        print(f"{'='*60}\n")

        print(f"Overall Quality Score: {metrics.overall_score:.2f}/100")
        print(f"\nDimension Scores:")
        print(f"  • Completeness:  {metrics.completeness_score:6.2f}/100")
        print(f"  • Consistency:   {metrics.consistency_score:6.2f}/100")
        print(f"  • Validity:      {metrics.validity_score:6.2f}/100")
        print(f"  • Accuracy:      {metrics.accuracy_score:6.2f}/100")
        print(f"  • Timeliness:    {metrics.timeliness_score:6.2f}/100")
        print(f"  • Uniqueness:    {metrics.uniqueness_score:6.2f}/100")

        print(f"\nKey Metrics:")
        print(f"  • Missing Data:  {metrics.missing_percentage:.2f}%")
        print(f"  • Outliers:      {metrics.outlier_percentage:.2f}%")
        print(f"  • Duplicates:    {metrics.duplicate_count}")
        print(f"  • Violations:    {metrics.invalid_count}")

        if metrics.issues:
            print(f"\n⚠ Issues Found ({len(metrics.issues)}):")
            for issue in metrics.issues[:5]:  # Show top 5
                print(f"  • [{issue['severity'].upper()}] {issue['message']}")

        print(f"\n{'='*60}\n")

    def generate_report(
        self,
        include_stores: Optional[List[str]] = None,
        format: str = 'dict'
    ) -> Union[Dict[str, Any], pd.DataFrame, str]:
        """
        Generate comprehensive quality report.

        Parameters
        ----------
        include_stores : list, optional
            Store IDs to include in report
        format : str, default='dict'
            Output format: 'dict', 'dataframe', 'html'

        Returns
        -------
        dict or DataFrame or str
            Quality report in specified format
        """
        if include_stores:
            stores_to_report = {
                k: v for k, v in self.store_metrics.items()
                if k in include_stores
            }
        else:
            stores_to_report = self.store_metrics

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_stores': len(stores_to_report),
            'total_validations': len(self.validation_history),
            'stores': {}
        }

        # Aggregate statistics
        if stores_to_report:
            scores = [m.overall_score for m in stores_to_report.values()]
            report['summary'] = {
                'average_quality_score': np.mean(scores),
                'min_quality_score': np.min(scores),
                'max_quality_score': np.max(scores),
                'std_quality_score': np.std(scores)
            }

        # Per-store details
        for store_id, metrics in stores_to_report.items():
            report['stores'][store_id] = metrics.to_dict()

        if format == 'dataframe':
            return self._report_to_dataframe(report)
        elif format == 'html':
            return self._report_to_html(report)
        else:
            return report

    def _report_to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Convert report to DataFrame."""
        rows = []

        for store_id, data in report['stores'].items():
            row = {'store_id': store_id}
            row.update(data['scores'])
            row.update(data['metrics'])
            rows.append(row)

        return pd.DataFrame(rows)

    def _report_to_html(self, report: Dict[str, Any]) -> str:
        """Convert report to HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .score-high {{ color: green; font-weight: bold; }}
                .score-medium {{ color: orange; font-weight: bold; }}
                .score-low {{ color: red; font-weight: bold; }}
                .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <div class="summary">
                <h3>Summary</h3>
                <p><strong>Generated:</strong> {report['generated_at']}</p>
                <p><strong>Total Stores:</strong> {report['total_stores']}</p>
                <p><strong>Total Validations:</strong> {report['total_validations']}</p>
        """

        if 'summary' in report:
            html += f"""
                <p><strong>Average Quality Score:</strong> {report['summary']['average_quality_score']:.2f}</p>
                <p><strong>Score Range:</strong> {report['summary']['min_quality_score']:.2f} - {report['summary']['max_quality_score']:.2f}</p>
            </div>
            """

        html += """
            <h2>Store Quality Metrics</h2>
            <table>
                <tr>
                    <th>Store ID</th>
                    <th>Overall Score</th>
                    <th>Completeness</th>
                    <th>Consistency</th>
                    <th>Validity</th>
                    <th>Missing %</th>
                    <th>Outliers</th>
                    <th>Duplicates</th>
                </tr>
        """

        for store_id, data in report['stores'].items():
            scores = data['scores']
            metrics = data['metrics']

            score_class = (
                'score-high' if scores['overall'] >= 80
                else 'score-medium' if scores['overall'] >= 60
                else 'score-low'
            )

            html += f"""
                <tr>
                    <td>{store_id}</td>
                    <td class="{score_class}">{scores['overall']:.2f}</td>
                    <td>{scores['completeness']:.2f}</td>
                    <td>{scores['consistency']:.2f}</td>
                    <td>{scores['validity']:.2f}</td>
                    <td>{metrics['missing_percentage']:.2f}%</td>
                    <td>{metrics['outlier_count']}</td>
                    <td>{metrics['duplicate_count']}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def save_report(self, report: str, filepath: str):
        """Save HTML report to file."""
        with open(filepath, 'w') as f:
            f.write(report)

        if self.verbose:
            print(f"✓ Report saved to {filepath}")

    def get_validation_history(
        self,
        store_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get validation history.

        Parameters
        ----------
        store_id : str, optional
            Filter by store ID
        limit : int, optional
            Maximum number of records

        Returns
        -------
        list
            Validation history records
        """
        history = self.validation_history

        if store_id:
            history = [h for h in history if h.get('store_id') == store_id]

        if limit:
            history = history[-limit:]

        return history


def create_quality_dashboard(
    validator: DataQualityValidator,
    output_file: str = 'data_quality_dashboard.html'
):
    """
    Create interactive quality dashboard.

    Parameters
    ----------
    validator : DataQualityValidator
        Validator instance with metrics
    output_file : str, default='data_quality_dashboard.html'
        Output file path
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("⚠ Plotly not installed. Install with: pip install plotly")
        return

    if not validator.store_metrics:
        print("⚠ No store metrics available. Run validation first.")
        return

    # Prepare data
    stores = list(validator.store_metrics.keys())
    metrics_list = list(validator.store_metrics.values())

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Overall Quality Scores by Store',
            'Quality Dimensions Heatmap',
            'Missing Data Distribution',
            'Outlier Distribution',
            'Quality Score Trends',
            'Issue Summary'
        ),
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "box"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )

    # 1. Overall quality scores
    overall_scores = [m.overall_score for m in metrics_list]
    colors = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in overall_scores]

    fig.add_trace(
        go.Bar(x=stores, y=overall_scores, marker_color=colors, name='Overall Score'),
        row=1, col=1
    )

    # 2. Quality dimensions heatmap
    dimensions = ['completeness', 'consistency', 'validity', 'accuracy', 'timeliness', 'uniqueness']
    heatmap_data = [
        [
            getattr(m, f'{dim}_score')
            for dim in dimensions
        ]
        for m in metrics_list
    ]

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=dimensions,
            y=stores,
            colorscale='RdYlGn',
            name='Dimensions'
        ),
        row=1, col=2
    )

    # 3. Missing data distribution
    missing_pcts = [m.missing_percentage for m in metrics_list]
    fig.add_trace(
        go.Box(y=missing_pcts, name='Missing %', marker_color='lightblue'),
        row=2, col=1
    )

    # 4. Outlier distribution
    outlier_counts = [m.outlier_count for m in metrics_list]
    fig.add_trace(
        go.Scatter(
            x=stores,
            y=outlier_counts,
            mode='markers+lines',
            name='Outliers',
            marker=dict(size=10, color='red')
        ),
        row=2, col=2
    )

    # 5. Quality score trends (if history available)
    if validator.validation_history:
        history_df = pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                'store_id': h['store_id'],
                'score': h['metrics']['scores']['overall']
            }
            for h in validator.validation_history
            if h.get('store_id')
        ])

        for store in stores:
            store_history = history_df[history_df['store_id'] == store]
            if not store_history.empty:
                fig.add_trace(
                    go.Scatter(
                        x=store_history['timestamp'],
                        y=store_history['score'],
                        mode='lines+markers',
                        name=store
                    ),
                    row=3, col=1
                )

    # 6. Issue summary
    issue_counts = [len(m.issues) for m in metrics_list]
    fig.add_trace(
        go.Bar(x=stores, y=issue_counts, marker_color='coral', name='Issues'),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Data Quality Dashboard",
        showlegend=True,
        height=1200,
        template='plotly_white'
    )

    # Save dashboard
    fig.write_html(output_file)
    print(f"✓ Dashboard saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Data Quality Validation System")
    print("=" * 60)

    # Create sample data with quality issues
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    sample_data = pd.DataFrame({
        'date': dates,
        'store': ['store_001'] * 100,
        'sales': np.random.normal(1000, 200, 100),
        'inventory': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100)
    })

    # Introduce quality issues
    sample_data.loc[10:15, 'sales'] = np.nan  # Missing data
    sample_data.loc[20, 'sales'] = -500  # Invalid negative sales
    sample_data.loc[25:27, :] = sample_data.loc[22:24, :].values  # Duplicates
    sample_data.loc[30, 'sales'] = 10000  # Outlier

    # Initialize validator
    validator = DataQualityValidator(verbose=True)

    # Run validation
    metrics = validator.validate(sample_data, store_id='store_001')

    # Generate report
    report = validator.generate_report(format='html')
    validator.save_report(report, 'data_quality_report.html')

    # Create dashboard
    create_quality_dashboard(validator, 'data_quality_dashboard.html')

    print("\n✓ Validation complete!")
    print(f"  Quality Score: {metrics.overall_score:.2f}/100")
    print(f"  Issues Found: {len(metrics.issues)}")
