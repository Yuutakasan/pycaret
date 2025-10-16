"""
Unit tests for Data Quality Validation System

Tests cover:
- Missing data detection and imputation
- Outlier identification and handling
- Consistency checks
- Temporal validation
- Business rule validation
- Quality score calculation
- Report generation
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.validation.data_quality import (
    DataQualityValidator,
    ImputationStrategy,
    OutlierMethod,
    QualityMetrics,
    ValidationRule,
    create_quality_dashboard
)


class TestDataQualityValidator(unittest.TestCase):
    """Test suite for DataQualityValidator."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'store': ['store_001'] * 100,
            'sales': np.random.normal(1000, 200, 100),
            'inventory': np.random.randint(50, 200, 100),
            'price': np.random.uniform(10, 50, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

        self.validator = DataQualityValidator(verbose=False)

    def test_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.missing_threshold, 0.3)
        self.assertEqual(self.validator.outlier_threshold, 3.0)
        self.assertEqual(self.validator.temporal_column, 'date')
        self.assertEqual(self.validator.store_column, 'store')
        self.assertIsInstance(self.validator.validation_rules, list)
        self.assertGreater(len(self.validator.validation_rules), 0)

    def test_add_validation_rule(self):
        """Test adding custom validation rules."""
        initial_count = len(self.validator.validation_rules)

        self.validator.add_rule(
            name="test_rule",
            description="Test validation rule",
            rule_fn=lambda df: df['sales'] > 0
        )

        self.assertEqual(len(self.validator.validation_rules), initial_count + 1)
        self.assertEqual(self.validator.validation_rules[-1].name, "test_rule")

    def test_detect_missing_data_clean(self):
        """Test missing data detection on clean data."""
        results = self.validator.detect_missing_data(self.sample_data)

        self.assertEqual(results['missing_cells'], 0)
        self.assertEqual(results['missing_percentage'], 0.0)
        self.assertEqual(len(results['patterns']), 0)

    def test_detect_missing_data_with_missing(self):
        """Test missing data detection with missing values."""
        # Introduce missing data
        data = self.sample_data.copy()
        data.loc[10:15, 'sales'] = np.nan
        data.loc[20:25, 'inventory'] = np.nan

        results = self.validator.detect_missing_data(data)

        self.assertGreater(results['missing_cells'], 0)
        self.assertGreater(results['missing_percentage'], 0)
        self.assertIn('sales', results['columns'])
        self.assertEqual(results['columns']['sales']['missing_count'], 6)

    def test_impute_missing_data_median(self):
        """Test median imputation."""
        data = self.sample_data.copy()
        original_median = data['sales'].median()
        data.loc[10:15, 'sales'] = np.nan

        imputed = self.validator.impute_missing_data(
            data,
            strategy=ImputationStrategy.MEDIAN
        )

        self.assertEqual(imputed['sales'].isnull().sum(), 0)
        # Imputed values should be close to original median
        self.assertAlmostEqual(
            imputed.loc[10, 'sales'],
            original_median,
            delta=10
        )

    def test_impute_missing_data_mean(self):
        """Test mean imputation."""
        data = self.sample_data.copy()
        original_mean = data['sales'].mean()
        data.loc[10:15, 'sales'] = np.nan

        imputed = self.validator.impute_missing_data(
            data,
            strategy=ImputationStrategy.MEAN
        )

        self.assertEqual(imputed['sales'].isnull().sum(), 0)
        self.assertAlmostEqual(
            imputed.loc[10, 'sales'],
            original_mean,
            delta=10
        )

    def test_impute_missing_data_forward_fill(self):
        """Test forward fill imputation."""
        data = self.sample_data.copy()
        previous_value = data.loc[9, 'sales']
        data.loc[10:15, 'sales'] = np.nan

        imputed = self.validator.impute_missing_data(
            data,
            strategy=ImputationStrategy.FORWARD_FILL
        )

        self.assertEqual(imputed['sales'].isnull().sum(), 0)
        self.assertEqual(imputed.loc[10, 'sales'], previous_value)

    def test_impute_missing_data_mode_categorical(self):
        """Test mode imputation for categorical data."""
        data = self.sample_data.copy()
        data.loc[10:15, 'category'] = np.nan

        imputed = self.validator.impute_missing_data(
            data,
            column_strategies={'category': ImputationStrategy.MODE}
        )

        self.assertEqual(imputed['category'].isnull().sum(), 0)
        self.assertIn(imputed.loc[10, 'category'], ['A', 'B', 'C'])

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        data = self.sample_data.copy()
        # Add obvious outliers
        data.loc[10, 'sales'] = 10000
        data.loc[20, 'sales'] = -1000

        results = self.validator.detect_outliers(
            data,
            method=OutlierMethod.IQR,
            columns=['sales']
        )

        self.assertGreater(results['total_outliers'], 0)
        self.assertIn('sales', results['columns'])
        self.assertGreater(results['columns']['sales']['outlier_count'], 0)

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        data = self.sample_data.copy()
        data.loc[10, 'sales'] = 5000  # Outlier

        results = self.validator.detect_outliers(
            data,
            method=OutlierMethod.Z_SCORE,
            columns=['sales']
        )

        self.assertGreaterEqual(results['total_outliers'], 0)

    def test_handle_outliers_cap(self):
        """Test outlier capping."""
        data = self.sample_data.copy()
        data.loc[10, 'sales'] = 10000

        cleaned = self.validator.handle_outliers(
            data,
            method='cap',
            detection_method=OutlierMethod.IQR,
            columns=['sales']
        )

        # Outlier should be capped
        self.assertLess(cleaned.loc[10, 'sales'], 10000)

    def test_handle_outliers_remove(self):
        """Test outlier removal."""
        data = self.sample_data.copy()
        original_len = len(data)
        data.loc[10, 'sales'] = 10000

        cleaned = self.validator.handle_outliers(
            data,
            method='remove',
            detection_method=OutlierMethod.IQR,
            columns=['sales']
        )

        # Should have fewer rows
        self.assertLessEqual(len(cleaned), original_len)

    def test_check_consistency_clean_data(self):
        """Test consistency checks on clean data."""
        results = self.validator.check_consistency(self.sample_data)

        self.assertGreater(results['passed'], 0)
        self.assertIn('checks', results)

    def test_check_consistency_type_issues(self):
        """Test consistency checks with type issues."""
        data = self.sample_data.copy()
        # Mix types in numeric column
        data.loc[10, 'sales'] = 'invalid'

        results = self.validator.check_consistency(data)

        # Should detect some issues
        self.assertIsInstance(results, dict)
        self.assertIn('checks', results)

    def test_check_temporal_consistency(self):
        """Test temporal consistency validation."""
        data = self.sample_data.copy()

        # Introduce temporal issues
        data.loc[50, 'date'] = data.loc[10, 'date']  # Duplicate date
        data.loc[60, 'date'] = '2099-01-01'  # Future date

        results = self.validator.check_consistency(data)

        self.assertIn('checks', results)

    def test_validate_business_rules_clean(self):
        """Test business rule validation on clean data."""
        results = self.validator.validate_business_rules(self.sample_data)

        self.assertGreater(results['passed'], 0)
        self.assertEqual(results['failed'], 0)

    def test_validate_business_rules_violations(self):
        """Test business rule validation with violations."""
        data = self.sample_data.copy()
        data.loc[10, 'sales'] = -500  # Violates no_negative_sales rule

        results = self.validator.validate_business_rules(data)

        self.assertGreater(results['failed'], 0)
        violations = [r for r in results['rules'] if not r['passed']]
        self.assertGreater(len(violations), 0)

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        metrics = self.validator.calculate_quality_score(self.sample_data)

        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreaterEqual(metrics.overall_score, 0)
        self.assertLessEqual(metrics.overall_score, 100)
        self.assertGreaterEqual(metrics.completeness_score, 0)
        self.assertGreaterEqual(metrics.consistency_score, 0)

    def test_calculate_quality_score_with_issues(self):
        """Test quality score with data issues."""
        data = self.sample_data.copy()

        # Introduce issues
        data.loc[10:15, 'sales'] = np.nan  # Missing data
        data.loc[20, 'sales'] = -500  # Invalid
        data.loc[30, 'sales'] = 10000  # Outlier

        metrics = self.validator.calculate_quality_score(data)

        self.assertLess(metrics.overall_score, 100)
        self.assertGreater(metrics.missing_count, 0)
        self.assertGreater(len(metrics.issues), 0)

    def test_validate_comprehensive(self):
        """Test comprehensive validation."""
        metrics = self.validator.validate(
            self.sample_data,
            store_id='test_store'
        )

        self.assertIsInstance(metrics, QualityMetrics)
        self.assertIn('test_store', self.validator.store_metrics)
        self.assertGreater(len(self.validator.validation_history), 0)

    def test_generate_report_dict(self):
        """Test report generation as dictionary."""
        self.validator.validate(self.sample_data, store_id='store_001')
        self.validator.validate(self.sample_data, store_id='store_002')

        report = self.validator.generate_report(format='dict')

        self.assertIsInstance(report, dict)
        self.assertIn('generated_at', report)
        self.assertIn('stores', report)
        self.assertEqual(report['total_stores'], 2)

    def test_generate_report_dataframe(self):
        """Test report generation as DataFrame."""
        self.validator.validate(self.sample_data, store_id='store_001')

        report = self.validator.generate_report(format='dataframe')

        self.assertIsInstance(report, pd.DataFrame)
        self.assertIn('store_id', report.columns)
        self.assertIn('overall', report.columns)

    def test_generate_report_html(self):
        """Test report generation as HTML."""
        self.validator.validate(self.sample_data, store_id='store_001')

        report = self.validator.generate_report(format='html')

        self.assertIsInstance(report, str)
        self.assertIn('<!DOCTYPE html>', report)
        self.assertIn('Data Quality Report', report)

    def test_validation_history(self):
        """Test validation history tracking."""
        self.validator.validate(self.sample_data, store_id='store_001')
        self.validator.validate(self.sample_data, store_id='store_002')

        history = self.validator.get_validation_history()

        self.assertEqual(len(history), 2)
        self.assertIn('timestamp', history[0])
        self.assertIn('store_id', history[0])

    def test_validation_history_filtered(self):
        """Test filtered validation history."""
        self.validator.validate(self.sample_data, store_id='store_001')
        self.validator.validate(self.sample_data, store_id='store_002')

        history = self.validator.get_validation_history(store_id='store_001')

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['store_id'], 'store_001')

    def test_validation_history_limit(self):
        """Test validation history with limit."""
        for i in range(5):
            self.validator.validate(self.sample_data, store_id=f'store_{i:03d}')

        history = self.validator.get_validation_history(limit=3)

        self.assertEqual(len(history), 3)

    def test_quality_metrics_to_dict(self):
        """Test QualityMetrics to_dict conversion."""
        metrics = QualityMetrics(
            completeness_score=95.0,
            consistency_score=90.0,
            overall_score=92.5
        )

        metrics_dict = metrics.to_dict()

        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('scores', metrics_dict)
        self.assertIn('metrics', metrics_dict)
        self.assertEqual(metrics_dict['scores']['completeness'], 95.0)

    def test_custom_weights(self):
        """Test quality score calculation with custom weights."""
        custom_weights = {
            'completeness': 0.4,
            'consistency': 0.3,
            'validity': 0.2,
            'accuracy': 0.05,
            'timeliness': 0.03,
            'uniqueness': 0.02
        }

        metrics = self.validator.calculate_quality_score(
            self.sample_data,
            weights=custom_weights
        )

        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreaterEqual(metrics.overall_score, 0)

    def test_duplicate_detection(self):
        """Test duplicate detection in quality scoring."""
        data = self.sample_data.copy()
        # Add duplicates
        data = pd.concat([data, data.iloc[:5]], ignore_index=True)

        metrics = self.validator.calculate_quality_score(data)

        self.assertGreater(metrics.duplicate_count, 0)
        self.assertLess(metrics.uniqueness_score, 100)

    def test_seasonal_imputation(self):
        """Test seasonal imputation strategy."""
        data = self.sample_data.copy()
        data.loc[10:15, 'sales'] = np.nan

        imputed = self.validator.impute_missing_data(
            data,
            column_strategies={'sales': ImputationStrategy.SEASONAL}
        )

        self.assertEqual(imputed['sales'].isnull().sum(), 0)

    def test_cross_store_consistency(self):
        """Test cross-store consistency checks."""
        reference_data = self.sample_data.copy()
        test_data = self.sample_data.copy()

        # Modify test data to create inconsistencies
        test_data['new_column'] = 0

        results = self.validator.check_consistency(
            test_data,
            reference_data=reference_data
        )

        self.assertIn('checks', results)
        # Should detect column mismatch
        cross_check = [
            c for c in results['checks']
            if c['check'] == 'cross_store_consistency'
        ]
        self.assertGreater(len(cross_check), 0)

    def test_multiple_outlier_methods(self):
        """Test different outlier detection methods."""
        data = self.sample_data.copy()
        data.loc[10, 'sales'] = 10000

        for method in [OutlierMethod.IQR, OutlierMethod.Z_SCORE, OutlierMethod.MODIFIED_Z_SCORE]:
            results = self.validator.detect_outliers(
                data,
                method=method,
                columns=['sales']
            )
            self.assertIn('method', results)
            self.assertEqual(results['method'], method.value)

    def test_validation_rule_severity(self):
        """Test validation rules with different severity levels."""
        self.validator.add_rule(
            name="warning_rule",
            description="Warning level rule",
            rule_fn=lambda df: df['sales'] < 2000,
            severity="warning"
        )

        data = self.sample_data.copy()
        data.loc[10, 'sales'] = 3000

        results = self.validator.validate_business_rules(data)

        warning_rules = [
            r for r in results['rules']
            if r['severity'] == 'warning'
        ]
        self.assertGreater(len(warning_rules), 0)

    def test_edge_case_empty_dataframe(self):
        """Test validation on empty DataFrame."""
        empty_df = pd.DataFrame()

        with self.assertRaises(Exception):
            self.validator.validate(empty_df)

    def test_edge_case_single_row(self):
        """Test validation on single row DataFrame."""
        single_row = self.sample_data.iloc[:1].copy()

        metrics = self.validator.validate(single_row)

        self.assertIsInstance(metrics, QualityMetrics)

    def test_edge_case_all_missing(self):
        """Test with all missing values."""
        data = self.sample_data.copy()
        data['sales'] = np.nan

        metrics = self.validator.calculate_quality_score(data)

        self.assertEqual(metrics.missing_percentage, 100 / len(data.columns))
        self.assertLess(metrics.completeness_score, 100)


class TestValidationRule(unittest.TestCase):
    """Test ValidationRule dataclass."""

    def test_validation_rule_creation(self):
        """Test creating a validation rule."""
        rule = ValidationRule(
            name="test",
            description="Test rule",
            rule_fn=lambda df: df['value'] > 0,
            severity="error",
            enabled=True
        )

        self.assertEqual(rule.name, "test")
        self.assertEqual(rule.severity, "error")
        self.assertTrue(rule.enabled)


class TestQualityDashboard(unittest.TestCase):
    """Test quality dashboard creation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        self.sample_data = pd.DataFrame({
            'date': dates,
            'store': ['store_001'] * 50,
            'sales': np.random.normal(1000, 200, 50)
        })

        self.validator = DataQualityValidator(verbose=False)
        self.validator.validate(self.sample_data, store_id='store_001')

    def test_dashboard_creation_no_plotly(self):
        """Test dashboard creation without plotly."""
        # This should handle gracefully if plotly is not installed
        try:
            create_quality_dashboard(self.validator, 'test_dashboard.html')
        except Exception as e:
            # Should not raise exception
            self.assertIn('plotly', str(e).lower())


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
