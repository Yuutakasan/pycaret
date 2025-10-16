"""
Data Quality Validation System - Example Usage

This example demonstrates how to use the data quality validation system
for retail sales data analysis.

Examples include:
- Basic validation
- Custom business rules
- Multi-store analysis
- Quality dashboard creation
- Missing data imputation strategies
- Outlier detection and handling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the data quality validator
import sys
sys.path.append('..')
from src.validation.data_quality import (
    DataQualityValidator,
    ImputationStrategy,
    OutlierMethod,
    create_quality_dashboard
)


def create_sample_retail_data(
    store_id: str,
    start_date: str = '2024-01-01',
    periods: int = 365,
    add_issues: bool = False
) -> pd.DataFrame:
    """
    Create sample retail sales data.

    Parameters
    ----------
    store_id : str
        Store identifier
    start_date : str
        Start date for time series
    periods : int
        Number of days
    add_issues : bool
        Add data quality issues for testing

    Returns
    -------
    pd.DataFrame
        Sample retail data
    """
    np.random.seed(hash(store_id) % 2**32)

    dates = pd.date_range(start_date, periods=periods, freq='D')

    # Base sales with weekly seasonality
    base_sales = 1000 + 200 * np.sin(np.arange(periods) * 2 * np.pi / 7)

    # Add random noise
    sales = base_sales + np.random.normal(0, 100, periods)

    df = pd.DataFrame({
        'date': dates,
        'store': store_id,
        'sales': sales,
        'inventory': np.random.randint(50, 300, periods),
        'price': np.random.uniform(10, 100, periods),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], periods),
        'promotion': np.random.choice([0, 1], periods, p=[0.8, 0.2])
    })

    if add_issues:
        # Introduce various data quality issues

        # 1. Missing data (5-10%)
        missing_indices = np.random.choice(
            df.index,
            size=int(len(df) * 0.07),
            replace=False
        )
        df.loc[missing_indices, 'sales'] = np.nan

        # 2. Negative values (business rule violation)
        negative_indices = np.random.choice(df.index, size=3, replace=False)
        df.loc[negative_indices, 'sales'] = -np.abs(df.loc[negative_indices, 'sales'])

        # 3. Outliers (extreme values)
        outlier_indices = np.random.choice(df.index, size=5, replace=False)
        df.loc[outlier_indices, 'sales'] = df.loc[outlier_indices, 'sales'] * 5

        # 4. Duplicates
        duplicate_rows = df.sample(n=3)
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)

        # 5. Inconsistent data types
        df.loc[10, 'inventory'] = 'unknown'

        # 6. Future dates
        df.loc[20, 'date'] = pd.Timestamp.now() + timedelta(days=365)

    return df


def example_basic_validation():
    """Example 1: Basic data quality validation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Data Quality Validation")
    print("="*80 + "\n")

    # Create sample data with quality issues
    df = create_sample_retail_data('store_001', add_issues=True)

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Initialize validator
    validator = DataQualityValidator(
        missing_threshold=0.1,
        outlier_threshold=3.0,
        verbose=True
    )

    # Run validation
    metrics = validator.validate(df, store_id='store_001')

    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(f"Overall Quality Score: {metrics.overall_score:.2f}/100")
    print(f"\nDimension Scores:")
    for dimension in ['completeness', 'consistency', 'validity', 'accuracy', 'timeliness', 'uniqueness']:
        score = getattr(metrics, f'{dimension}_score')
        print(f"  {dimension.capitalize():15s}: {score:6.2f}/100")

    print(f"\nIssues Found: {len(metrics.issues)}")
    for issue in metrics.issues:
        print(f"  - [{issue['severity'].upper()}] {issue['message']}")


def example_custom_business_rules():
    """Example 2: Adding custom business rules."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Business Rules")
    print("="*80 + "\n")

    df = create_sample_retail_data('store_001', add_issues=True)

    validator = DataQualityValidator(verbose=True)

    # Add custom business rules
    validator.add_rule(
        name="sales_inventory_ratio",
        description="Sales should not exceed inventory",
        rule_fn=lambda df: df['sales'] <= df['inventory'].astype(float) * 50,
        severity="warning"
    )

    validator.add_rule(
        name="promotion_price_check",
        description="Promoted items should have price > 0",
        rule_fn=lambda df: (df['promotion'] == 0) | (df['price'] > 0),
        severity="error"
    )

    validator.add_rule(
        name="reasonable_sales_range",
        description="Sales should be between 0 and 10000",
        rule_fn=lambda df: (df['sales'].fillna(0) >= 0) & (df['sales'].fillna(0) <= 10000),
        severity="error"
    )

    # Validate with custom rules
    results = validator.validate_business_rules(df)

    print(f"\nBusiness Rule Validation:")
    print(f"  Total Rules: {results['total_rules']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Warnings: {results['warnings']}")

    print(f"\nRule Details:")
    for rule in results['rules']:
        status = "✓ PASS" if rule['passed'] else "✗ FAIL"
        print(f"  {status} - {rule['name']}")
        if not rule['passed']:
            print(f"      Violations: {rule['violations']} ({rule['violation_percentage']:.2f}%)")


def example_missing_data_imputation():
    """Example 3: Missing data detection and imputation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Missing Data Imputation Strategies")
    print("="*80 + "\n")

    df = create_sample_retail_data('store_001', add_issues=True)

    validator = DataQualityValidator(verbose=True)

    # Detect missing data
    print("Analyzing missing data...")
    missing_info = validator.detect_missing_data(df)

    print(f"\nMissing Data Summary:")
    print(f"  Total Missing Cells: {missing_info['missing_cells']}")
    print(f"  Missing Percentage: {missing_info['missing_percentage']:.2f}%")

    print(f"\nPer-Column Missing Data:")
    for col, info in missing_info['columns'].items():
        if info['missing_count'] > 0:
            print(f"  {col}: {info['missing_count']} ({info['missing_percentage']:.2f}%)")

    # Test different imputation strategies
    strategies = {
        'median': ImputationStrategy.MEDIAN,
        'mean': ImputationStrategy.MEAN,
        'forward_fill': ImputationStrategy.FORWARD_FILL,
        'seasonal': ImputationStrategy.SEASONAL
    }

    print(f"\n{'='*60}")
    print("Testing Imputation Strategies:")
    print(f"{'='*60}")

    for name, strategy in strategies.items():
        df_imputed = validator.impute_missing_data(
            df.copy(),
            strategy=strategy
        )
        remaining_missing = df_imputed.isnull().sum().sum()
        print(f"  {name:15s}: {remaining_missing} missing values remaining")


def example_outlier_detection():
    """Example 4: Outlier detection and handling."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Outlier Detection and Handling")
    print("="*80 + "\n")

    df = create_sample_retail_data('store_001', add_issues=True)

    validator = DataQualityValidator(verbose=True)

    # Test different outlier detection methods
    methods = {
        'IQR': OutlierMethod.IQR,
        'Z-Score': OutlierMethod.Z_SCORE,
        'Modified Z-Score': OutlierMethod.MODIFIED_Z_SCORE
    }

    print("Comparing Outlier Detection Methods:\n")

    for name, method in methods.items():
        results = validator.detect_outliers(
            df,
            method=method,
            columns=['sales', 'price']
        )

        print(f"{name}:")
        print(f"  Total Outliers: {results['total_outliers']}")
        for col, info in results['columns'].items():
            if info['outlier_count'] > 0:
                print(f"    {col}: {info['outlier_count']} ({info['outlier_percentage']:.2f}%)")
        print()

    # Handle outliers
    print("Handling Outliers:")
    print("-" * 60)

    df_capped = validator.handle_outliers(
        df.copy(),
        method='cap',
        detection_method=OutlierMethod.IQR
    )
    print(f"  Capping: Max sales = {df_capped['sales'].max():.2f}")

    df_removed = validator.handle_outliers(
        df.copy(),
        method='remove',
        detection_method=OutlierMethod.IQR
    )
    print(f"  Removal: {len(df) - len(df_removed)} rows removed")


def example_multi_store_analysis():
    """Example 5: Multi-store quality analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Multi-Store Quality Analysis")
    print("="*80 + "\n")

    # Create data for multiple stores
    stores = ['store_001', 'store_002', 'store_003', 'store_004', 'store_005']

    validator = DataQualityValidator(verbose=False)

    print("Validating multiple stores...")

    for store_id in stores:
        # Vary quality issues across stores
        add_issues = store_id in ['store_002', 'store_004']
        df = create_sample_retail_data(store_id, add_issues=add_issues)

        validator.validate(df, store_id=store_id)
        print(f"  ✓ Validated {store_id}")

    # Generate comparative report
    print(f"\n{'='*60}")
    print("Multi-Store Quality Comparison")
    print(f"{'='*60}\n")

    report_df = validator.generate_report(format='dataframe')

    print(report_df[['store_id', 'overall', 'completeness', 'validity', 'uniqueness']].to_string(index=False))

    # Identify problematic stores
    print(f"\n{'='*60}")
    print("Store Quality Analysis")
    print(f"{'='*60}\n")

    avg_score = report_df['overall'].mean()
    print(f"Average Quality Score: {avg_score:.2f}")

    low_quality = report_df[report_df['overall'] < 80]
    if not low_quality.empty:
        print(f"\nStores Needing Attention ({len(low_quality)}):")
        for _, row in low_quality.iterrows():
            print(f"  • {row['store_id']}: {row['overall']:.2f}/100")
            if row['missing_percentage'] > 5:
                print(f"    - High missing data: {row['missing_percentage']:.1f}%")
            if row['duplicate_count'] > 0:
                print(f"    - Duplicates found: {row['duplicate_count']}")


def example_temporal_consistency():
    """Example 6: Temporal consistency validation."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Temporal Consistency Validation")
    print("="*80 + "\n")

    df = create_sample_retail_data('store_001', add_issues=True)

    validator = DataQualityValidator(verbose=True)

    # Check consistency
    results = validator.check_consistency(df)

    print(f"\nConsistency Check Results:")
    print(f"  Checks Passed: {results['passed']}")
    print(f"  Checks Failed: {results['failed']}")

    print(f"\nDetailed Check Results:")
    for check in results['checks']:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"  {status} - {check['check']}")
        if not check['passed'] and 'issues' in check:
            for issue in check['issues']:
                print(f"      • {issue}")


def example_quality_dashboard():
    """Example 7: Generate quality dashboard."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Quality Dashboard Generation")
    print("="*80 + "\n")

    # Create data for multiple stores
    stores = ['store_001', 'store_002', 'store_003', 'store_004']

    validator = DataQualityValidator(verbose=False)

    print("Preparing data for dashboard...")

    for i, store_id in enumerate(stores):
        add_issues = i % 2 == 1  # Add issues to every other store
        df = create_sample_retail_data(store_id, add_issues=add_issues)

        # Validate multiple times to build history
        for j in range(3):
            validator.validate(df, store_id=store_id)

    # Generate HTML report
    print("\nGenerating reports...")

    html_report = validator.generate_report(format='html')
    validator.save_report(html_report, 'data_quality_report.html')
    print("  ✓ HTML report saved to: data_quality_report.html")

    # Generate dashboard
    try:
        create_quality_dashboard(validator, 'data_quality_dashboard.html')
        print("  ✓ Dashboard saved to: data_quality_dashboard.html")
    except ImportError:
        print("  ⚠ Plotly not installed. Dashboard requires: pip install plotly")

    # Save DataFrame report
    df_report = validator.generate_report(format='dataframe')
    df_report.to_csv('data_quality_report.csv', index=False)
    print("  ✓ CSV report saved to: data_quality_report.csv")

    print("\nValidation History:")
    history = validator.get_validation_history(limit=5)
    for record in history:
        print(f"  • {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{record['store_id']}: {record['metrics']['scores']['overall']:.2f}")


def example_complete_workflow():
    """Example 8: Complete data quality workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Complete Data Quality Workflow")
    print("="*80 + "\n")

    # 1. Load data
    print("Step 1: Loading data...")
    df = create_sample_retail_data('store_001', add_issues=True)
    print(f"  Loaded {len(df)} records")

    # 2. Initialize validator with custom settings
    print("\nStep 2: Initializing validator...")
    validator = DataQualityValidator(
        missing_threshold=0.15,
        outlier_threshold=3.0,
        duplicate_subset=['date', 'store'],
        verbose=True
    )

    # 3. Add business rules
    print("\nStep 3: Adding business rules...")
    validator.add_rule(
        name="sales_range",
        description="Sales must be between 0 and 10000",
        rule_fn=lambda df: (df['sales'].fillna(0) >= 0) & (df['sales'].fillna(0) <= 10000)
    )

    # 4. Run initial validation
    print("\nStep 4: Running initial validation...")
    initial_metrics = validator.validate(df, store_id='store_001')

    # 5. Clean data based on findings
    print("\nStep 5: Cleaning data...")

    # Impute missing values
    df_clean = validator.impute_missing_data(
        df,
        strategy=ImputationStrategy.SEASONAL
    )

    # Handle outliers
    df_clean = validator.handle_outliers(
        df_clean,
        method='cap',
        detection_method=OutlierMethod.IQR
    )

    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['date', 'store'])

    # Remove invalid records (negative sales)
    df_clean = df_clean[df_clean['sales'] >= 0]

    print(f"  Cleaned data: {len(df)} -> {len(df_clean)} records")

    # 6. Validate cleaned data
    print("\nStep 6: Validating cleaned data...")
    final_metrics = validator.validate(df_clean, store_id='store_001_cleaned')

    # 7. Compare results
    print(f"\n{'='*60}")
    print("Quality Improvement Summary")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 60)

    metrics_to_compare = [
        ('Overall Score', 'overall_score'),
        ('Completeness', 'completeness_score'),
        ('Validity', 'validity_score'),
        ('Uniqueness', 'uniqueness_score'),
        ('Missing %', 'missing_percentage'),
        ('Outliers', 'outlier_count'),
        ('Duplicates', 'duplicate_count')
    ]

    for label, attr in metrics_to_compare:
        before = getattr(initial_metrics, attr)
        after = getattr(final_metrics, attr)
        change = after - before

        if 'score' in attr:
            change_str = f"+{change:,.2f}" if change >= 0 else f"{change:,.2f}"
        else:
            change_str = f"{change:+,.0f}"

        print(f"{label:<25} {before:12.2f} {after:12.2f} {change_str:>12}")

    print("\n" + "="*60)
    print("✓ Workflow Complete!")
    print("="*60 + "\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*20 + "DATA QUALITY VALIDATION SYSTEM")
    print(" "*25 + "Example Usage Guide")
    print("="*80)

    examples = [
        ("Basic Validation", example_basic_validation),
        ("Custom Business Rules", example_custom_business_rules),
        ("Missing Data Imputation", example_missing_data_imputation),
        ("Outlier Detection", example_outlier_detection),
        ("Multi-Store Analysis", example_multi_store_analysis),
        ("Temporal Consistency", example_temporal_consistency),
        ("Quality Dashboard", example_quality_dashboard),
        ("Complete Workflow", example_complete_workflow)
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {str(e)}\n")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
