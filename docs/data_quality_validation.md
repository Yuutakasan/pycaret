# Data Quality Validation System

## Overview

The Data Quality Validation System provides comprehensive automated validation for retail sales data, including missing data detection, outlier identification, consistency checks, temporal validation, and business rule enforcement.

## Features

### 1. **Missing Data Detection & Imputation**
- Automatic detection of missing values across all columns
- Pattern analysis (systematic missingness, high-missing columns)
- Multiple imputation strategies:
  - Mean/Median/Mode
  - Forward/Backward Fill
  - Interpolation
  - Seasonal (time-aware)
  - KNN-based
  - Zero-fill

### 2. **Outlier Identification & Handling**
- Multiple detection methods:
  - IQR (Interquartile Range)
  - Z-Score
  - Modified Z-Score
  - Isolation Forest
  - DBSCAN
  - Local Outlier Factor
- Handling strategies:
  - Capping at boundaries
  - Removal
  - Transformation (log, sqrt)

### 3. **Data Consistency Checks**
- Data type consistency
- Value range validation
- Cross-store consistency
- Column alignment
- Mixed type detection

### 4. **Temporal Consistency Validation**
- Date format validation
- Chronological ordering
- Duplicate date detection
- Gap identification
- Future date detection
- Date range reasonableness

### 5. **Business Rule Validation**
- Customizable rule framework
- Built-in retail rules:
  - No negative sales
  - No negative inventory
  - No negative prices
  - Reasonable date ranges
- Rule severity levels (error/warning/info)
- Violation tracking and reporting

### 6. **Quality Score Calculation**
Six quality dimensions with configurable weights:
- **Completeness** (25%): Missing data percentage
- **Consistency** (20%): Data type and value consistency
- **Validity** (20%): Business rule compliance
- **Accuracy** (15%): Outlier-based accuracy
- **Timeliness** (10%): Data freshness
- **Uniqueness** (10%): Duplicate detection

### 7. **Comprehensive Reporting**
- Multi-store comparison
- Quality metrics dashboard
- HTML/CSV/DataFrame reports
- Validation history tracking
- Issue prioritization

## Installation

```bash
# Core dependencies (included in pycaret)
pip install pandas numpy scipy

# Optional: For interactive dashboards
pip install plotly
```

## Quick Start

### Basic Usage

```python
from src.validation.data_quality import DataQualityValidator

# Initialize validator
validator = DataQualityValidator(
    missing_threshold=0.3,
    outlier_threshold=3.0,
    verbose=True
)

# Validate data
metrics = validator.validate(df, store_id='store_001')

print(f"Quality Score: {metrics.overall_score:.2f}/100")
```

### Custom Business Rules

```python
# Add custom validation rule
validator.add_rule(
    name="sales_inventory_ratio",
    description="Sales should not exceed inventory * 50",
    rule_fn=lambda df: df['sales'] <= df['inventory'] * 50,
    severity="warning"
)

# Validate with custom rules
results = validator.validate_business_rules(df)
```

### Missing Data Imputation

```python
from src.validation.data_quality import ImputationStrategy

# Impute using different strategies
df_imputed = validator.impute_missing_data(
    df,
    strategy=ImputationStrategy.SEASONAL,
    column_strategies={
        'sales': ImputationStrategy.SEASONAL,
        'price': ImputationStrategy.MEDIAN,
        'category': ImputationStrategy.MODE
    }
)
```

### Outlier Detection

```python
from src.validation.data_quality import OutlierMethod

# Detect outliers using IQR method
outliers = validator.detect_outliers(
    df,
    method=OutlierMethod.IQR,
    columns=['sales', 'price']
)

# Handle outliers by capping
df_clean = validator.handle_outliers(
    df,
    method='cap',
    detection_method=OutlierMethod.IQR
)
```

### Multi-Store Analysis

```python
# Validate multiple stores
for store_id in ['store_001', 'store_002', 'store_003']:
    df_store = df[df['store'] == store_id]
    validator.validate(df_store, store_id=store_id)

# Generate comparative report
report_df = validator.generate_report(format='dataframe')
print(report_df[['store_id', 'overall', 'completeness', 'validity']])
```

### Quality Dashboard

```python
from src.validation.data_quality import create_quality_dashboard

# Create interactive dashboard
create_quality_dashboard(
    validator,
    output_file='data_quality_dashboard.html'
)
```

## API Reference

### DataQualityValidator Class

#### Constructor

```python
DataQualityValidator(
    missing_threshold: float = 0.3,
    outlier_threshold: float = 3.0,
    duplicate_subset: Optional[List[str]] = None,
    temporal_column: str = 'date',
    store_column: str = 'store',
    verbose: bool = True
)
```

**Parameters:**
- `missing_threshold`: Maximum acceptable missing value proportion (0-1)
- `outlier_threshold`: Z-score threshold for outlier detection
- `duplicate_subset`: Columns to check for duplicates
- `temporal_column`: Name of date/time column
- `store_column`: Name of store identifier column
- `verbose`: Print validation progress

#### Main Methods

##### validate()
```python
validate(
    df: pd.DataFrame,
    store_id: Optional[str] = None,
    save_metrics: bool = True
) -> QualityMetrics
```
Run comprehensive validation and return quality metrics.

##### detect_missing_data()
```python
detect_missing_data(
    df: pd.DataFrame,
    store_id: Optional[str] = None
) -> Dict[str, Any]
```
Detect and analyze missing data patterns.

##### impute_missing_data()
```python
impute_missing_data(
    df: pd.DataFrame,
    strategy: Union[str, ImputationStrategy] = ImputationStrategy.MEDIAN,
    column_strategies: Optional[Dict[str, ImputationStrategy]] = None
) -> pd.DataFrame
```
Impute missing values using specified strategies.

##### detect_outliers()
```python
detect_outliers(
    df: pd.DataFrame,
    method: Union[str, OutlierMethod] = OutlierMethod.IQR,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]
```
Detect outliers in numeric columns.

##### handle_outliers()
```python
handle_outliers(
    df: pd.DataFrame,
    method: str = 'cap',
    detection_method: Union[str, OutlierMethod] = OutlierMethod.IQR,
    columns: Optional[List[str]] = None
) -> pd.DataFrame
```
Handle outliers using specified method ('cap', 'remove', 'transform').

##### check_consistency()
```python
check_consistency(
    df: pd.DataFrame,
    store_id: Optional[str] = None,
    reference_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]
```
Check data consistency within and across stores.

##### validate_business_rules()
```python
validate_business_rules(df: pd.DataFrame) -> Dict[str, Any]
```
Validate all enabled business rules.

##### calculate_quality_score()
```python
calculate_quality_score(
    df: pd.DataFrame,
    store_id: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> QualityMetrics
```
Calculate comprehensive quality score across all dimensions.

##### generate_report()
```python
generate_report(
    include_stores: Optional[List[str]] = None,
    format: str = 'dict'
) -> Union[Dict[str, Any], pd.DataFrame, str]
```
Generate quality report in specified format ('dict', 'dataframe', 'html').

### QualityMetrics Class

```python
@dataclass
class QualityMetrics:
    completeness_score: float
    consistency_score: float
    validity_score: float
    accuracy_score: float
    timeliness_score: float
    uniqueness_score: float
    overall_score: float

    missing_count: int
    missing_percentage: float
    outlier_count: int
    outlier_percentage: float
    duplicate_count: int
    invalid_count: int

    issues: List[Dict[str, Any]]
    warnings: List[str]
```

## Quality Dimensions Explained

### 1. Completeness (25% weight)
Measures the proportion of non-missing values.

**Score Calculation:**
```
Completeness = 100 - (missing_percentage)
```

**Thresholds:**
- ≥90%: Excellent
- 80-90%: Good
- 70-80%: Fair
- <70%: Poor

### 2. Consistency (20% weight)
Measures data type consistency, value ranges, and cross-store alignment.

**Checks:**
- Data type consistency
- Value range validation
- Mixed type detection
- Cross-store schema alignment

### 3. Validity (20% weight)
Measures compliance with business rules.

**Score Calculation:**
```
Validity = (passed_rules / total_rules) * 100
```

### 4. Accuracy (15% weight)
Measures data accuracy based on outlier detection.

**Score Calculation:**
```
Accuracy = 100 - outlier_percentage
```

### 5. Timeliness (10% weight)
Measures data freshness.

**Score Calculation:**
```
Timeliness = max(0, 100 - (days_old / 30) * 10)
```

### 6. Uniqueness (10% weight)
Measures duplicate record prevalence.

**Score Calculation:**
```
Uniqueness = 100 - (duplicate_count / total_records) * 100
```

## Validation Workflow

### Recommended Workflow

```python
# 1. Initialize validator
validator = DataQualityValidator(verbose=True)

# 2. Add custom business rules
validator.add_rule(
    name="custom_rule",
    description="Business-specific validation",
    rule_fn=lambda df: df['sales'] > 0
)

# 3. Run initial validation
initial_metrics = validator.validate(df, store_id='store_001')

# 4. Clean data based on findings
if initial_metrics.overall_score < 80:
    # Impute missing values
    df = validator.impute_missing_data(df, strategy=ImputationStrategy.SEASONAL)

    # Handle outliers
    df = validator.handle_outliers(df, method='cap')

    # Remove duplicates
    df = df.drop_duplicates()

# 5. Re-validate
final_metrics = validator.validate(df, store_id='store_001_cleaned')

# 6. Generate report
report = validator.generate_report(format='html')
validator.save_report(report, 'quality_report.html')
```

## Best Practices

### 1. **Set Appropriate Thresholds**
```python
# For high-quality requirements
validator = DataQualityValidator(
    missing_threshold=0.05,  # Max 5% missing
    outlier_threshold=2.5    # Stricter outlier detection
)

# For exploratory analysis
validator = DataQualityValidator(
    missing_threshold=0.3,   # Max 30% missing
    outlier_threshold=3.5    # More lenient
)
```

### 2. **Use Column-Specific Strategies**
```python
# Different imputation for different data types
df_imputed = validator.impute_missing_data(
    df,
    column_strategies={
        'sales': ImputationStrategy.SEASONAL,      # Time-aware
        'price': ImputationStrategy.MEDIAN,        # Robust to outliers
        'category': ImputationStrategy.MODE,       # Most frequent
        'inventory': ImputationStrategy.FORWARD_FILL  # Carry forward
    }
)
```

### 3. **Define Business-Specific Rules**
```python
# Industry-specific validation
validator.add_rule(
    name="promotion_impact",
    description="Promoted items should have higher sales",
    rule_fn=lambda df: (
        (df['promotion'] == 0) |
        (df['sales'] > df['sales'].quantile(0.5))
    ),
    severity="warning"
)
```

### 4. **Monitor Trends Over Time**
```python
# Regular validation
for date in date_range:
    df_date = df[df['date'] == date]
    validator.validate(df_date, store_id=f'store_{date}')

# Analyze trends
history = validator.get_validation_history()
quality_trends = pd.DataFrame([
    {
        'date': h['timestamp'],
        'score': h['metrics']['scores']['overall']
    }
    for h in history
])
```

### 5. **Use Multi-Store Comparison**
```python
# Identify problematic stores
report_df = validator.generate_report(format='dataframe')
low_quality_stores = report_df[report_df['overall'] < 70]

for _, store in low_quality_stores.iterrows():
    print(f"Action needed for {store['store_id']}: {store['overall']:.2f}")
```

## Performance Considerations

### Large Datasets

```python
# Process in chunks for large datasets
chunk_size = 10000

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    metrics = validator.validate(chunk, store_id=f'chunk_{i}')
```

### Parallel Processing

```python
from multiprocessing import Pool

def validate_store(store_id):
    df_store = df[df['store'] == store_id]
    return validator.validate(df_store, store_id=store_id)

# Validate stores in parallel
with Pool(processes=4) as pool:
    results = pool.map(validate_store, store_ids)
```

## Troubleshooting

### Common Issues

**Issue: High missing data score impact**
```python
# Solution: Use more lenient threshold
validator.missing_threshold = 0.5
```

**Issue: False outlier detection**
```python
# Solution: Use more robust method
outliers = validator.detect_outliers(
    df,
    method=OutlierMethod.MODIFIED_Z_SCORE
)
```

**Issue: Business rule too strict**
```python
# Solution: Change to warning
validator.add_rule(
    name="strict_rule",
    description="...",
    rule_fn=lambda df: ...,
    severity="warning"  # Instead of "error"
)
```

## Examples

See `/mnt/d/github/pycaret/examples/data_quality_example.py` for comprehensive examples including:

1. Basic validation
2. Custom business rules
3. Missing data imputation strategies
4. Outlier detection and handling
5. Multi-store analysis
6. Temporal consistency validation
7. Dashboard generation
8. Complete data cleaning workflow

## Testing

Run the test suite:

```bash
pytest tests/test_data_quality.py -v
```

Test coverage includes:
- Missing data detection and imputation
- Outlier identification (IQR, Z-score, Modified Z-score)
- Consistency checks
- Business rule validation
- Quality score calculation
- Report generation
- Edge cases (empty data, single row, all missing)

## Integration with PyCaret

```python
from pycaret.regression import setup, compare_models
from src.validation.data_quality import DataQualityValidator

# Validate before training
validator = DataQualityValidator()
metrics = validator.validate(df)

if metrics.overall_score >= 80:
    # Data is clean, proceed with modeling
    exp = setup(data=df, target='sales')
    best_model = compare_models()
else:
    # Clean data first
    df_clean = validator.impute_missing_data(df)
    df_clean = validator.handle_outliers(df_clean)

    exp = setup(data=df_clean, target='sales')
    best_model = compare_models()
```

## License

MIT License - See LICENSE file for details

## Contributors

PyCaret Development Team

## Support

- GitHub Issues: https://github.com/pycaret/pycaret/issues
- Documentation: https://pycaret.gitbook.io/
- Community: https://github.com/pycaret/pycaret/discussions
