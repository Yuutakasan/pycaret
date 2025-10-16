# Data Quality Validation Module

## Overview

Comprehensive data quality validation system for retail analytics with automated detection, cleaning, and reporting capabilities.

## Files

- **data_quality.py** - Main validation system implementation
- **README.md** - This file

## Quick Start

```python
from src.validation.data_quality import DataQualityValidator

# Initialize
validator = DataQualityValidator()

# Validate
metrics = validator.validate(df, store_id='store_001')
print(f"Quality Score: {metrics.overall_score:.2f}/100")
```

## Key Features

### 🔍 Detection
- Missing data patterns
- Outliers (IQR, Z-score, Modified Z-score)
- Duplicates
- Type inconsistencies
- Temporal gaps

### 🛠️ Cleaning
- Multiple imputation strategies
- Outlier handling (cap, remove, transform)
- Duplicate removal
- Type correction

### ✅ Validation
- Business rule enforcement
- Temporal consistency
- Cross-store consistency
- Custom validation rules

### 📊 Reporting
- Quality scores (6 dimensions)
- HTML/CSV/DataFrame reports
- Interactive dashboards
- Trend analysis

## Quality Dimensions

1. **Completeness** (25%) - Missing data
2. **Consistency** (20%) - Data types and ranges
3. **Validity** (20%) - Business rules
4. **Accuracy** (15%) - Outliers
5. **Timeliness** (10%) - Data freshness
6. **Uniqueness** (10%) - Duplicates

## Usage Examples

### Basic Validation
```python
validator = DataQualityValidator(verbose=True)
metrics = validator.validate(df, store_id='store_001')
```

### Custom Business Rules
```python
validator.add_rule(
    name="sales_range",
    description="Sales must be positive",
    rule_fn=lambda df: df['sales'] > 0,
    severity="error"
)
```

### Missing Data Imputation
```python
from src.validation.data_quality import ImputationStrategy

df_clean = validator.impute_missing_data(
    df,
    strategy=ImputationStrategy.SEASONAL,
    column_strategies={
        'sales': ImputationStrategy.SEASONAL,
        'price': ImputationStrategy.MEDIAN
    }
)
```

### Outlier Detection
```python
from src.validation.data_quality import OutlierMethod

outliers = validator.detect_outliers(
    df,
    method=OutlierMethod.IQR,
    columns=['sales', 'price']
)
```

### Generate Reports
```python
# HTML report
html_report = validator.generate_report(format='html')
validator.save_report(html_report, 'quality_report.html')

# DataFrame report
df_report = validator.generate_report(format='dataframe')
df_report.to_csv('quality_report.csv')
```

### Create Dashboard
```python
from src.validation.data_quality import create_quality_dashboard

create_quality_dashboard(validator, 'dashboard.html')
```

## API Reference

### DataQualityValidator

**Main Methods:**
- `validate()` - Comprehensive validation
- `detect_missing_data()` - Find missing values
- `impute_missing_data()` - Fill missing values
- `detect_outliers()` - Identify outliers
- `handle_outliers()` - Clean outliers
- `check_consistency()` - Validate consistency
- `validate_business_rules()` - Check business rules
- `calculate_quality_score()` - Calculate quality metrics
- `generate_report()` - Create reports

**Configuration:**
- `missing_threshold` - Max acceptable missing proportion (default: 0.3)
- `outlier_threshold` - Z-score threshold (default: 3.0)
- `temporal_column` - Date column name (default: 'date')
- `store_column` - Store ID column (default: 'store')

## Testing

```bash
pytest tests/test_data_quality.py -v
```

Test coverage:
- Missing data detection and imputation
- Outlier identification (multiple methods)
- Consistency validation
- Business rule validation
- Quality scoring
- Report generation
- Edge cases

## Documentation

Full documentation: `/mnt/d/github/pycaret/docs/data_quality_validation.md`

Examples: `/mnt/d/github/pycaret/examples/data_quality_example.py`

## Dependencies

Core (included in PyCaret):
- pandas
- numpy
- scipy

Optional (for dashboards):
- plotly

## Integration with PyCaret

```python
from pycaret.regression import setup, compare_models
from src.validation.data_quality import DataQualityValidator

# Validate before modeling
validator = DataQualityValidator()
metrics = validator.validate(df)

if metrics.overall_score >= 80:
    exp = setup(data=df, target='sales')
    best_model = compare_models()
else:
    # Clean first
    df_clean = validator.impute_missing_data(df)
    df_clean = validator.handle_outliers(df_clean)
    exp = setup(data=df_clean, target='sales')
    best_model = compare_models()
```

## Best Practices

1. **Set appropriate thresholds** based on use case
2. **Use column-specific strategies** for imputation
3. **Define business-specific rules** for your domain
4. **Monitor trends over time** with validation history
5. **Compare across stores** to identify issues

## Performance Tips

**Large Datasets:**
```python
# Process in chunks
for chunk in pd.read_csv('data.csv', chunksize=10000):
    validator.validate(chunk)
```

**Parallel Processing:**
```python
from multiprocessing import Pool

with Pool(4) as pool:
    results = pool.map(validate_store, store_ids)
```

## License

MIT License

## Support

- GitHub: https://github.com/pycaret/pycaret
- Docs: https://pycaret.gitbook.io/
- Issues: https://github.com/pycaret/pycaret/issues
