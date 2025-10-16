# Advanced Demand Forecasting System - PyCaret

A comprehensive demand forecasting solution built on PyCaret 3.x Time Series module with support for multiple models, multi-horizon predictions, and advanced feature engineering.

## Features

### Multiple Forecasting Models
- **Statistical Models**: ARIMA, ETS, TBATS, Theta, Seasonal Naive
- **Machine Learning**: XGBoost, LightGBM, CatBoost (with temporal features)
- **Deep Learning Ready**: LSTM support via external integration
- **Prophet**: Facebook Prophet integration (optional)
- **Ensemble Methods**: Simple, weighted, and stacking ensembles

### Multi-Horizon Forecasting
- Configurable forecast horizons (default: 7, 30, 90 days)
- Simultaneous predictions across multiple time periods
- Confidence intervals and prediction bounds for all horizons

### Store & Product Level Predictions
- Hierarchical forecasting support
- Store-level, product-level, and combined forecasts
- Automatic reconciliation of hierarchical predictions

### Advanced Feature Engineering
- **Temporal Features**: Day of week, month, quarter, cyclical encoding
- **Holiday Detection**: US Federal holidays (extensible to other countries)
- **Weather Integration**: Optional weather data support
- **Promotion Tracking**: Promotional event indicators
- **Lag Features**: Multiple lag periods (1, 7, 14, 28, 30, 90 days)
- **Rolling Statistics**: Mean, std, min, max over various windows
- **Seasonal Decomposition**: Trend, seasonal, and residual components

### Model Ensemble & Selection
- Automated model comparison with cross-validation
- Performance-based model selection
- Weighted ensemble with optimized weights
- Stacking ensemble for advanced predictions

### Comprehensive Metrics
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Bias**: Mean error
- **Coverage**: Prediction interval coverage

## Installation

### Core Requirements
```bash
pip install pycaret>=3.4.0
```

### Optional Dependencies
```bash
# For Prophet support
pip install prophet>=1.0.1

# For XGBoost
pip install xgboost>=2.0.0

# For advanced features
pip install statsforecast
```

## Quick Start

### Basic Forecasting

```python
from demand_forecast import quick_forecast
import pandas as pd

# Load your sales data
sales_data = pd.read_csv('sales.csv')

# Generate 30-day forecast
forecast, system = quick_forecast(
    data=sales_data,
    target='sales',
    date_column='date',
    fh=30,
    verbose=True
)

print(forecast.head())
```

### Advanced Usage

```python
from demand_forecast import DemandForecastingSystem

# Initialize system with multiple horizons
system = DemandForecastingSystem(
    forecast_horizons=[7, 30, 90],
    seasonal_period=7,
    confidence_level=0.95
)

# Setup experiment
system.setup(
    data=sales_data,
    target='sales',
    date_column='date',
    fh=90,
    fold=5,
    verbose=True
)

# Compare multiple models
results = system.compare_models(
    n_select=5,
    sort='MAPE',
    verbose=True
)

# Create ensemble
ensemble = system.create_ensemble(
    method='weighted',
    optimize_weights=True
)

# Generate multi-horizon forecasts
forecasts = system.forecast_multi_horizon()

# Evaluate 30-day forecast
metrics_30d = system.evaluate_forecast(forecasts[30])
print(f"30-Day MAPE: {metrics_30d['MAPE']:.2f}%")
print(f"30-Day RMSE: {metrics_30d['RMSE']:.2f}")
```

### Feature Engineering

```python
# Engineer features with holidays and promotions
system = DemandForecastingSystem(forecast_horizons=[30])

# Prepare promotion data
promotions = pd.DataFrame({
    'date': pd.date_range('2024-01-01', '2024-12-31'),
    'is_promotion': [1, 0, 0, ...],  # Binary indicator
    'discount_pct': [0.2, 0, 0, ...]  # Discount percentage
}).set_index('date')

# Engineer features
enriched_data = system.engineer_features(
    data=sales_data.set_index('date'),
    include_holidays=True,
    include_promotions=True,
    promotion_data=promotions,
    holiday_country='US'
)

# Train with enriched features
system.setup(data=enriched_data, target='sales', date_column='date')
system.compare_models()
forecast = system.predict(fh=30)
```

### Hierarchical Forecasting

```python
# Forecast at store and product levels
hierarchical_forecasts = system.forecast_hierarchy(
    data=sales_data,
    hierarchy_columns=['store_id', 'product_id'],
    target='sales',
    date_column='date',
    fh=30,
    reconciliation='ols',
    verbose=True
)

# Access specific forecasts
store_1_product_A = hierarchical_forecasts['store_id=1_product_id=A']
print(store_1_product_A.head())
```

### Model Persistence

```python
# Save trained model
system.save_model('models/demand_forecast_v1.pkl')

# Load for inference
new_system = DemandForecastingSystem()
new_system.load_model('models/demand_forecast_v1.pkl')

# Generate predictions
new_forecast = new_system.predict(fh=30)
```

## API Reference

### DemandForecastingSystem

Main class for demand forecasting.

**Initialization:**
```python
DemandForecastingSystem(
    forecast_horizons=[7, 30, 90],
    seasonal_period=7,
    confidence_level=0.95,
    random_state=42
)
```

**Key Methods:**

- `setup()`: Initialize forecasting experiment
- `engineer_features()`: Create temporal and domain features
- `compare_models()`: Compare multiple forecasting models
- `create_ensemble()`: Build ensemble from top models
- `predict()`: Generate forecast for specified horizon
- `forecast_multi_horizon()`: Generate forecasts for all horizons
- `evaluate_forecast()`: Calculate accuracy metrics
- `forecast_hierarchy()`: Hierarchical forecasting
- `save_model()`: Save model to disk
- `load_model()`: Load model from disk
- `plot_forecast()`: Visualize forecast
- `get_feature_importance()`: Extract feature importance

## Model Selection Guide

### Statistical Models (Best for Clean Patterns)
- **ARIMA**: Stationary data with clear trends
- **ETS**: Data with exponential smoothing patterns
- **TBATS**: Multiple seasonal patterns
- **Theta**: Simple, robust forecasts
- **Seasonal Naive**: Baseline comparison

### Machine Learning Models (Best for Complex Patterns)
- **XGBoost**: Non-linear patterns with many features
- **LightGBM**: Fast training, large datasets
- **CatBoost**: Categorical features, robust to overfitting

### Prophet (Best for Business Data)
- Strong seasonal patterns
- Holiday effects
- Changepoint detection

### Ensemble (Best Overall)
- Combines strengths of multiple models
- Reduces variance
- More robust predictions

## Performance Tips

1. **Data Preparation**
   - Clean outliers before training
   - Handle missing values appropriately
   - Ensure consistent time intervals

2. **Feature Engineering**
   - Start with temporal features
   - Add domain-specific features incrementally
   - Monitor feature importance

3. **Model Selection**
   - Always compare multiple models
   - Use cross-validation for robust evaluation
   - Consider ensemble for production

4. **Hyperparameter Tuning**
   - Use PyCaret's `tune_model()` for optimization
   - Focus on top-performing models
   - Balance accuracy vs. training time

5. **Validation**
   - Use time-based cross-validation
   - Test on hold-out period
   - Monitor prediction interval coverage

## Examples

See the included example scripts:

- `examples/basic_forecast.py`: Simple forecasting workflow
- `examples/advanced_features.py`: Feature engineering examples
- `examples/hierarchical_forecast.py`: Multi-level forecasting
- `examples/ensemble_models.py`: Ensemble creation

## Integration with PyCaret

This module extends PyCaret's time series capabilities with:
- Simplified API for demand forecasting use cases
- Built-in feature engineering pipelines
- Multi-horizon forecast management
- Hierarchical forecasting utilities
- Production-ready model persistence

All PyCaret time series methods remain accessible through the `experiment` attribute.

## Requirements

- Python >= 3.9
- PyCaret >= 3.4.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn < 1.5
- statsmodels >= 0.12.1
- sktime >= 0.31.0
- pmdarima >= 2.0.4

Optional:
- prophet >= 1.0.1
- xgboost >= 2.0.0
- statsforecast
- matplotlib (for plotting)

## License

MIT License - see PyCaret project license

## Contributing

Contributions welcome! Please ensure:
- Code follows PyCaret style guidelines
- Tests pass for all forecasting methods
- Documentation updated for new features
- Examples provided for new functionality

## Support

- PyCaret Documentation: https://pycaret.gitbook.io/
- PyCaret GitHub: https://github.com/pycaret/pycaret
- Issues: https://github.com/pycaret/pycaret/issues

## Citation

```bibtex
@software{pycaret,
  title = {PyCaret: An open source, low-code machine learning library in Python},
  author = {Moez Ali and contributors},
  year = {2020},
  url = {https://github.com/pycaret/pycaret}
}
```
