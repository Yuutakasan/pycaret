"""
Basic Demand Forecasting Example

Demonstrates simple demand forecasting workflow with PyCaret.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, '..')

from demand_forecast import DemandForecastingSystem, quick_forecast


def generate_sample_data(n_days=365):
    """Generate synthetic sales data for demonstration."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # Base demand with trend
    base_demand = 100 + np.arange(n_days) * 0.2

    # Weekly seasonality (higher on weekends)
    weekly_pattern = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)

    # Monthly seasonality
    monthly_pattern = 15 * np.sin(2 * np.pi * np.arange(n_days) / 30)

    # Random noise
    noise = np.random.normal(0, 10, n_days)

    # Combine components
    sales = base_demand + weekly_pattern + monthly_pattern + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative

    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'store_id': 'STORE_001',
        'product_id': 'PRODUCT_A'
    })

    return df


def example_quick_forecast():
    """Example 1: Quick forecasting with automatic model selection."""
    print("=" * 70)
    print("EXAMPLE 1: Quick Forecast")
    print("=" * 70)

    # Generate sample data
    data = generate_sample_data(n_days=365)

    # Split into train and test
    train_data = data.iloc[:-30]  # Hold out last 30 days
    test_data = data.iloc[-30:]

    print(f"\nTraining data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")

    # Quick forecast
    forecast, system = quick_forecast(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        seasonal_period=7,
        verbose=True
    )

    print("\nForecast (first 5 days):")
    print(forecast.head())

    # Evaluate on test data
    forecast['actual'] = test_data.set_index('date')['sales']
    metrics = system.evaluate_forecast(forecast)

    print("\n" + "-" * 70)
    print("FORECAST EVALUATION")
    print("-" * 70)
    for metric, value in metrics.items():
        if metric != 'n_samples':
            print(f"{metric:15s}: {value:10.2f}")
        else:
            print(f"{metric:15s}: {value:10d}")

    return forecast, system


def example_step_by_step():
    """Example 2: Step-by-step forecasting process."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Step-by-Step Forecasting")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=365)
    train_data = data.iloc[:-30]

    # Initialize system
    system = DemandForecastingSystem(
        forecast_horizons=[7, 30, 90],
        seasonal_period=7,
        confidence_level=0.95
    )

    # Setup experiment
    print("\nStep 1: Setup experiment")
    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=90,
        fold=3,
        verbose=False
    )
    print("✓ Setup complete")

    # Compare models
    print("\nStep 2: Compare models")
    results = system.compare_models(
        include=['naive', 'snaive', 'arima', 'ets', 'lr_cds_dt', 'lightgbm_cds_dt'],
        n_select=3,
        verbose=False
    )
    print("\nTop 3 Models:")
    print(results.head(3)[['Model', 'MAPE', 'RMSE', 'MAE']])

    # Create ensemble
    print("\nStep 3: Create ensemble")
    ensemble = system.create_ensemble(method='weighted', optimize_weights=True)
    print("✓ Ensemble created")
    print(f"  Ensemble weights: {system.ensemble_weights}")

    # Multi-horizon forecasts
    print("\nStep 4: Generate multi-horizon forecasts")
    forecasts = system.forecast_multi_horizon()

    for horizon, forecast_df in forecasts.items():
        print(f"\n  {horizon}-day forecast generated: {len(forecast_df)} predictions")

    # Evaluate 30-day forecast
    print("\nStep 5: Evaluate 30-day forecast")
    forecast_30d = forecasts[30]
    forecast_30d['actual'] = data.iloc[-90:-60].set_index('date')['sales']
    metrics = system.evaluate_forecast(forecast_30d)

    print("\n30-Day Forecast Metrics:")
    for metric, value in metrics.items():
        if metric != 'n_samples':
            print(f"  {metric}: {value:.2f}")

    return system, forecasts


def example_model_comparison():
    """Example 3: Detailed model comparison."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Model Comparison")
    print("=" * 70)

    data = generate_sample_data(n_days=365)
    train_data = data.iloc[:-30]

    system = DemandForecastingSystem(forecast_horizons=[30])

    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        fold=5,
        verbose=False
    )

    # Compare all available models
    print("\nComparing models with 5-fold cross-validation...")
    results = system.compare_models(
        n_select=10,
        sort='MAPE',
        verbose=False
    )

    print("\nModel Comparison Results:")
    print("=" * 70)
    print(results[['Model', 'MAPE', 'RMSE', 'MAE', 'R2']].to_string(index=False))

    return results


def example_save_load():
    """Example 4: Save and load model."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Save and Load Model")
    print("=" * 70)

    data = generate_sample_data(n_days=365)

    # Train and save
    print("\nTraining model...")
    system = DemandForecastingSystem()
    system.setup(data=data, target='sales', date_column='date', verbose=False)
    system.compare_models(n_select=1, verbose=False)

    model_path = 'demand_forecast_model.pkl'
    print(f"\nSaving model to {model_path}...")
    system.save_model(model_path)

    # Load and predict
    print(f"\nLoading model from {model_path}...")
    new_system = DemandForecastingSystem()
    new_system.load_model(model_path)

    print("\nGenerating forecast with loaded model...")
    forecast = new_system.predict(fh=30)

    print(f"\nForecast generated: {len(forecast)} days")
    print("\nFirst 5 predictions:")
    print(forecast.head())

    # Clean up
    import os
    if os.path.exists(model_path + '.pkl'):
        os.remove(model_path + '.pkl')
    if os.path.exists(model_path.replace('.pkl', '_metadata.pkl')):
        os.remove(model_path.replace('.pkl', '_metadata.pkl'))
    print("\n✓ Model files cleaned up")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PyCaret Demand Forecasting - Basic Examples")
    print("=" * 70)

    try:
        # Run examples
        forecast1, system1 = example_quick_forecast()
        system2, forecasts2 = example_step_by_step()
        results3 = example_model_comparison()
        example_save_load()

        print("\n\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
