"""
Ensemble Models Example

Demonstrates creating and optimizing ensemble forecasting models
for improved prediction accuracy.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '..')

from demand_forecast import DemandForecastingSystem


def generate_sample_data(n_days=730):
    """Generate synthetic sales data with complex patterns."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # Multiple seasonal components
    base = 100 + np.arange(n_days) * 0.2
    weekly = 25 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 30.5)
    yearly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)

    # Non-linear trend
    nonlinear = 10 * np.sin(np.arange(n_days) / 50)

    # Random events
    np.random.seed(42)
    events = np.zeros(n_days)
    event_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
    events[event_indices] = np.random.uniform(20, 50, len(event_indices))

    # Noise
    noise = np.random.normal(0, 8, n_days)

    # Combine
    sales = base + weekly + monthly + yearly + nonlinear + events + noise
    sales = np.maximum(sales, 0)

    return pd.DataFrame({
        'date': dates,
        'sales': sales
    })


def example_simple_ensemble():
    """Example 1: Create simple averaging ensemble."""
    print("=" * 70)
    print("EXAMPLE 1: Simple Averaging Ensemble")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=730)
    train_data = data.iloc[:-30]
    test_data = data.iloc[-30:]

    print(f"\nData: {len(train_data)} training days, {len(test_data)} test days")

    # Initialize system
    system = DemandForecastingSystem(forecast_horizons=[30])

    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        verbose=False
    )

    # Compare models and select top 5
    print("\nComparing models...")
    results = system.compare_models(
        include=['naive', 'snaive', 'arima', 'ets', 'theta', 'lr_cds_dt',
                'en_cds_dt', 'lightgbm_cds_dt'],
        n_select=5,
        verbose=False
    )

    print("\nTop 5 Models:")
    print(results.head(5)[['Model', 'MAPE', 'RMSE', 'MAE']])

    # Create simple ensemble (equal weights)
    print("\nCreating simple averaging ensemble...")
    ensemble_simple = system.create_ensemble(
        method='simple',
        optimize_weights=False
    )

    print("✓ Simple ensemble created")

    # Generate predictions
    forecast_ensemble = system.predict(fh=30, model=ensemble_simple)
    forecast_ensemble['actual'] = test_data.set_index('date')['sales']

    # Evaluate
    metrics_ensemble = system.evaluate_forecast(forecast_ensemble)

    print("\n" + "=" * 70)
    print("ENSEMBLE PERFORMANCE")
    print("=" * 70)
    for metric, value in metrics_ensemble.items():
        if metric != 'n_samples':
            print(f"{metric:15s}: {value:10.2f}")

    return ensemble_simple, forecast_ensemble


def example_weighted_ensemble():
    """Example 2: Create weighted ensemble with optimized weights."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Weighted Ensemble with Optimization")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=730)
    train_data = data.iloc[:-30]
    test_data = data.iloc[-30:]

    # Initialize system
    system = DemandForecastingSystem(forecast_horizons=[30])

    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        fold=5,
        verbose=False
    )

    # Compare models
    print("\nComparing models with 5-fold CV...")
    results = system.compare_models(
        include=['arima', 'ets', 'theta', 'lr_cds_dt', 'lightgbm_cds_dt'],
        n_select=5,
        verbose=False
    )

    print("\nModel Performance:")
    print(results[['Model', 'MAPE', 'RMSE']].to_string(index=False))

    # Create weighted ensemble
    print("\nCreating weighted ensemble with optimized weights...")
    ensemble_weighted = system.create_ensemble(
        method='weighted',
        optimize_weights=True
    )

    print("\n✓ Weighted ensemble created")
    print("\nOptimized Weights:")
    for i, weight in enumerate(system.ensemble_weights):
        model_name = results.iloc[i]['Model']
        print(f"  {model_name:20s}: {weight:.4f}")

    # Generate predictions
    forecast_weighted = system.predict(fh=30, model=ensemble_weighted)
    forecast_weighted['actual'] = test_data.set_index('date')['sales']

    # Evaluate
    metrics_weighted = system.evaluate_forecast(forecast_weighted)

    print("\n" + "=" * 70)
    print("WEIGHTED ENSEMBLE PERFORMANCE")
    print("=" * 70)
    for metric, value in metrics_weighted.items():
        if metric != 'n_samples':
            print(f"{metric:15s}: {value:10.2f}")

    return ensemble_weighted, forecast_weighted


def example_ensemble_comparison():
    """Example 3: Compare different ensemble strategies."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Ensemble Strategy Comparison")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=730)
    train_data = data.iloc[:-30]
    test_data = data.iloc[-30:]

    ensemble_results = []

    # Strategy 1: Individual best model
    print("\n1. Best Individual Model")
    system1 = DemandForecastingSystem(forecast_horizons=[30])
    system1.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)
    results1 = system1.compare_models(n_select=1, verbose=False)

    forecast1 = system1.predict(fh=30)
    forecast1['actual'] = test_data.set_index('date')['sales']
    metrics1 = system1.evaluate_forecast(forecast1)

    ensemble_results.append({
        'Strategy': 'Best Model Only',
        'Model': results1.iloc[0]['Model'],
        'MAPE': metrics1.get('MAPE', np.nan),
        'RMSE': metrics1.get('RMSE', np.nan),
        'MAE': metrics1.get('MAE', np.nan)
    })

    print(f"   Model: {results1.iloc[0]['Model']}")
    print(f"   MAPE: {metrics1.get('MAPE', np.nan):.2f}%")

    # Strategy 2: Simple ensemble (top 3)
    print("\n2. Simple Ensemble (Top 3)")
    system2 = DemandForecastingSystem(forecast_horizons=[30])
    system2.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)
    system2.compare_models(n_select=3, verbose=False)
    ensemble2 = system2.create_ensemble(method='simple', optimize_weights=False)

    forecast2 = system2.predict(fh=30, model=ensemble2)
    forecast2['actual'] = test_data.set_index('date')['sales']
    metrics2 = system2.evaluate_forecast(forecast2)

    ensemble_results.append({
        'Strategy': 'Simple Ensemble (3)',
        'Model': 'Blend-3',
        'MAPE': metrics2.get('MAPE', np.nan),
        'RMSE': metrics2.get('RMSE', np.nan),
        'MAE': metrics2.get('MAE', np.nan)
    })

    print(f"   MAPE: {metrics2.get('MAPE', np.nan):.2f}%")

    # Strategy 3: Simple ensemble (top 5)
    print("\n3. Simple Ensemble (Top 5)")
    system3 = DemandForecastingSystem(forecast_horizons=[30])
    system3.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)
    system3.compare_models(n_select=5, verbose=False)
    ensemble3 = system3.create_ensemble(method='simple', optimize_weights=False)

    forecast3 = system3.predict(fh=30, model=ensemble3)
    forecast3['actual'] = test_data.set_index('date')['sales']
    metrics3 = system3.evaluate_forecast(forecast3)

    ensemble_results.append({
        'Strategy': 'Simple Ensemble (5)',
        'Model': 'Blend-5',
        'MAPE': metrics3.get('MAPE', np.nan),
        'RMSE': metrics3.get('RMSE', np.nan),
        'MAE': metrics3.get('MAE', np.nan)
    })

    print(f"   MAPE: {metrics3.get('MAPE', np.nan):.2f}%")

    # Strategy 4: Weighted ensemble (top 3)
    print("\n4. Weighted Ensemble (Top 3)")
    system4 = DemandForecastingSystem(forecast_horizons=[30])
    system4.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)
    system4.compare_models(n_select=3, verbose=False)
    ensemble4 = system4.create_ensemble(method='weighted', optimize_weights=True)

    forecast4 = system4.predict(fh=30, model=ensemble4)
    forecast4['actual'] = test_data.set_index('date')['sales']
    metrics4 = system4.evaluate_forecast(forecast4)

    ensemble_results.append({
        'Strategy': 'Weighted Ensemble (3)',
        'Model': 'Weighted-3',
        'MAPE': metrics4.get('MAPE', np.nan),
        'RMSE': metrics4.get('RMSE', np.nan),
        'MAE': metrics4.get('MAE', np.nan)
    })

    print(f"   MAPE: {metrics4.get('MAPE', np.nan):.2f}%")
    print(f"   Weights: {system4.ensemble_weights}")

    # Strategy 5: Weighted ensemble (top 5)
    print("\n5. Weighted Ensemble (Top 5)")
    system5 = DemandForecastingSystem(forecast_horizons=[30])
    system5.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)
    system5.compare_models(n_select=5, verbose=False)
    ensemble5 = system5.create_ensemble(method='weighted', optimize_weights=True)

    forecast5 = system5.predict(fh=30, model=ensemble5)
    forecast5['actual'] = test_data.set_index('date')['sales']
    metrics5 = system5.evaluate_forecast(forecast5)

    ensemble_results.append({
        'Strategy': 'Weighted Ensemble (5)',
        'Model': 'Weighted-5',
        'MAPE': metrics5.get('MAPE', np.nan),
        'RMSE': metrics5.get('RMSE', np.nan),
        'MAE': metrics5.get('MAE', np.nan)
    })

    print(f"   MAPE: {metrics5.get('MAPE', np.nan):.2f}%")
    print(f"   Weights: {system5.ensemble_weights}")

    # Summary comparison
    results_df = pd.DataFrame(ensemble_results)

    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGY COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Find best strategy
    best_idx = results_df['MAPE'].idxmin()
    best_strategy = results_df.iloc[best_idx]

    print("\n" + "=" * 70)
    print(f"✓ BEST STRATEGY: {best_strategy['Strategy']}")
    print(f"  MAPE: {best_strategy['MAPE']:.2f}%")
    print(f"  RMSE: {best_strategy['RMSE']:.2f}")
    print("=" * 70)

    return results_df


def example_ensemble_diversity():
    """Example 4: Analyze ensemble diversity."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Ensemble Diversity Analysis")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=730)
    train_data = data.iloc[:-30]
    test_data = data.iloc[-30:]

    # Initialize system
    system = DemandForecastingSystem(forecast_horizons=[30])
    system.setup(data=train_data, target='sales', date_column='date', fh=30, verbose=False)

    # Compare diverse set of models
    print("\nTraining diverse model types...")
    results = system.compare_models(
        include=['naive', 'arima', 'ets', 'lr_cds_dt', 'lightgbm_cds_dt'],
        n_select=5,
        verbose=False
    )

    print("\nModel Types:")
    print(results[['Model', 'MAPE']].to_string(index=False))

    # Generate individual predictions
    print("\nGenerating individual model predictions...")
    individual_forecasts = {}

    for model_key, model in system.models.items():
        forecast = system.predict(fh=30, model=model)
        individual_forecasts[model_key] = forecast.iloc[:, 0].values

    # Calculate prediction diversity
    print("\nCalculating prediction diversity...")

    # Pairwise correlations
    forecast_matrix = np.array(list(individual_forecasts.values()))
    correlations = np.corrcoef(forecast_matrix)

    print("\nPairwise Prediction Correlations:")
    model_names = list(individual_forecasts.keys())
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                print(f"  {name1} vs {name2}: {correlations[i, j]:.4f}")

    # Average correlation (measure of diversity)
    avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
    print(f"\nAverage Correlation: {avg_correlation:.4f}")
    print(f"Diversity Score: {1 - avg_correlation:.4f} (higher is more diverse)")

    # Create ensemble
    print("\nCreating ensemble from diverse models...")
    ensemble = system.create_ensemble(method='weighted', optimize_weights=True)

    forecast_ensemble = system.predict(fh=30, model=ensemble)
    forecast_ensemble['actual'] = test_data.set_index('date')['sales']
    metrics = system.evaluate_forecast(forecast_ensemble)

    print("\nEnsemble Performance:")
    print(f"  MAPE: {metrics.get('MAPE', np.nan):.2f}%")
    print(f"  RMSE: {metrics.get('RMSE', np.nan):.2f}")

    # Compare to average of individual models
    individual_mapes = [results.iloc[i]['MAPE'] for i in range(len(system.models))]
    avg_individual_mape = np.mean(individual_mapes)

    print(f"\nComparison:")
    print(f"  Average individual MAPE: {avg_individual_mape:.2f}%")
    print(f"  Ensemble MAPE: {metrics.get('MAPE', np.nan):.2f}%")
    print(f"  Improvement: {avg_individual_mape - metrics.get('MAPE', 0):.2f}%")

    return correlations, metrics


def example_multi_horizon_ensemble():
    """Example 5: Ensemble for multiple forecast horizons."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Horizon Ensemble")
    print("=" * 70)

    # Generate data
    data = generate_sample_data(n_days=730)

    # Initialize with multiple horizons
    system = DemandForecastingSystem(forecast_horizons=[7, 30, 90])

    print("\nSetting up multi-horizon forecasting system...")
    print("  Horizons: 7-day, 30-day, 90-day")

    system.setup(
        data=data.iloc[:-90],
        target='sales',
        date_column='date',
        fh=90,
        verbose=False
    )

    # Compare models
    print("\nComparing models...")
    results = system.compare_models(n_select=5, verbose=False)
    print(results[['Model', 'MAPE', 'RMSE']].head())

    # Create ensemble
    print("\nCreating weighted ensemble...")
    ensemble = system.create_ensemble(method='weighted', optimize_weights=True)

    # Generate multi-horizon forecasts
    print("\nGenerating forecasts for all horizons...")
    forecasts = system.forecast_multi_horizon(model=ensemble)

    # Evaluate each horizon
    print("\n" + "=" * 70)
    print("MULTI-HORIZON ENSEMBLE PERFORMANCE")
    print("=" * 70)

    for horizon in [7, 30, 90]:
        forecast = forecasts[horizon]

        # Add actuals
        actual_start = -90 + (90 - horizon)
        actual_end = actual_start + horizon if horizon < 90 else None
        forecast['actual'] = data.iloc[actual_start:actual_end].set_index('date')['sales']

        # Evaluate
        metrics = system.evaluate_forecast(forecast)

        print(f"\n{horizon}-Day Forecast:")
        print(f"  MAPE: {metrics.get('MAPE', np.nan):.2f}%")
        print(f"  RMSE: {metrics.get('RMSE', np.nan):.2f}")
        print(f"  MAE: {metrics.get('MAE', np.nan):.2f}")
        if 'Coverage' in metrics:
            print(f"  95% CI Coverage: {metrics['Coverage']:.2f}%")

    return forecasts


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PyCaret Demand Forecasting - Ensemble Examples")
    print("=" * 70)

    try:
        ensemble1, forecast1 = example_simple_ensemble()
        ensemble2, forecast2 = example_weighted_ensemble()
        comparison_results = example_ensemble_comparison()
        correlations, diversity_metrics = example_ensemble_diversity()
        multi_horizon_forecasts = example_multi_horizon_ensemble()

        print("\n\n" + "=" * 70)
        print("ALL ENSEMBLE EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
