"""
Advanced Feature Engineering Example

Demonstrates advanced feature engineering for demand forecasting
including holidays, weather, promotions, and custom features.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '..')

from demand_forecast import DemandForecastingSystem, forecast_with_features


def generate_advanced_sample_data(n_days=730):
    """Generate synthetic sales data with external factors."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # Base demand
    base = 100 + np.arange(n_days) * 0.15

    # Seasonal patterns
    weekly = 25 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 30.5)
    yearly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)

    # Holiday effect (random spikes)
    np.random.seed(42)
    holiday_effect = np.zeros(n_days)
    holiday_indices = np.random.choice(n_days, size=int(n_days * 0.03), replace=False)
    holiday_effect[holiday_indices] = np.random.uniform(30, 60, len(holiday_indices))

    # Weather effect (temperature)
    temp_effect = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)

    # Promotion effect
    promo_effect = np.zeros(n_days)
    promo_indices = np.random.choice(n_days, size=int(n_days * 0.15), replace=False)
    promo_effect[promo_indices] = np.random.uniform(20, 40, len(promo_indices))

    # Random noise
    noise = np.random.normal(0, 8, n_days)

    # Combine
    sales = base + weekly + monthly + yearly + holiday_effect + temp_effect + promo_effect + noise
    sales = np.maximum(sales, 0)

    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })

    # Create weather data
    weather = pd.DataFrame({
        'date': dates,
        'temperature': 60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 5, n_days),
        'precipitation': np.maximum(0, np.random.normal(0.1, 0.15, n_days)),
        'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 10, n_days)
    }).set_index('date')

    # Create promotion data
    promotions = pd.DataFrame({
        'date': dates,
        'is_promotion': (promo_effect > 0).astype(int),
        'discount_pct': np.where(promo_effect > 0, np.random.uniform(0.1, 0.3, n_days), 0)
    }).set_index('date')

    return df, weather, promotions


def example_feature_engineering():
    """Example 1: Complete feature engineering pipeline."""
    print("=" * 70)
    print("EXAMPLE 1: Feature Engineering Pipeline")
    print("=" * 70)

    # Generate data
    data, weather, promotions = generate_advanced_sample_data(n_days=730)

    # Initialize system
    system = DemandForecastingSystem(
        forecast_horizons=[30],
        seasonal_period=7
    )

    # Engineer features
    print("\nEngineering features...")
    print("  - Temporal features (day, week, month, cyclical)")
    print("  - Holiday indicators")
    print("  - Weather data")
    print("  - Promotion data")
    print("  - Lag features")
    print("  - Rolling statistics")
    print("  - Seasonal decomposition")

    enriched_data = system.engineer_features(
        data=data.set_index('date'),
        include_holidays=True,
        include_weather=True,
        include_promotions=True,
        weather_data=weather,
        promotion_data=promotions
    )

    print(f"\n✓ Feature engineering complete")
    print(f"  Original features: {len(data.columns)}")
    print(f"  Engineered features: {len(enriched_data.columns)}")
    print(f"  Feature names: {system.feature_names[:10]}... (showing first 10)")

    # Display sample
    print("\nSample of engineered features:")
    print(enriched_data[['sales', 'dayofweek', 'is_weekend', 'is_holiday',
                         'temperature', 'is_promotion']].head(10))

    return enriched_data, system


def example_feature_importance():
    """Example 2: Analyze feature importance."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Feature Importance Analysis")
    print("=" * 70)

    # Generate and prepare data
    data, weather, promotions = generate_advanced_sample_data(n_days=730)

    system = DemandForecastingSystem(forecast_horizons=[30])

    # Engineer features
    enriched_data = system.engineer_features(
        data=data.set_index('date'),
        include_holidays=True,
        include_weather=True,
        include_promotions=True,
        weather_data=weather,
        promotion_data=promotions
    )

    # Train with tree-based model to get feature importance
    print("\nTraining model...")
    train_data = enriched_data.iloc[:-30]

    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        verbose=False
    )

    # Use only tree-based models
    results = system.compare_models(
        include=['lightgbm_cds_dt'],
        n_select=1,
        verbose=False
    )

    print("✓ Model trained")

    # Get feature importance
    print("\nExtracting feature importance...")
    importance_df = system.get_feature_importance(top_n=15)

    if len(importance_df) > 0:
        print("\nTop 15 Most Important Features:")
        print("=" * 70)
        for idx, row in importance_df.iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"{row['feature']:25s} {bar} {row['importance']:.4f}")
    else:
        print("Feature importance not available for this model")

    return importance_df


def example_with_custom_features():
    """Example 3: Adding custom domain-specific features."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Custom Domain Features")
    print("=" * 70)

    # Generate data
    data, weather, promotions = generate_advanced_sample_data(n_days=730)

    # Add custom features
    print("\nAdding custom domain features...")

    data_custom = data.copy()
    data_custom['date'] = pd.to_datetime(data_custom['date'])
    data_custom = data_custom.set_index('date')

    # Payday effect (15th and last day of month)
    data_custom['is_payday'] = (
        (data_custom.index.day == 15) |
        (data_custom.index.day >= 28)
    ).astype(int)

    # School holidays (approximate: summer and winter)
    data_custom['is_school_holiday'] = (
        ((data_custom.index.month >= 6) & (data_custom.index.month <= 8)) |  # Summer
        ((data_custom.index.month == 12) & (data_custom.index.day >= 20)) |  # Winter
        ((data_custom.index.month == 1) & (data_custom.index.day <= 5))
    ).astype(int)

    # Special events (Black Friday approximation)
    data_custom['is_black_friday'] = (
        (data_custom.index.month == 11) &
        (data_custom.index.day >= 23) &
        (data_custom.index.day <= 29) &
        (data_custom.index.dayofweek == 4)  # Friday
    ).astype(int)

    # Competitor activity (simulated)
    np.random.seed(42)
    data_custom['competitor_promo'] = np.random.binomial(1, 0.1, len(data_custom))

    print("  ✓ Payday indicators")
    print("  ✓ School holiday flags")
    print("  ✓ Black Friday detection")
    print("  ✓ Competitor activity")

    # Initialize and train
    system = DemandForecastingSystem(forecast_horizons=[30])

    train_data = data_custom.iloc[:-30]

    system.setup(
        data=train_data,
        target='sales',
        date_column='date',
        fh=30,
        verbose=False
    )

    results = system.compare_models(n_select=3, verbose=False)

    print("\n✓ Model trained with custom features")
    print("\nTop 3 Models:")
    print(results.head(3)[['Model', 'MAPE', 'RMSE']])

    # Generate forecast
    forecast = system.predict(fh=30)

    print(f"\n✓ 30-day forecast generated")

    return forecast, system


def example_feature_ablation():
    """Example 4: Feature ablation study."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Feature Ablation Study")
    print("=" * 70)

    data, weather, promotions = generate_advanced_sample_data(n_days=730)

    # Test different feature combinations
    feature_configs = [
        {'name': 'Baseline (no features)', 'holidays': False, 'weather': False, 'promotions': False},
        {'name': 'With holidays', 'holidays': True, 'weather': False, 'promotions': False},
        {'name': 'With weather', 'holidays': False, 'weather': True, 'promotions': False},
        {'name': 'With promotions', 'holidays': False, 'weather': False, 'promotions': True},
        {'name': 'All features', 'holidays': True, 'weather': True, 'promotions': True},
    ]

    results_summary = []

    for config in feature_configs:
        print(f"\nTesting: {config['name']}")

        system = DemandForecastingSystem(forecast_horizons=[30])

        # Engineer features based on config
        enriched_data = system.engineer_features(
            data=data.set_index('date'),
            include_holidays=config['holidays'],
            include_weather=config['weather'],
            include_promotions=config['promotions'],
            weather_data=weather if config['weather'] else None,
            promotion_data=promotions if config['promotions'] else None
        )

        train_data = enriched_data.iloc[:-30]
        test_data = enriched_data.iloc[-30:]

        system.setup(
            data=train_data,
            target='sales',
            date_column='date',
            fh=30,
            verbose=False
        )

        # Use single model for fair comparison
        model_results = system.compare_models(
            include=['lr_cds_dt'],
            n_select=1,
            verbose=False
        )

        # Predict and evaluate
        forecast = system.predict(fh=30)
        forecast['actual'] = test_data['sales']
        metrics = system.evaluate_forecast(forecast)

        results_summary.append({
            'Configuration': config['name'],
            'Features': len(enriched_data.columns),
            'MAPE': metrics.get('MAPE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAE': metrics.get('MAE', np.nan)
        })

        print(f"  Features: {len(enriched_data.columns)}")
        print(f"  MAPE: {metrics.get('MAPE', np.nan):.2f}%")

    # Summary table
    print("\n" + "=" * 70)
    print("FEATURE ABLATION RESULTS")
    print("=" * 70)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # Find best configuration
    best_config = summary_df.loc[summary_df['MAPE'].idxmin()]
    print(f"\n✓ Best configuration: {best_config['Configuration']}")
    print(f"  MAPE: {best_config['MAPE']:.2f}%")

    return summary_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PyCaret Demand Forecasting - Advanced Feature Examples")
    print("=" * 70)

    try:
        enriched_data, system1 = example_feature_engineering()
        importance_df = example_feature_importance()
        forecast, system3 = example_with_custom_features()
        ablation_results = example_feature_ablation()

        print("\n\n" + "=" * 70)
        print("ALL ADVANCED EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
