"""
Hierarchical Forecasting Example

Demonstrates multi-level forecasting for stores and products
with hierarchical reconciliation.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '..')

from demand_forecast import DemandForecastingSystem


def generate_hierarchical_data(n_days=365, n_stores=3, n_products=4):
    """Generate synthetic sales data for multiple stores and products."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    data_list = []

    for store_id in range(1, n_stores + 1):
        for product_id in range(1, n_products + 1):
            # Store-specific base demand
            store_factor = 1.0 + (store_id - 1) * 0.3

            # Product-specific base demand
            product_factor = 1.0 + (product_id - 1) * 0.2

            # Base demand
            base = 50 * store_factor * product_factor + np.arange(n_days) * 0.1

            # Weekly seasonality
            weekly = 10 * store_factor * np.sin(2 * np.pi * np.arange(n_days) / 7)

            # Monthly seasonality (product-specific)
            monthly = 8 * product_factor * np.sin(2 * np.pi * np.arange(n_days) / 30.5)

            # Random noise
            noise = np.random.normal(0, 5 * store_factor, n_days)

            # Combine
            sales = base + weekly + monthly + noise
            sales = np.maximum(sales, 0)

            # Create records
            for i, date in enumerate(dates):
                data_list.append({
                    'date': date,
                    'store_id': f'STORE_{store_id:02d}',
                    'product_id': f'PRODUCT_{chr(64 + product_id)}',
                    'sales': sales[i]
                })

    return pd.DataFrame(data_list)


def example_simple_hierarchy():
    """Example 1: Basic hierarchical forecasting."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Hierarchical Forecasting")
    print("=" * 70)

    # Generate data
    data = generate_hierarchical_data(n_days=365, n_stores=2, n_products=3)

    print(f"\nDataset overview:")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Stores: {data['store_id'].nunique()}")
    print(f"  Products: {data['product_id'].nunique()}")
    print(f"  Total combinations: {len(data.groupby(['store_id', 'product_id']))}")
    print(f"  Total records: {len(data)}")

    print("\nSample data:")
    print(data.head(10))

    # Initialize system
    system = DemandForecastingSystem(
        forecast_horizons=[30],
        seasonal_period=7
    )

    # Generate hierarchical forecasts
    print("\nGenerating hierarchical forecasts...")
    print("This may take a few minutes...")

    hierarchical_forecasts = system.forecast_hierarchy(
        data=data,
        hierarchy_columns=['store_id', 'product_id'],
        target='sales',
        date_column='date',
        fh=30,
        verbose=True
    )

    print(f"\n✓ Hierarchical forecasting complete")
    print(f"  Generated forecasts for {len(hierarchical_forecasts)} combinations")

    # Display sample forecasts
    print("\nSample forecast (STORE_01 - PRODUCT_A):")
    key = 'store_id=STORE_01_product_id=PRODUCT_A'
    if key in hierarchical_forecasts:
        print(hierarchical_forecasts[key].head())

    return hierarchical_forecasts


def example_store_level_aggregation():
    """Example 2: Forecast at different aggregation levels."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Level Aggregation")
    print("=" * 70)

    # Generate data
    data = generate_hierarchical_data(n_days=365, n_stores=3, n_products=4)

    print("\nForecasting at different aggregation levels:")

    # Level 1: Total (all stores, all products)
    print("\n1. Total level (all stores, all products)")
    total_data = data.groupby('date')['sales'].sum().reset_index()

    system_total = DemandForecastingSystem(forecast_horizons=[30])
    system_total.setup(
        data=total_data,
        target='sales',
        date_column='date',
        fh=30,
        verbose=False
    )
    system_total.compare_models(n_select=1, verbose=False)
    forecast_total = system_total.predict(fh=30)

    print(f"   ✓ Total forecast generated")
    print(f"   Average daily forecast: {forecast_total.iloc[:, 0].mean():.2f}")

    # Level 2: By store (aggregate products)
    print("\n2. Store level (aggregate products)")
    store_forecasts = {}

    for store in data['store_id'].unique():
        store_data = data[data['store_id'] == store].groupby('date')['sales'].sum().reset_index()

        system_store = DemandForecastingSystem(forecast_horizons=[30])
        system_store.setup(
            data=store_data,
            target='sales',
            date_column='date',
            fh=30,
            verbose=False
        )
        system_store.compare_models(n_select=1, verbose=False)
        forecast_store = system_store.predict(fh=30)
        store_forecasts[store] = forecast_store

        print(f"   ✓ {store}: Avg forecast = {forecast_store.iloc[:, 0].mean():.2f}")

    # Level 3: By product (aggregate stores)
    print("\n3. Product level (aggregate stores)")
    product_forecasts = {}

    for product in data['product_id'].unique():
        product_data = data[data['product_id'] == product].groupby('date')['sales'].sum().reset_index()

        system_product = DemandForecastingSystem(forecast_horizons=[30])
        system_product.setup(
            data=product_data,
            target='sales',
            date_column='date',
            fh=30,
            verbose=False
        )
        system_product.compare_models(n_select=1, verbose=False)
        forecast_product = system_product.predict(fh=30)
        product_forecasts[product] = forecast_product

        print(f"   ✓ {product}: Avg forecast = {forecast_product.iloc[:, 0].mean():.2f}")

    # Check consistency
    print("\n4. Checking hierarchical consistency")

    # Sum of store forecasts should equal total
    store_sum = sum([f.iloc[:, 0].sum() for f in store_forecasts.values()])
    total_sum = forecast_total.iloc[:, 0].sum()

    print(f"   Total forecast sum: {total_sum:.2f}")
    print(f"   Sum of store forecasts: {store_sum:.2f}")
    print(f"   Difference: {abs(total_sum - store_sum):.2f} ({abs(total_sum - store_sum) / total_sum * 100:.2f}%)")

    return {
        'total': forecast_total,
        'stores': store_forecasts,
        'products': product_forecasts
    }


def example_bottom_up_reconciliation():
    """Example 3: Bottom-up forecast reconciliation."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Bottom-Up Reconciliation")
    print("=" * 70)

    # Generate data
    data = generate_hierarchical_data(n_days=365, n_stores=2, n_products=3)

    print("\nStep 1: Generate bottom-level forecasts (store-product combinations)")

    # Get all combinations
    combinations = data[['store_id', 'product_id']].drop_duplicates()

    bottom_forecasts = {}

    for idx, row in combinations.iterrows():
        store = row['store_id']
        product = row['product_id']

        # Filter data
        subset = data[(data['store_id'] == store) & (data['product_id'] == product)]

        # Forecast
        system = DemandForecastingSystem(forecast_horizons=[30])
        system.setup(
            data=subset,
            target='sales',
            date_column='date',
            fh=30,
            verbose=False
        )
        system.compare_models(n_select=1, verbose=False)
        forecast = system.predict(fh=30)

        key = f"{store}_{product}"
        bottom_forecasts[key] = forecast

        print(f"   ✓ {store} - {product}: {forecast.iloc[:, 0].mean():.2f}")

    print(f"\n✓ {len(bottom_forecasts)} bottom-level forecasts generated")

    # Step 2: Reconcile to higher levels
    print("\nStep 2: Reconcile to higher levels")

    # Aggregate to store level
    store_reconciled = {}
    for store in data['store_id'].unique():
        store_keys = [k for k in bottom_forecasts.keys() if k.startswith(store)]

        # Sum forecasts
        forecast_sum = sum([bottom_forecasts[k].iloc[:, 0] for k in store_keys])
        store_reconciled[store] = forecast_sum

        print(f"   {store}: {forecast_sum.mean():.2f} (from {len(store_keys)} products)")

    # Aggregate to product level
    product_reconciled = {}
    for product in data['product_id'].unique():
        product_keys = [k for k in bottom_forecasts.keys() if k.endswith(product)]

        # Sum forecasts
        forecast_sum = sum([bottom_forecasts[k].iloc[:, 0] for k in product_keys])
        product_reconciled[product] = forecast_sum

        print(f"   {product}: {forecast_sum.mean():.2f} (from {len(product_keys)} stores)")

    # Aggregate to total
    total_reconciled = sum([f.iloc[:, 0] for f in bottom_forecasts.values()])

    print(f"\nTotal (reconciled): {total_reconciled.mean():.2f}")

    print("\n✓ Bottom-up reconciliation complete")

    return {
        'bottom': bottom_forecasts,
        'stores': store_reconciled,
        'products': product_reconciled,
        'total': total_reconciled
    }


def example_performance_by_hierarchy():
    """Example 4: Compare forecast performance by hierarchy level."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Performance Analysis by Hierarchy Level")
    print("=" * 70)

    # Generate data
    data = generate_hierarchical_data(n_days=365, n_stores=3, n_products=4)

    # Split data
    train_data = data[data['date'] < '2023-11-01']
    test_data = data[data['date'] >= '2023-11-01']

    print(f"\nData split:")
    print(f"  Training: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"  Testing: {test_data['date'].min()} to {test_data['date'].max()}")

    results = []

    # Forecast and evaluate each combination
    combinations = train_data[['store_id', 'product_id']].drop_duplicates()

    print(f"\nEvaluating {len(combinations)} store-product combinations...")

    for idx, row in combinations.iterrows():
        store = row['store_id']
        product = row['product_id']

        # Filter data
        train_subset = train_data[(train_data['store_id'] == store) &
                                  (train_data['product_id'] == product)]
        test_subset = test_data[(test_data['store_id'] == store) &
                               (test_data['product_id'] == product)]

        try:
            # Forecast
            system = DemandForecastingSystem(forecast_horizons=[30])
            system.setup(
                data=train_subset,
                target='sales',
                date_column='date',
                fh=30,
                verbose=False
            )
            system.compare_models(n_select=1, verbose=False)
            forecast = system.predict(fh=30)

            # Evaluate
            forecast['actual'] = test_subset.set_index('date')['sales']
            metrics = system.evaluate_forecast(forecast)

            results.append({
                'Store': store,
                'Product': product,
                'MAPE': metrics.get('MAPE', np.nan),
                'RMSE': metrics.get('RMSE', np.nan),
                'MAE': metrics.get('MAE', np.nan)
            })

        except Exception as e:
            print(f"   ⚠ Error for {store}-{product}: {str(e)}")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("FORECAST PERFORMANCE BY COMBINATION")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nMAPE:")
    print(f"  Mean: {results_df['MAPE'].mean():.2f}%")
    print(f"  Median: {results_df['MAPE'].median():.2f}%")
    print(f"  Std: {results_df['MAPE'].std():.2f}%")
    print(f"  Min: {results_df['MAPE'].min():.2f}%")
    print(f"  Max: {results_df['MAPE'].max():.2f}%")

    print(f"\nRMSE:")
    print(f"  Mean: {results_df['RMSE'].mean():.2f}")
    print(f"  Median: {results_df['RMSE'].median():.2f}")

    # Best and worst performers
    print("\n" + "=" * 70)
    print("TOP 3 BEST PERFORMERS (lowest MAPE)")
    print("=" * 70)
    print(results_df.nsmallest(3, 'MAPE')[['Store', 'Product', 'MAPE']].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 3 WORST PERFORMERS (highest MAPE)")
    print("=" * 70)
    print(results_df.nlargest(3, 'MAPE')[['Store', 'Product', 'MAPE']].to_string(index=False))

    return results_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PyCaret Demand Forecasting - Hierarchical Examples")
    print("=" * 70)

    try:
        hierarchical_forecasts = example_simple_hierarchy()
        multi_level_forecasts = example_store_level_aggregation()
        reconciled_forecasts = example_bottom_up_reconciliation()
        performance_results = example_performance_by_hierarchy()

        print("\n\n" + "=" * 70)
        print("ALL HIERARCHICAL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
