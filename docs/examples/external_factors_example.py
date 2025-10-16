"""
External Factors Analysis - Complete Example

This example demonstrates how to use the external factors module
for Japanese convenience store sales forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import external factors module
import sys
sys.path.append('/mnt/d/github/pycaret')

from src.analysis.external_factors import (
    WeatherAPIClient,
    CalendarAPIClient,
    EconomicDataClient,
    ExternalFactorsAnalyzer
)


def generate_sample_sales_data(start_date, end_date):
    """Generate synthetic sales data for demonstration"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Base sales with trend and seasonality
    n_days = len(date_range)
    trend = np.linspace(100000, 120000, n_days)

    # Weekly seasonality (higher on weekends)
    weekly_pattern = np.array([0.9, 0.95, 1.0, 1.0, 1.05, 1.15, 1.1])
    seasonality = np.tile(weekly_pattern, n_days // 7 + 1)[:n_days]

    # Random variation
    noise = np.random.normal(0, 5000, n_days)

    sales = trend * seasonality + noise

    df = pd.DataFrame({
        'date': date_range,
        'sales': sales,
        'store_id': 'TOKYO_001'
    })

    return df


def main():
    """Run complete external factors analysis example"""

    print("=" * 80)
    print("External Factors Analysis - Complete Example")
    print("=" * 80)

    # 1. Generate sample sales data
    print("\n1. Generating sample sales data...")
    print("-" * 80)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    sales_df = generate_sample_sales_data(start_date, end_date)

    print(f"Generated sales data: {len(sales_df)} days")
    print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"Average daily sales: ¥{sales_df['sales'].mean():,.0f}")
    print(f"\nFirst few rows:")
    print(sales_df.head())

    # 2. Initialize API clients
    print("\n2. Initializing API clients...")
    print("-" * 80)

    # Use free APIs (no API key required)
    weather_client = WeatherAPIClient(provider="open-meteo")
    calendar_client = CalendarAPIClient(provider="nager")
    econ_client = EconomicDataClient(provider="worldbank")

    print("✓ Weather client: Open-Meteo (free)")
    print("✓ Calendar client: Nager.Date (free)")
    print("✓ Economic client: World Bank (free)")

    # 3. Create analyzer
    print("\n3. Creating External Factors Analyzer...")
    print("-" * 80)

    analyzer = ExternalFactorsAnalyzer(
        weather_client=weather_client,
        calendar_client=calendar_client,
        economic_client=econ_client
    )

    print("✓ Analyzer initialized")

    # 4. Integrate calendar events
    print("\n4. Integrating calendar events...")
    print("-" * 80)

    try:
        sales_with_calendar = analyzer.integrate_calendar_events(
            sales_df,
            date_column='date'
        )

        new_features = set(sales_with_calendar.columns) - set(sales_df.columns)
        print(f"✓ Added {len(new_features)} calendar features:")
        for feature in sorted(new_features):
            print(f"  - {feature}")

        # Show holiday impact
        holiday_sales = sales_with_calendar[
            sales_with_calendar['is_holiday'] == 1
        ]['sales'].mean()
        non_holiday_sales = sales_with_calendar[
            sales_with_calendar['is_holiday'] == 0
        ]['sales'].mean()

        print(f"\nHoliday Impact:")
        print(f"  Average sales on holidays: ¥{holiday_sales:,.0f}")
        print(f"  Average sales on non-holidays: ¥{non_holiday_sales:,.0f}")
        print(f"  Difference: {((holiday_sales / non_holiday_sales - 1) * 100):.1f}%")

    except Exception as e:
        print(f"✗ Error integrating calendar events: {e}")
        sales_with_calendar = sales_df

    # 5. Integrate weather data (sample - limited by API)
    print("\n5. Integrating weather data (sample)...")
    print("-" * 80)

    # Tokyo coordinates
    tokyo_lat, tokyo_lon = 35.6762, 139.6503

    try:
        # Get current weather as demonstration
        current_weather = weather_client.get_current_weather(tokyo_lat, tokyo_lon)
        print(f"✓ Current weather in Tokyo:")
        print(f"  Temperature: {current_weather.temperature}°C")
        print(f"  Condition: {current_weather.condition}")
        print(f"  Humidity: {current_weather.humidity}%")
        print(f"  Precipitation: {current_weather.precipitation}mm")

        # Note: Historical weather for full year requires API calls
        print("\n  Note: Full historical weather integration requires API calls")
        print("  For production use, fetch historical data and cache it")

    except Exception as e:
        print(f"✗ Error fetching weather: {e}")

    # 6. Integrate economic indicators
    print("\n6. Integrating economic indicators...")
    print("-" * 80)

    try:
        sales_with_econ = analyzer.integrate_economic_indicators(
            sales_with_calendar,
            date_column='date'
        )

        new_features = set(sales_with_econ.columns) - set(sales_with_calendar.columns)
        print(f"✓ Added {len(new_features)} economic features:")
        for feature in sorted(new_features):
            non_null = sales_with_econ[feature].notna().sum()
            print(f"  - {feature} ({non_null} non-null values)")

    except Exception as e:
        print(f"✗ Error integrating economic indicators: {e}")
        sales_with_econ = sales_with_calendar

    # 7. Analyze correlations
    print("\n7. Analyzing correlations with sales...")
    print("-" * 80)

    try:
        correlations = analyzer.calculate_correlations(
            sales_with_econ,
            target_column='sales'
        )

        print("✓ Top 10 correlated features:")
        print("\n" + correlations[
            ['feature', 'pearson_correlation', 'pearson_pvalue', 'is_significant']
        ].head(10).to_string(index=False))

        significant_features = correlations[correlations['is_significant']].shape[0]
        print(f"\nTotal significant correlations (p < 0.05): {significant_features}")

    except Exception as e:
        print(f"✗ Error calculating correlations: {e}")

    # 8. Quantify impact
    print("\n8. Quantifying feature impact...")
    print("-" * 80)

    try:
        impact = analyzer.quantify_impact(
            sales_with_econ,
            target_column='sales',
            method='regression'
        )

        print("✓ Top 10 important features (Random Forest):")
        for i, (feature, importance) in enumerate(list(impact.items())[:10], 1):
            print(f"  {i:2d}. {feature:30s}: {importance:.4f}")

    except Exception as e:
        print(f"✗ Error quantifying impact: {e}")

    # 9. Create comprehensive summary
    print("\n9. Creating feature summary...")
    print("-" * 80)

    try:
        summary = analyzer.create_feature_summary(
            sales_with_econ,
            target_column='sales'
        )

        print(f"✓ Feature summary created: {summary.shape[0]} features analyzed")
        print("\nTop 5 features by importance:")
        print("\n" + summary[
            ['feature', 'pearson_correlation', 'rf_importance', 'is_significant']
        ].head(5).to_string(index=False))

        # Save summary
        output_file = '/mnt/d/github/pycaret/docs/external_factors_summary.csv'
        summary.to_csv(output_file, index=False)
        print(f"\n✓ Summary saved to: {output_file}")

    except Exception as e:
        print(f"✗ Error creating summary: {e}")

    # 10. Visualization (optional)
    print("\n10. Creating visualizations...")
    print("-" * 80)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sales over time
        axes[0, 0].plot(sales_with_econ['date'], sales_with_econ['sales'])
        axes[0, 0].set_title('Daily Sales Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales (¥)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Holiday vs Non-Holiday
        holiday_data = [
            sales_with_econ[sales_with_econ['is_holiday'] == 0]['sales'].values,
            sales_with_econ[sales_with_econ['is_holiday'] == 1]['sales'].values
        ]
        axes[0, 1].boxplot(holiday_data, labels=['Non-Holiday', 'Holiday'])
        axes[0, 1].set_title('Sales Distribution: Holiday vs Non-Holiday')
        axes[0, 1].set_ylabel('Sales (¥)')

        # Days to holiday effect
        axes[1, 0].scatter(
            sales_with_econ['days_to_holiday'],
            sales_with_econ['sales'],
            alpha=0.5
        )
        axes[1, 0].set_title('Sales vs Days to Nearest Holiday')
        axes[1, 0].set_xlabel('Days to Holiday')
        axes[1, 0].set_ylabel('Sales (¥)')

        # Event impact
        if 'event_impact' in sales_with_econ.columns:
            impact_groups = sales_with_econ.groupby('event_impact')['sales'].mean()
            axes[1, 1].bar(impact_groups.index, impact_groups.values)
            axes[1, 1].set_title('Average Sales by Event Impact Level')
            axes[1, 1].set_xlabel('Event Impact Level')
            axes[1, 1].set_ylabel('Average Sales (¥)')

        plt.tight_layout()

        output_plot = '/mnt/d/github/pycaret/docs/external_factors_analysis.png'
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_plot}")

    except Exception as e:
        print(f"✗ Error creating visualization: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nFinal dataset shape: {sales_with_econ.shape}")
    print(f"Original features: {len(sales_df.columns)}")
    print(f"Total features: {len(sales_with_econ.columns)}")
    print(f"New features added: {len(sales_with_econ.columns) - len(sales_df.columns)}")

    print("\nKey Insights:")
    print(f"  - Holiday effect: {((holiday_sales / non_holiday_sales - 1) * 100):.1f}% increase")
    print(f"  - Significant correlations: {significant_features} features")
    print(f"  - Top predictor: {list(impact.items())[0][0]}")

    return sales_with_econ


if __name__ == "__main__":
    # Run the example
    enriched_data = main()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("Check the docs folder for output files:")
    print("  - external_factors_summary.csv")
    print("  - external_factors_analysis.png")
    print("=" * 80)
