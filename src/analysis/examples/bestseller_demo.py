"""
Bestseller Analysis System - Demonstration Script

This script demonstrates all major features of the bestseller analysis system
with realistic synthetic data.

Usage:
    python bestseller_demo.py

Author: PyCaret Development Team
Date: 2025-10-08
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.bestseller_analysis import BestsellerAnalysisSystem


def generate_sample_data():
    """Generate realistic sample sales data"""
    print("Generating sample data...")

    np.random.seed(42)

    # Generate 1 year of data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']
    products = [f'P{i:03d}' for i in range(1, 51)]

    data = []
    for date in dates:
        # Add seasonality
        month = date.month
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)

        for store in stores:
            # Store-specific factors
            store_factor = 1.0 + (hash(store) % 10) / 20

            n_products = np.random.randint(5, 15)
            for product in np.random.choice(products, n_products, replace=False):
                # Product popularity
                product_rank = int(product[1:])
                popularity = 1.0 / (1 + product_rank / 10)

                quantity = int(np.random.poisson(10 * popularity * seasonal_factor * store_factor))
                if quantity == 0:
                    continue

                base_price = 10 + (product_rank % 10) * 10
                price = base_price * (1 + np.random.uniform(-0.1, 0.1))

                data.append({
                    'date': date,
                    'store_id': store,
                    'product_id': product,
                    'product_name': f'Product {product}',
                    'category': f'Category {(product_rank - 1) // 10 + 1}',
                    'quantity': quantity,
                    'price': price,
                    'revenue': quantity * price,
                    'profit': quantity * price * 0.25,
                    'transaction_id': f'T{len(data):06d}'
                })

    df = pd.DataFrame(data)
    print(f"Generated {len(df):,} sales records")
    return df


def generate_store_features():
    """Generate store characteristics"""
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']

    return pd.DataFrame({
        'store_id': stores,
        'store_name': [f'Store {s}' for s in stores],
        'size_sqm': [500, 600, 550, 480, 520],
        'location_type': ['urban', 'suburban', 'urban', 'rural', 'suburban'],
        'customer_segment': ['premium', 'mid', 'premium', 'budget', 'mid'],
        'latitude': [40.7, 40.8, 40.6, 41.0, 40.9],
        'longitude': [-74.0, -74.1, -74.2, -73.9, -74.0]
    })


def demo_top_performers(analyzer, sales_data):
    """Demonstrate top performers identification"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 1: TOP PERFORMERS IDENTIFICATION")
    print("=" * 80)

    top_performers = analyzer.identify_top_performers(
        sales_data,
        store_id='S001',
        top_n=10,
        metric='revenue'
    )

    print(f"\nTop 10 Products in Store S001 (by Revenue):")
    print("-" * 80)

    for i, product in enumerate(top_performers[:10], 1):
        print(f"\n{i}. {product.product_name} ({product.product_id})")
        print(f"   Revenue: ${product.total_revenue:,.2f}")
        print(f"   Quantity: {product.total_quantity:,} units")
        print(f"   Avg Price: ${product.avg_price:.2f}")
        print(f"   Classification: {product.category.value.upper()}")
        print(f"   Velocity: {product.velocity_category.value}")
        print(f"   Growth Rate: {product.growth_rate:+.1%}")
        print(f"   Market Share: {product.market_share_pct:.2f}%")
        print(f"   Daily Sales: ${product.daily_avg_sales:.2f}")


def demo_cross_store_comparison(analyzer, sales_data):
    """Demonstrate cross-store comparison"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 2: CROSS-STORE COMPARISON")
    print("=" * 80)

    # Find a product sold in multiple stores
    product_counts = sales_data.groupby('product_id')['store_id'].nunique()
    product_id = product_counts[product_counts >= 4].index[0]

    comparison = analyzer.compare_across_stores(
        sales_data,
        product_id,
        min_stores=3
    )

    print(f"\nCross-Store Analysis for {comparison.product_name} ({comparison.product_id}):")
    print("-" * 80)
    print(f"Selling in {len(comparison.stores_selling)} stores: {', '.join(comparison.stores_selling)}")
    print(f"Average Rank: {comparison.avg_rank:.1f}")
    print(f"Best Performing Store: {comparison.best_performing_store}")
    print(f"Worst Performing Store: {comparison.worst_performing_store}")
    print(f"Performance Consistency: {comparison.performance_consistency:.1%}")
    print(f"Rank Variance: {comparison.rank_variance:.2f}")

    print(f"\nHigh Potential Stores: {', '.join(comparison.high_potential_stores) if comparison.high_potential_stores else 'None'}")
    print(f"Underutilized Stores: {', '.join(comparison.underutilized_stores) if comparison.underutilized_stores else 'None'}")

    if comparison.statistical_tests:
        print(f"\nStatistical Analysis:")
        for test_name, results in comparison.statistical_tests.items():
            print(f"  {test_name.upper()}:")
            print(f"    p-value: {results['p_value']:.4f}")
            print(f"    Significant: {'Yes' if results['significant'] else 'No'}")


def demo_what_if_analysis(analyzer, sales_data, store_features):
    """Demonstrate what-if analysis"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 3: WHAT-IF ANALYSIS")
    print("=" * 80)

    scenario = analyzer.what_if_analysis(
        sales_data,
        target_store='S001',
        store_features=store_features,
        top_n_similar=3,
        min_similarity=0.5
    )

    print(f"\nWhat-If Analysis for {scenario.target_store}:")
    print("-" * 80)

    print(f"\nSimilar Stores Found:")
    for store_id, similarity in scenario.similar_stores:
        print(f"  {store_id}: {similarity:.1%} similarity")

    print(f"\nSimilarity Metrics:")
    print(f"  Weather Similarity: {scenario.weather_similarity:.1%}")
    print(f"  Size Similarity: {scenario.size_similarity:.1%}")
    print(f"  Demographic Similarity: {scenario.demographic_similarity:.1%}")

    print(f"\nExpected Impact:")
    print(f"  Revenue Lift: ${scenario.expected_revenue_lift:,.2f}")
    print(f"  Sales Increase: {scenario.expected_sales_increase:,} units")
    print(f"  Revenue Lift %: {scenario.expected_impact['revenue_lift_pct']:.1f}%")

    print(f"\nTop 5 Predicted Bestsellers:")
    for i, product in enumerate(scenario.predicted_bestsellers[:5], 1):
        print(f"  {i}. {product['product_id']}")
        print(f"     Expected Revenue: ${product['expected_revenue']:,.2f}")
        print(f"     Confidence: {product['confidence']:.1%}")
        print(f"     In Store: {'Yes' if product['already_in_store'] else 'No'}")

    print(f"\nRecommendations:")
    print(f"  New Products to Add: {len(scenario.products_to_add)}")
    print(f"    {', '.join(scenario.products_to_add[:5])}")
    print(f"  Products to Promote: {len(scenario.products_to_promote)}")
    print(f"    {', '.join(scenario.products_to_promote[:5])}")


def demo_product_recommendations(analyzer, sales_data, store_features):
    """Demonstrate product recommendations"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 4: PRODUCT RECOMMENDATIONS")
    print("=" * 80)

    recommendations = analyzer.recommend_products(
        sales_data,
        store_id='S005',  # Use a different store
        store_features=store_features,
        top_n=5,
        include_success_stories=True
    )

    print(f"\nProduct Recommendations for {recommendations.store_id}:")
    print("-" * 80)

    print(f"\nCurrent Performance:")
    perf = recommendations.current_performance
    print(f"  Total Revenue: ${perf['total_revenue']:,.2f}")
    print(f"  Product Count: {perf['total_products']}")
    print(f"  Avg Daily Revenue: ${perf['avg_daily_revenue']:,.2f}")

    print(f"\nRecommended Products:")
    for i, product in enumerate(recommendations.recommended_products, 1):
        print(f"\n  {i}. {product['product_id']}")
        print(f"     Expected Monthly Revenue: ${product['expected_monthly_revenue']:,.2f}")
        print(f"     Expected Monthly Quantity: {product['expected_monthly_quantity']} units")
        print(f"     Confidence: {product['confidence_score']:.1%}")
        print(f"     Selling in {product['selling_in_similar_stores']} similar stores")

    print(f"\nExpected Impact:")
    print(f"  Monthly Revenue Potential: ${recommendations.revenue_potential:,.2f}")
    print(f"  Monthly Profit Potential: ${recommendations.profit_potential:,.2f}")
    print(f"  Implementation Priority: {recommendations.implementation_priority.upper()}")

    print(f"\nSuccess Stories from Similar Stores:")
    for story in recommendations.success_stories[:2]:
        print(f"  {story['store_id']} ({story['similarity_score']:.1%} similar)")
        print(f"    Total Revenue: ${story['revenue_achieved']:,.2f}")
        print(f"    Top Products: {', '.join(story['top_products'][:3])}")


def demo_sales_velocity(analyzer, sales_data):
    """Demonstrate sales velocity calculation"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 5: SALES VELOCITY ANALYSIS")
    print("=" * 80)

    velocity_data = analyzer.calculate_sales_velocity(
        sales_data,
        store_id='S001',
        window_days=30
    )

    # Show top velocity products
    top_velocity = velocity_data.nlargest(5, 'velocity_30d')

    print(f"\nTop 5 Products by 30-Day Velocity (Store S001):")
    print("-" * 80)

    for i, (_, row) in enumerate(top_velocity.iterrows(), 1):
        print(f"\n{i}. {row['product_id']}")
        print(f"   7-Day Velocity: ${row['velocity_7d']:,.2f}/day")
        print(f"   30-Day Velocity: ${row['velocity_30d']:,.2f}/day")
        print(f"   Growth Rate: {row['growth_rate']:+.1%}")
        print(f"   Acceleration: {row['acceleration']:+.1%}")
        print(f"   Trend Slope: {row['trend_slope']:+.2f}")
        print(f"   Category: {row['velocity_category'].value}")


def demo_seasonal_trends(analyzer, sales_data):
    """Demonstrate seasonal trend analysis"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 6: SEASONAL TREND ANALYSIS")
    print("=" * 80)

    # Analyze overall trends
    seasonal = analyzer.analyze_seasonal_trends(sales_data)

    print(f"\nSeasonal Analysis (All Products):")
    print("-" * 80)
    print(f"Seasonal Pattern: {seasonal['seasonal_pattern'].value}")
    print(f"Seasonal Strength: {seasonal['seasonal_strength']:.2f}")
    print(f"Peak Month: {seasonal['peak_month']}")
    print(f"Low Month: {seasonal['low_month']}")
    print(f"Peak Multiplier: {seasonal['peak_multiplier']:.2f}x average")

    print(f"\nMonthly Revenue Pattern:")
    for month, stats in list(seasonal['monthly_patterns'].items())[:6]:
        print(f"  Month {month}: ${stats['mean']:,.2f} avg (CV: {stats['cv']:.2f})")

    print(f"\nWeekend Effect:")
    weekend = seasonal['weekend_effect']
    print(f"  Weekend Avg: ${weekend['weekend_avg']:,.2f}")
    print(f"  Weekday Avg: ${weekend['weekday_avg']:,.2f}")
    print(f"  Weekend Premium: {weekend['weekend_premium_pct']:+.1f}%")

    print(f"\nGrowth Metrics:")
    print(f"  Year-over-Year Growth: {seasonal['yoy_growth_rate']:+.1%}")
    print(f"  Data Span: {seasonal['data_span_years']:.1f} years")


def demo_market_basket(analyzer, sales_data):
    """Demonstrate market basket analysis"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 7: MARKET BASKET ANALYSIS")
    print("=" * 80)

    try:
        # Prepare transaction data
        transaction_data = sales_data[['transaction_id', 'product_id', 'quantity']].copy()

        basket = analyzer.market_basket_analysis(
            transaction_data,
            min_support=0.005,  # Lower threshold for demo
            min_confidence=0.1,
            min_lift=1.0
        )

        print(f"\nMarket Basket Insights:")
        print("-" * 80)
        print(f"Average Basket Size: {basket.avg_basket_size:.1f} items")

        print(f"\nTop 10 Product Associations (by Lift):")
        for i, (prod_a, prod_b, lift) in enumerate(basket.product_pairs[:10], 1):
            print(f"  {i}. {prod_a} → {prod_b}: {lift:.2f}x lift")

        if basket.recommended_bundles:
            print(f"\nRecommended Product Bundles:")
            for i, bundle in enumerate(basket.recommended_bundles[:5], 1):
                print(f"  {i}. {bundle['products']}")
                print(f"     Support: {bundle['support']:.2%}")
                print(f"     Type: {bundle['bundle_type']}")

        print(f"\nBasket Value Distribution:")
        for metric, value in basket.basket_value_distribution.items():
            print(f"  {metric.capitalize()}: {value:.1f} items")

    except ImportError:
        print("\nMarket basket analysis requires mlxtend package.")
        print("Install with: pip install mlxtend>=0.19.0")


def demo_ml_prediction(analyzer, sales_data):
    """Demonstrate ML model training and prediction"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 8: MACHINE LEARNING PREDICTION")
    print("=" * 80)

    # Prepare training data
    features = sales_data.groupby(['product_id', 'store_id']).agg({
        'revenue': 'sum',
        'quantity': 'sum',
        'date': ['count', 'min', 'max']
    }).reset_index()

    features.columns = ['product_id', 'store_id', 'revenue', 'quantity',
                       'num_transactions', 'first_date', 'last_date']

    features['days_in_stock'] = (
        pd.to_datetime(features['last_date']) -
        pd.to_datetime(features['first_date'])
    ).dt.days + 1

    feature_columns = ['quantity', 'num_transactions', 'days_in_stock']

    print("\nTraining Bestseller Prediction Model...")
    results = analyzer.train_bestseller_predictor(
        features,
        feature_columns=feature_columns,
        test_size=0.2
    )

    print(f"\nModel Performance:")
    print("-" * 80)
    print(f"Model Type: {results['model_type']}")
    print(f"Train Accuracy: {results['train_accuracy']:.2%}")
    print(f"Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"CV Mean Accuracy: {results['cv_mean_accuracy']:.2%} ± {results['cv_std_accuracy']:.2%}")

    print(f"\nFeature Importance:")
    for feature, importance in list(results['feature_importance'].items()):
        print(f"  {feature}: {importance:.3f}")

    print(f"\nClass Distribution:")
    print(f"  Training:")
    for class_label, count in results['class_distribution']['train'].items():
        print(f"    Class {class_label}: {count} samples")

    # Predict on sample products
    sample_features = features[feature_columns].head(10)
    predictions = analyzer.predict_bestseller_probability(sample_features)

    print(f"\nSample Predictions (First 10 Products):")
    print("-" * 80)
    for i, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        print(f"{i}. Bestseller Probability: {row['bestseller_probability']:.1%}")
        print(f"   Predicted: {'Yes' if row['is_bestseller_predicted'] else 'No'}")
        print(f"   Confidence: {row['confidence_level']}")


def main():
    """Main demonstration runner"""
    print("=" * 80)
    print("BESTSELLER ANALYSIS SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)

    # Generate data
    sales_data = generate_sample_data()
    store_features = generate_store_features()

    # Initialize analyzer
    analyzer = BestsellerAnalysisSystem(random_state=42)

    # Run demonstrations
    demo_top_performers(analyzer, sales_data)
    demo_cross_store_comparison(analyzer, sales_data)
    demo_what_if_analysis(analyzer, sales_data, store_features)
    demo_product_recommendations(analyzer, sales_data, store_features)
    demo_sales_velocity(analyzer, sales_data)
    demo_seasonal_trends(analyzer, sales_data)
    demo_market_basket(analyzer, sales_data)
    demo_ml_prediction(analyzer, sales_data)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll features demonstrated successfully!")
    print("See docs/bestseller_analysis_guide.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
