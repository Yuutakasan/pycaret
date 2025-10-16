# Bestseller Analysis System - User Guide

## Overview

The Bestseller Analysis System is a comprehensive machine learning-powered solution for identifying, analyzing, and predicting bestselling products across retail stores. It provides actionable insights for inventory optimization, product recommendations, and sales strategy.

## Features

### 1. Top Performers Identification
- **Multi-dimensional ranking** by revenue, quantity, profit, or velocity
- **Category-based analysis** for product segments
- **Store-level granularity** for location-specific insights
- **Automated classification** into performance tiers (Superstar, Bestseller, Performer, Average, Underperformer)

### 2. Cross-Store Comparison
- **Performance benchmarking** across locations
- **Statistical significance testing** for differences
- **Consistency scoring** to identify reliable performers
- **Potential identification** for underutilized markets

### 3. What-If Analysis
- **Predictive modeling** for product performance in new stores
- **Similarity matching** based on store characteristics, weather, and demographics
- **Confidence scoring** for prediction reliability
- **Impact estimation** for revenue and sales lift

### 4. Product Recommendations
- **Intelligent suggestions** for underperforming stores
- **Evidence-based recommendations** from similar successful stores
- **Priority scoring** (High/Medium/Low) for implementation
- **Risk assessment** for new product introductions

### 5. Sales Velocity Tracking
- **Real-time velocity calculation** with 7-day and 30-day windows
- **Growth rate analysis** and trend detection
- **Acceleration metrics** for momentum tracking
- **Category assignment** (Explosive, Fast, Moderate, Slow, Declining)

### 6. Market Basket Analysis
- **Association rule mining** for product relationships
- **Bundle recommendations** based on co-purchase patterns
- **Cross-sell opportunities** identification
- **Lift calculation** for product pairs

### 7. Seasonal Trend Detection
- **Pattern recognition** for seasonal peaks
- **Monthly and quarterly analysis**
- **Weekend vs weekday effects**
- **Year-over-year growth tracking**

### 8. Machine Learning Predictions
- **RandomForest, XGBoost, or GradientBoosting** models
- **Feature importance analysis**
- **Cross-validation** for model reliability
- **Probability scoring** for bestseller likelihood

## Installation

```bash
# Install PyCaret with full dependencies
pip install pycaret[full]

# Or install specific dependencies
pip install pycaret xgboost>=2.0.0 mlxtend>=0.19.0
```

## Quick Start

```python
from analysis.bestseller_analysis import BestsellerAnalysisSystem
import pandas as pd

# Initialize the system
analyzer = BestsellerAnalysisSystem()

# Load your sales data
sales_data = pd.read_csv('sales.csv')
# Required columns: product_id, store_id, date, quantity, revenue

# Identify top performers
top_products = analyzer.identify_top_performers(
    sales_data,
    store_id='S001',
    top_n=20,
    metric='revenue'
)

# Print results
for product in top_products[:5]:
    print(f"{product.product_name}: ${product.total_revenue:,.2f}")
    print(f"  Category: {product.category.value}")
    print(f"  Velocity: {product.velocity_category.value}")
    print(f"  Growth: {product.growth_rate:.1%}")
```

## Detailed Usage Examples

### Example 1: Identify Top Performers by Store

```python
# Get top 10 products by revenue for Store S001
top_performers = analyzer.identify_top_performers(
    sales_data,
    store_id='S001',
    top_n=10,
    metric='revenue'
)

# Access detailed metrics
for product in top_performers:
    print(f"Product: {product.product_name}")
    print(f"  Total Revenue: ${product.total_revenue:,.2f}")
    print(f"  Daily Avg Sales: ${product.daily_avg_sales:,.2f}")
    print(f"  Market Share: {product.market_share_pct:.2f}%")
    print(f"  Sales Rank: {product.sales_rank}")
    print(f"  Category: {product.category.value}")
    print(f"  Velocity: {product.velocity_category.value}")
    print(f"  Seasonal Pattern: {product.seasonal_pattern}")
    print()
```

### Example 2: Cross-Store Performance Analysis

```python
# Compare product P001 performance across all stores
comparison = analyzer.compare_across_stores(
    sales_data,
    product_id='P001',
    min_stores=3
)

print(f"Product: {comparison.product_name}")
print(f"Selling in {len(comparison.stores_selling)} stores")
print(f"Average Rank: {comparison.avg_rank:.1f}")
print(f"Best Store: {comparison.best_performing_store}")
print(f"Worst Store: {comparison.worst_performing_store}")
print(f"Performance Consistency: {comparison.performance_consistency:.2%}")
print(f"\nHigh Potential Stores: {comparison.high_potential_stores}")
print(f"Underutilized Stores: {comparison.underutilized_stores}")

# Check statistical significance
if 'anova' in comparison.statistical_tests:
    anova = comparison.statistical_tests['anova']
    print(f"\nANOVA p-value: {anova['p_value']:.4f}")
    print(f"Significant difference: {anova['significant']}")
```

### Example 3: What-If Analysis for New Product Launch

```python
# Prepare store features
store_features = pd.DataFrame({
    'store_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'size_sqm': [500, 600, 550, 480, 520],
    'location_type': ['urban', 'suburban', 'urban', 'rural', 'suburban'],
    'customer_segment': ['premium', 'mid', 'premium', 'budget', 'mid']
})

# Run what-if analysis for Store S001
scenario = analyzer.what_if_analysis(
    sales_data,
    target_store='S001',
    store_features=store_features,
    top_n_similar=5,
    min_similarity=0.7
)

print(f"Target Store: {scenario.target_store}")
print(f"\nSimilar Stores:")
for store_id, similarity in scenario.similar_stores:
    print(f"  {store_id}: {similarity:.1%} similar")

print(f"\nExpected Revenue Lift: ${scenario.expected_revenue_lift:,.2f}")
print(f"Expected Sales Increase: {scenario.expected_sales_increase} units")

print(f"\nTop 5 Predicted Bestsellers:")
for product in scenario.predicted_bestsellers[:5]:
    print(f"  {product['product_id']}: ${product['expected_revenue']:,.2f}")
    print(f"    Confidence: {product['confidence']:.1%}")
    print(f"    Already in store: {product['already_in_store']}")

print(f"\nProducts to Add: {scenario.products_to_add[:5]}")
print(f"Products to Promote: {scenario.products_to_promote[:5]}")
```

### Example 4: Product Recommendations for Underperforming Store

```python
# Get recommendations for Store S001
recommendations = analyzer.recommend_products(
    sales_data,
    store_id='S001',
    store_features=store_features,
    top_n=10,
    include_success_stories=True
)

print(f"Store: {recommendations.store_id}")
print(f"\nCurrent Performance:")
print(f"  Total Revenue: ${recommendations.current_performance['total_revenue']:,.2f}")
print(f"  Products Sold: {recommendations.current_performance['total_products']}")
print(f"  Avg Daily Revenue: ${recommendations.current_performance['avg_daily_revenue']:,.2f}")

print(f"\nRecommended Products:")
for product in recommendations.recommended_products:
    print(f"  {product['product_id']}")
    print(f"    Expected Monthly Revenue: ${product['expected_monthly_revenue']:,.2f}")
    print(f"    Expected Monthly Quantity: {product['expected_monthly_quantity']}")
    print(f"    Confidence: {product['confidence_score']:.1%}")

print(f"\nRevenue Potential: ${recommendations.revenue_potential:,.2f}/month")
print(f"Profit Potential: ${recommendations.profit_potential:,.2f}/month")
print(f"Priority: {recommendations.implementation_priority}")

# Success stories from similar stores
print(f"\nSuccess Stories:")
for story in recommendations.success_stories:
    print(f"  Store {story['store_id']} ({story['similarity_score']:.1%} similar)")
    print(f"    Revenue: ${story['revenue_achieved']:,.2f}")
    print(f"    Top Products: {story['top_products'][:3]}")
```

### Example 5: Sales Velocity Analysis

```python
# Calculate velocity for all products in Store S001
velocity_data = analyzer.calculate_sales_velocity(
    sales_data,
    store_id='S001',
    window_days=30
)

# Show top velocity products
top_velocity = velocity_data.nlargest(10, 'velocity_30d')

print("Top 10 Products by 30-Day Velocity:")
for _, row in top_velocity.iterrows():
    print(f"{row['product_id']}:")
    print(f"  7-day velocity: ${row['velocity_7d']:,.2f}/day")
    print(f"  30-day velocity: ${row['velocity_30d']:,.2f}/day")
    print(f"  Growth rate: {row['growth_rate']:.1%}")
    print(f"  Trend: {row['trend_slope']:.2f} (R²={row['trend_r2']:.2f})")
    print(f"  Category: {row['velocity_category'].value}")
    print()
```

### Example 6: Market Basket Analysis

```python
# Prepare transaction data
# Required columns: transaction_id, product_id, quantity
transaction_data = pd.read_csv('transactions.csv')

# Perform market basket analysis
basket_insights = analyzer.market_basket_analysis(
    transaction_data,
    min_support=0.01,
    min_confidence=0.1,
    min_lift=1.0
)

print(f"Average Basket Size: {basket_insights.avg_basket_size:.1f} items")

print("\nTop Product Associations (by Lift):")
for prod_a, prod_b, lift in basket_insights.product_pairs[:10]:
    print(f"  {prod_a} → {prod_b}: {lift:.2f}x lift")

print("\nRecommended Product Bundles:")
for bundle in basket_insights.recommended_bundles[:5]:
    print(f"  {bundle['products']}")
    print(f"    Support: {bundle['support']:.1%}")
    print(f"    Type: {bundle['bundle_type']}")

print("\nCross-Sell Opportunities:")
for product, opportunities in list(basket_insights.cross_sell_opportunities.items())[:3]:
    print(f"  When customer buys {product}, suggest:")
    for opp in opportunities[:3]:
        print(f"    - {opp['product']} (confidence: {opp['confidence']:.1%}, lift: {opp['lift']:.2f})")
```

### Example 7: Seasonal Trend Analysis

```python
# Analyze seasonal patterns for Product P001
seasonal_analysis = analyzer.analyze_seasonal_trends(
    sales_data,
    product_id='P001'
)

print(f"Seasonal Pattern: {seasonal_analysis['seasonal_pattern'].value}")
print(f"Seasonal Strength: {seasonal_analysis['seasonal_strength']:.2f}")
print(f"Peak Month: {seasonal_analysis['peak_month']}")
print(f"Low Month: {seasonal_analysis['low_month']}")
print(f"Peak Multiplier: {seasonal_analysis['peak_multiplier']:.2f}x average")

print("\nMonthly Patterns:")
for month, stats in seasonal_analysis['monthly_patterns'].items():
    print(f"  Month {month}: ${stats['mean']:,.2f} avg (CV: {stats['cv']:.2f})")

print("\nWeekend Effect:")
weekend = seasonal_analysis['weekend_effect']
print(f"  Weekend Avg: ${weekend['weekend_avg']:,.2f}")
print(f"  Weekday Avg: ${weekend['weekday_avg']:,.2f}")
print(f"  Weekend Premium: {weekend['weekend_premium_pct']:.1f}%")

print(f"\nYear-over-Year Growth: {seasonal_analysis['yoy_growth_rate']:.1%}")
print(f"Data Span: {seasonal_analysis['data_span_years']:.1f} years")
print(f"Sufficient History: {seasonal_analysis['sufficient_history']}")
```

### Example 8: Machine Learning Bestseller Prediction

```python
# Prepare training data with features
training_data = sales_data.groupby(['product_id', 'store_id']).agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'date': ['count', 'min', 'max']
}).reset_index()

# Flatten columns
training_data.columns = ['product_id', 'store_id', 'revenue', 'quantity',
                         'num_transactions', 'first_date', 'last_date']

# Add features
training_data['days_in_stock'] = (
    pd.to_datetime(training_data['last_date']) -
    pd.to_datetime(training_data['first_date'])
).dt.days

# Define feature columns
feature_columns = ['quantity', 'num_transactions', 'days_in_stock']

# Train model
training_results = analyzer.train_bestseller_predictor(
    training_data,
    feature_columns=feature_columns,
    test_size=0.2
)

print("Model Training Results:")
print(f"  Model Type: {training_results['model_type']}")
print(f"  Train Accuracy: {training_results['train_accuracy']:.2%}")
print(f"  Test Accuracy: {training_results['test_accuracy']:.2%}")
print(f"  CV Mean Accuracy: {training_results['cv_mean_accuracy']:.2%} ± {training_results['cv_std_accuracy']:.2%}")

print("\nFeature Importance:")
for feature, importance in list(training_results['feature_importance'].items())[:5]:
    print(f"  {feature}: {importance:.3f}")

# Predict on new products
new_products = pd.DataFrame({
    'quantity': [100, 50, 200],
    'num_transactions': [20, 15, 40],
    'days_in_stock': [30, 30, 30]
})

predictions = analyzer.predict_bestseller_probability(new_products)

print("\nPredictions:")
for idx, row in predictions.iterrows():
    print(f"Product {idx}:")
    print(f"  Bestseller Probability: {row['bestseller_probability']:.1%}")
    print(f"  Predicted: {'Yes' if row['is_bestseller_predicted'] else 'No'}")
    print(f"  Confidence: {row['confidence_level']}")
```

## Configuration Options

### Bestseller Thresholds
```python
analyzer = BestsellerAnalysisSystem(
    bestseller_thresholds={
        'superstar': 0.05,    # Top 5%
        'bestseller': 0.20,   # Top 20%
        'performer': 0.50,    # Top 50%
        'average': 0.80       # Top 80%
    }
)
```

### Velocity Thresholds
```python
analyzer = BestsellerAnalysisSystem(
    velocity_thresholds={
        'explosive': 2.0,  # 200% growth
        'fast': 0.5,       # 50% growth
        'moderate': 0.1,   # 10% growth
        'slow': 0.0        # 0% growth
    }
)
```

### ML Model Selection
```python
# Use RandomForest (default)
analyzer = BestsellerAnalysisSystem(ml_model_type='random_forest')

# Use XGBoost (requires xgboost package)
analyzer = BestsellerAnalysisSystem(ml_model_type='xgboost', use_xgboost=True)

# Use Gradient Boosting
analyzer = BestsellerAnalysisSystem(ml_model_type='gradient_boosting')
```

## Data Requirements

### Minimum Required Columns

**Sales Data:**
- `product_id` (str): Unique product identifier
- `store_id` (str): Unique store identifier
- `date` (datetime): Transaction date
- `quantity` (int): Units sold
- `revenue` (float): Total revenue

**Optional Columns (for enhanced analysis):**
- `product_name` (str): Product name
- `category` (str): Product category
- `price` (float): Unit price
- `profit` (float): Total profit
- `cost` (float): Cost per unit

**Store Features (for what-if analysis):**
- `store_id` (str): Store identifier
- `size_sqm` (float): Store size
- `location_type` (str): Urban/suburban/rural
- `customer_segment` (str): Target segment
- `latitude` (float): Geographic latitude
- `longitude` (float): Geographic longitude

**Transaction Data (for market basket analysis):**
- `transaction_id` (str): Unique transaction ID
- `product_id` (str): Product identifier
- `quantity` (int): Quantity purchased

## Performance Tips

1. **Filter Data**: Apply store/category filters before analysis to improve performance
2. **Batch Processing**: Process multiple stores in parallel for large datasets
3. **Cache Results**: Use the built-in cache for repeated analyses
4. **Sampling**: Use data sampling for exploratory analysis on large datasets
5. **Feature Selection**: Limit features to most important ones for ML training

## Troubleshooting

### Common Issues

**Issue: No bestsellers found**
- Solution: Lower the `bestseller_thresholds` or increase sample size

**Issue: Market basket analysis returns empty results**
- Solution: Lower `min_support` threshold (try 0.005 or 0.001)

**Issue: What-if analysis finds no similar stores**
- Solution: Lower `min_similarity` threshold to 0.5 or below

**Issue: ML model accuracy is low**
- Solution: Add more features, increase training data, or try different model types

## API Reference

See inline documentation for detailed API reference:
```python
help(BestsellerAnalysisSystem)
help(BestsellerAnalysisSystem.identify_top_performers)
# etc.
```

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub: https://github.com/pycaret/pycaret
- Documentation: https://pycaret.gitbook.io/

---

**Last Updated**: 2025-10-08
**Version**: 1.0.0
