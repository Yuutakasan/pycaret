"""
Unit Tests for Bestseller Analysis System

This test suite validates all functionality of the bestseller analysis module including:
- Top performers identification
- Cross-store comparison
- What-if analysis
- Product recommendations
- Sales velocity calculation
- Market basket analysis
- Seasonal trend detection
- ML model training and prediction

Author: PyCaret Development Team
Date: 2025-10-08
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.bestseller_analysis import (
    BestsellerAnalysisSystem,
    BestsellerCategory,
    VelocityCategory,
    SeasonalPattern,
    BestsellerMetrics,
    CrossStoreComparison,
    WhatIfScenario,
    ProductRecommendation,
    MarketBasketInsights
)


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing"""
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']
    products = [f'P{i:03d}' for i in range(1, 51)]

    data = []
    for date in dates:
        for store in stores:
            n_products = np.random.randint(5, 15)
            for product in np.random.choice(products, n_products, replace=False):
                quantity = np.random.randint(1, 20)
                price = np.random.uniform(10, 100)
                data.append({
                    'date': date,
                    'store_id': store,
                    'product_id': product,
                    'product_name': f'Product {product}',
                    'category': f'Category {int(product[1:]) % 5 + 1}',
                    'quantity': quantity,
                    'price': price,
                    'revenue': quantity * price,
                    'profit': quantity * price * 0.25
                })

    return pd.DataFrame(data)


@pytest.fixture
def sample_store_features():
    """Generate sample store features for testing"""
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']

    return pd.DataFrame({
        'store_id': stores,
        'size_sqm': [500, 600, 550, 480, 520],
        'location_type': ['urban', 'suburban', 'urban', 'rural', 'suburban'],
        'customer_segment': ['premium', 'mid', 'premium', 'budget', 'mid'],
        'latitude': [40.7, 40.8, 40.6, 41.0, 40.9],
        'longitude': [-74.0, -74.1, -74.2, -73.9, -74.0]
    })


@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data for market basket analysis"""
    np.random.seed(42)

    transactions = []
    for i in range(1000):
        transaction_id = f'T{i:04d}'
        n_items = np.random.randint(1, 6)
        products = np.random.choice(range(1, 21), n_items, replace=False)

        for product in products:
            transactions.append({
                'transaction_id': transaction_id,
                'product_id': f'P{product:03d}',
                'quantity': np.random.randint(1, 5)
            })

    return pd.DataFrame(transactions)


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return BestsellerAnalysisSystem(random_state=42)


class TestBestsellerIdentification:
    """Test suite for bestseller identification"""

    def test_identify_top_performers_basic(self, analyzer, sample_sales_data):
        """Test basic top performers identification"""
        results = analyzer.identify_top_performers(
            sample_sales_data,
            store_id='S001',
            top_n=10,
            metric='revenue'
        )

        assert len(results) <= 10
        assert all(isinstance(r, BestsellerMetrics) for r in results)
        assert all(r.store_id == 'S001' for r in results)

        # Check ordering by revenue
        revenues = [r.total_revenue for r in results]
        assert revenues == sorted(revenues, reverse=True)

    def test_identify_top_performers_by_category(self, analyzer, sample_sales_data):
        """Test top performers by category"""
        results = analyzer.identify_top_performers(
            sample_sales_data,
            category='Category 1',
            top_n=5
        )

        assert len(results) <= 5
        assert all(r.category == 'Category 1' for r in results)

    def test_identify_top_performers_metrics(self, analyzer, sample_sales_data):
        """Test different ranking metrics"""
        for metric in ['revenue', 'quantity', 'profit', 'velocity']:
            results = analyzer.identify_top_performers(
                sample_sales_data,
                top_n=5,
                metric=metric
            )
            assert len(results) > 0

    def test_bestseller_categories(self, analyzer, sample_sales_data):
        """Test bestseller categorization"""
        results = analyzer.identify_top_performers(
            sample_sales_data,
            top_n=20
        )

        categories = [r.category for r in results]
        assert BestsellerCategory.SUPERSTAR in categories or BestsellerCategory.BESTSELLER in categories

    def test_velocity_categories(self, analyzer, sample_sales_data):
        """Test velocity categorization"""
        results = analyzer.identify_top_performers(
            sample_sales_data,
            top_n=10
        )

        velocities = [r.velocity_category for r in results]
        assert len(velocities) > 0
        assert all(isinstance(v, VelocityCategory) for v in velocities)


class TestCrossStoreComparison:
    """Test suite for cross-store comparison"""

    def test_compare_across_stores_basic(self, analyzer, sample_sales_data):
        """Test basic cross-store comparison"""
        # Find a product that exists in multiple stores
        product_counts = sample_sales_data.groupby('product_id')['store_id'].nunique()
        product_id = product_counts[product_counts >= 3].index[0]

        result = analyzer.compare_across_stores(
            sample_sales_data,
            product_id,
            min_stores=3
        )

        assert isinstance(result, CrossStoreComparison)
        assert result.product_id == product_id
        assert len(result.stores_selling) >= 3
        assert result.best_performing_store in result.stores_selling
        assert result.worst_performing_store in result.stores_selling

    def test_compare_performance_metrics(self, analyzer, sample_sales_data):
        """Test performance metrics in comparison"""
        product_counts = sample_sales_data.groupby('product_id')['store_id'].nunique()
        product_id = product_counts[product_counts >= 3].index[0]

        result = analyzer.compare_across_stores(
            sample_sales_data,
            product_id
        )

        assert result.avg_rank > 0
        assert result.rank_variance >= 0
        assert result.revenue_variance >= 0
        assert 0 <= result.performance_consistency <= 1

    def test_high_potential_stores(self, analyzer, sample_sales_data):
        """Test identification of high potential stores"""
        product_counts = sample_sales_data.groupby('product_id')['store_id'].nunique()
        product_id = product_counts[product_counts >= 3].index[0]

        result = analyzer.compare_across_stores(
            sample_sales_data,
            product_id
        )

        assert isinstance(result.high_potential_stores, list)
        assert isinstance(result.underutilized_stores, list)


class TestWhatIfAnalysis:
    """Test suite for what-if analysis"""

    def test_what_if_basic(self, analyzer, sample_sales_data, sample_store_features):
        """Test basic what-if analysis"""
        result = analyzer.what_if_analysis(
            sample_sales_data,
            target_store='S001',
            store_features=sample_store_features,
            top_n_similar=3
        )

        assert isinstance(result, WhatIfScenario)
        assert result.target_store == 'S001'
        assert len(result.similar_stores) <= 3
        assert len(result.predicted_bestsellers) > 0

    def test_what_if_similarity_scores(self, analyzer, sample_sales_data, sample_store_features):
        """Test similarity scores in what-if analysis"""
        result = analyzer.what_if_analysis(
            sample_sales_data,
            target_store='S001',
            store_features=sample_store_features,
            min_similarity=0.5
        )

        # Check similarity scores
        for store_id, similarity in result.similar_stores:
            assert 0 <= similarity <= 1
            assert store_id != 'S001'

    def test_what_if_predictions(self, analyzer, sample_sales_data, sample_store_features):
        """Test predicted bestsellers"""
        result = analyzer.what_if_analysis(
            sample_sales_data,
            target_store='S001',
            store_features=sample_store_features
        )

        assert len(result.predicted_bestsellers) > 0

        for pred in result.predicted_bestsellers:
            assert 'product_id' in pred
            assert 'expected_revenue' in pred
            assert 'confidence' in pred
            assert 0 <= pred['confidence'] <= 1

    def test_what_if_expected_impact(self, analyzer, sample_sales_data, sample_store_features):
        """Test expected impact calculations"""
        result = analyzer.what_if_analysis(
            sample_sales_data,
            target_store='S001',
            store_features=sample_store_features
        )

        assert result.expected_revenue_lift >= 0
        assert result.expected_sales_increase >= 0
        assert 'revenue_lift_pct' in result.expected_impact


class TestProductRecommendations:
    """Test suite for product recommendations"""

    def test_recommend_products_basic(self, analyzer, sample_sales_data, sample_store_features):
        """Test basic product recommendations"""
        result = analyzer.recommend_products(
            sample_sales_data,
            store_id='S001',
            store_features=sample_store_features,
            top_n=5
        )

        assert isinstance(result, ProductRecommendation)
        assert result.store_id == 'S001'
        assert len(result.recommended_products) <= 5

    def test_recommend_current_performance(self, analyzer, sample_sales_data, sample_store_features):
        """Test current performance assessment"""
        result = analyzer.recommend_products(
            sample_sales_data,
            store_id='S001',
            store_features=sample_store_features
        )

        perf = result.current_performance
        assert 'total_revenue' in perf
        assert 'total_products' in perf
        assert 'avg_daily_revenue' in perf
        assert perf['total_revenue'] > 0

    def test_recommend_success_stories(self, analyzer, sample_sales_data, sample_store_features):
        """Test success stories inclusion"""
        result = analyzer.recommend_products(
            sample_sales_data,
            store_id='S001',
            store_features=sample_store_features,
            include_success_stories=True
        )

        assert len(result.success_stories) > 0
        for story in result.success_stories:
            assert 'store_id' in story
            assert 'similarity_score' in story
            assert 'revenue_achieved' in story

    def test_recommend_priority_levels(self, analyzer, sample_sales_data, sample_store_features):
        """Test priority level assignment"""
        result = analyzer.recommend_products(
            sample_sales_data,
            store_id='S001',
            store_features=sample_store_features
        )

        assert result.implementation_priority in ['high', 'medium', 'low']


class TestSalesVelocity:
    """Test suite for sales velocity calculation"""

    def test_calculate_velocity_basic(self, analyzer, sample_sales_data):
        """Test basic velocity calculation"""
        result = analyzer.calculate_sales_velocity(
            sample_sales_data,
            store_id='S001',
            window_days=30
        )

        assert isinstance(result, pd.DataFrame)
        assert 'velocity_7d' in result.columns
        assert 'velocity_30d' in result.columns
        assert 'growth_rate' in result.columns
        assert 'velocity_category' in result.columns

    def test_calculate_velocity_by_product(self, analyzer, sample_sales_data):
        """Test velocity for specific product"""
        product_id = sample_sales_data['product_id'].iloc[0]

        result = analyzer.calculate_sales_velocity(
            sample_sales_data,
            product_id=product_id,
            window_days=30
        )

        assert len(result) > 0
        assert all(result['product_id'] == product_id)

    def test_velocity_trends(self, analyzer, sample_sales_data):
        """Test trend metrics in velocity"""
        result = analyzer.calculate_sales_velocity(
            sample_sales_data,
            store_id='S001'
        )

        assert 'trend_slope' in result.columns
        assert 'trend_r2' in result.columns
        assert 'acceleration' in result.columns


class TestMarketBasketAnalysis:
    """Test suite for market basket analysis"""

    def test_market_basket_basic(self, analyzer, sample_transaction_data):
        """Test basic market basket analysis"""
        try:
            result = analyzer.market_basket_analysis(
                sample_transaction_data,
                min_support=0.01,
                min_confidence=0.1
            )

            assert isinstance(result, MarketBasketInsights)
            assert isinstance(result.product_pairs, list)
            assert isinstance(result.frequent_itemsets, pd.DataFrame)

        except ImportError:
            pytest.skip("mlxtend not available")

    def test_market_basket_associations(self, analyzer, sample_transaction_data):
        """Test association rules"""
        try:
            result = analyzer.market_basket_analysis(
                sample_transaction_data,
                min_support=0.01
            )

            if len(result.product_pairs) > 0:
                for prod_a, prod_b, lift in result.product_pairs[:5]:
                    assert lift >= 1.0

        except ImportError:
            pytest.skip("mlxtend not available")

    def test_market_basket_bundles(self, analyzer, sample_transaction_data):
        """Test bundle recommendations"""
        try:
            result = analyzer.market_basket_analysis(
                sample_transaction_data,
                min_support=0.01
            )

            assert isinstance(result.recommended_bundles, list)

        except ImportError:
            pytest.skip("mlxtend not available")


class TestSeasonalAnalysis:
    """Test suite for seasonal trend analysis"""

    def test_seasonal_trends_basic(self, analyzer, sample_sales_data):
        """Test basic seasonal analysis"""
        result = analyzer.analyze_seasonal_trends(
            sample_sales_data,
            product_id=sample_sales_data['product_id'].iloc[0]
        )

        assert 'seasonal_pattern' in result
        assert 'seasonal_strength' in result
        assert 'peak_month' in result
        assert 'monthly_patterns' in result

    def test_seasonal_pattern_detection(self, analyzer, sample_sales_data):
        """Test seasonal pattern detection"""
        result = analyzer.analyze_seasonal_trends(
            sample_sales_data
        )

        assert isinstance(result['seasonal_pattern'], SeasonalPattern)
        assert 1 <= result['peak_month'] <= 12
        assert 1 <= result['low_month'] <= 12

    def test_weekend_effects(self, analyzer, sample_sales_data):
        """Test weekend effect analysis"""
        result = analyzer.analyze_seasonal_trends(
            sample_sales_data
        )

        assert 'weekend_effect' in result
        weekend = result['weekend_effect']
        assert 'weekend_avg' in weekend
        assert 'weekday_avg' in weekend


class TestMachineLearning:
    """Test suite for ML model training and prediction"""

    def test_train_predictor_basic(self, analyzer, sample_sales_data):
        """Test basic model training"""
        # Prepare features
        features = sample_sales_data.groupby(['product_id', 'store_id']).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'date': 'count'
        }).reset_index()

        features.columns = ['product_id', 'store_id', 'revenue', 'quantity', 'num_transactions']

        feature_columns = ['quantity', 'num_transactions']

        result = analyzer.train_bestseller_predictor(
            features,
            feature_columns=feature_columns,
            test_size=0.2
        )

        assert 'train_accuracy' in result
        assert 'test_accuracy' in result
        assert 'cv_mean_accuracy' in result
        assert result['train_accuracy'] > 0
        assert result['test_accuracy'] > 0

    def test_predict_probability(self, analyzer, sample_sales_data):
        """Test probability prediction"""
        # First train model
        features = sample_sales_data.groupby(['product_id', 'store_id']).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'date': 'count'
        }).reset_index()

        features.columns = ['product_id', 'store_id', 'revenue', 'quantity', 'num_transactions']

        feature_columns = ['quantity', 'num_transactions']

        analyzer.train_bestseller_predictor(
            features,
            feature_columns=feature_columns
        )

        # Predict on new data
        test_features = features[feature_columns].head(10)
        predictions = analyzer.predict_bestseller_probability(test_features)

        assert 'bestseller_probability' in predictions.columns
        assert 'is_bestseller_predicted' in predictions.columns
        assert all(predictions['bestseller_probability'].between(0, 1))


class TestEdgeCases:
    """Test suite for edge cases and error handling"""

    def test_empty_data(self, analyzer):
        """Test handling of empty data"""
        empty_df = pd.DataFrame(columns=['product_id', 'store_id', 'date', 'quantity', 'revenue'])

        results = analyzer.identify_top_performers(empty_df, top_n=10)
        assert len(results) == 0

    def test_missing_columns(self, analyzer):
        """Test handling of missing columns"""
        incomplete_df = pd.DataFrame({
            'product_id': ['P001'],
            'store_id': ['S001']
        })

        with pytest.raises(ValueError):
            analyzer.identify_top_performers(incomplete_df)

    def test_single_store(self, analyzer, sample_sales_data):
        """Test with single store"""
        single_store = sample_sales_data[sample_sales_data['store_id'] == 'S001']

        results = analyzer.identify_top_performers(single_store, top_n=5)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
