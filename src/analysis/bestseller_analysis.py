"""
Bestseller Analysis System with Machine Learning

This module provides comprehensive bestseller identification, prediction, and recommendation
capabilities including:
- Top performers identification by store and product category
- Cross-store bestseller comparison and benchmarking
- What-if analysis: predict bestsellers for similar stores with matching conditions
- Product recommendation engine for underperforming stores
- Sales velocity calculation and trending analysis
- Market basket analysis for product associations
- Seasonal bestseller trends and pattern recognition
- Machine learning models for bestseller prediction
- Real-time performance monitoring and alerts

Author: PyCaret Development Team
License: MIT
Version: 1.0.0
Date: 2025-10-08
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Optional ML dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost>=2.0.0")

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    warnings.warn("mlxtend not available. Install with: pip install mlxtend>=0.19.0")

warnings.filterwarnings('ignore')


class BestsellerCategory(Enum):
    """Bestseller classification categories"""
    SUPERSTAR = "superstar"  # Top 5%: Exceptional performance
    BESTSELLER = "bestseller"  # Top 5-20%: Strong performance
    PERFORMER = "performer"  # Top 20-50%: Good performance
    AVERAGE = "average"  # 50-80%: Average performance
    UNDERPERFORMER = "underperformer"  # Bottom 20%: Needs attention


class VelocityCategory(Enum):
    """Sales velocity categories"""
    EXPLOSIVE = "explosive"  # >200% growth rate
    FAST = "fast"  # 50-200% growth rate
    MODERATE = "moderate"  # 10-50% growth rate
    SLOW = "slow"  # 0-10% growth rate
    DECLINING = "declining"  # <0% growth rate


class SeasonalPattern(Enum):
    """Seasonal pattern types"""
    SUMMER_PEAK = "summer_peak"
    WINTER_PEAK = "winter_peak"
    SPRING_PEAK = "spring_peak"
    FALL_PEAK = "fall_peak"
    HOLIDAY_DRIVEN = "holiday_driven"
    YEAR_ROUND = "year_round"
    WEEKEND_DRIVEN = "weekend_driven"
    WEEKDAY_DRIVEN = "weekday_driven"


@dataclass
class BestsellerMetrics:
    """Comprehensive metrics for a bestseller product"""
    product_id: str
    product_name: str
    category: str
    store_id: str

    # Sales metrics
    total_revenue: float
    total_quantity: int
    avg_price: float
    profit_margin: float

    # Performance metrics
    sales_rank: int
    revenue_contribution_pct: float
    category: BestsellerCategory

    # Velocity metrics
    daily_avg_sales: float
    weekly_avg_sales: float
    monthly_avg_sales: float
    growth_rate: float
    velocity_category: VelocityCategory

    # Temporal metrics
    days_in_stock: int
    stockout_days: int
    availability_rate: float

    # Comparative metrics
    market_share_pct: float
    category_rank: int
    store_rank: int

    # Seasonality
    seasonal_pattern: Optional[SeasonalPattern] = None
    peak_season_multiplier: float = 1.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossStoreComparison:
    """Cross-store bestseller comparison results"""
    product_id: str
    product_name: str

    # Performance across stores
    stores_selling: List[str]
    avg_rank: float
    best_performing_store: str
    worst_performing_store: str

    # Variance analysis
    rank_variance: float
    revenue_variance: float
    performance_consistency: float  # 0-1 score

    # Recommendations
    high_potential_stores: List[str]
    underutilized_stores: List[str]

    # Statistical significance
    statistical_tests: Dict[str, Dict[str, float]]


@dataclass
class WhatIfScenario:
    """What-if analysis results for bestseller prediction"""
    target_store: str
    similar_stores: List[Tuple[str, float]]  # (store_id, similarity_score)

    # Predicted bestsellers
    predicted_bestsellers: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]

    # Expected performance
    expected_revenue_lift: float
    expected_sales_increase: int

    # Conditions matched
    weather_similarity: float
    size_similarity: float
    demographic_similarity: float
    product_mix_similarity: float

    # Recommendations
    products_to_add: List[str]
    products_to_promote: List[str]
    expected_impact: Dict[str, float]


@dataclass
class ProductRecommendation:
    """Product recommendation for underperforming stores"""
    store_id: str
    current_performance: Dict[str, float]

    # Recommended products
    recommended_products: List[Dict[str, Any]]

    # Expected impact
    revenue_potential: float
    profit_potential: float
    implementation_priority: str  # 'high', 'medium', 'low'

    # Supporting evidence
    success_stories: List[Dict[str, Any]]  # Similar stores that succeeded
    market_demand: Dict[str, float]
    risk_assessment: Dict[str, str]


@dataclass
class MarketBasketInsights:
    """Market basket analysis results"""
    product_pairs: List[Tuple[str, str, float]]  # (product_a, product_b, lift)
    frequent_itemsets: pd.DataFrame
    association_rules: pd.DataFrame

    # Bundle recommendations
    recommended_bundles: List[Dict[str, Any]]
    cross_sell_opportunities: Dict[str, List[str]]

    # Performance metrics
    avg_basket_size: float
    basket_value_distribution: Dict[str, float]


class BestsellerAnalysisSystem:
    """
    Comprehensive bestseller analysis system with machine learning capabilities.

    Features:
    - Top performers identification across stores and categories
    - Cross-store benchmarking and comparison
    - What-if analysis for product performance prediction
    - Intelligent product recommendations
    - Sales velocity tracking and trending
    - Market basket analysis for product associations
    - Seasonal pattern recognition
    - Machine learning models for prediction

    Attributes:
        ml_model: Machine learning model for bestseller prediction
        scaler: Feature scaler for normalization
        label_encoders: Dictionary of label encoders for categorical features
        velocity_thresholds: Thresholds for velocity categorization
        bestseller_thresholds: Percentile thresholds for bestseller categories
    """

    def __init__(
        self,
        bestseller_thresholds: Optional[Dict[str, float]] = None,
        velocity_thresholds: Optional[Dict[str, float]] = None,
        min_sample_size: int = 30,
        confidence_level: float = 0.95,
        ml_model_type: str = 'random_forest',
        use_xgboost: bool = True,
        random_state: int = 42
    ):
        """
        Initialize bestseller analysis system.

        Parameters
        ----------
        bestseller_thresholds : dict, optional
            Percentile thresholds for bestseller categories
            Default: {'superstar': 0.05, 'bestseller': 0.20, 'performer': 0.50, 'average': 0.80}
        velocity_thresholds : dict, optional
            Growth rate thresholds for velocity categories
            Default: {'explosive': 2.0, 'fast': 0.5, 'moderate': 0.1, 'slow': 0.0}
        min_sample_size : int, default=30
            Minimum sample size for statistical analysis
        confidence_level : float, default=0.95
            Confidence level for statistical tests
        ml_model_type : str, default='random_forest'
            Type of ML model: 'random_forest', 'xgboost', or 'gradient_boosting'
        use_xgboost : bool, default=True
            Use XGBoost if available
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.bestseller_thresholds = bestseller_thresholds or {
            'superstar': 0.05,
            'bestseller': 0.20,
            'performer': 0.50,
            'average': 0.80
        }

        self.velocity_thresholds = velocity_thresholds or {
            'explosive': 2.0,  # 200% growth
            'fast': 0.5,       # 50% growth
            'moderate': 0.1,   # 10% growth
            'slow': 0.0        # 0% growth
        }

        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.random_state = random_state

        # ML components
        self.ml_model_type = ml_model_type
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.ml_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Analysis cache
        self.bestseller_cache = {}
        self.velocity_cache = {}
        self.seasonal_patterns = {}

    def identify_top_performers(
        self,
        sales_data: pd.DataFrame,
        store_id: Optional[str] = None,
        category: Optional[str] = None,
        top_n: int = 20,
        metric: str = 'revenue'
    ) -> List[BestsellerMetrics]:
        """
        Identify top performing products by store and category.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Sales data with columns: product_id, store_id, date, quantity, revenue, etc.
        store_id : str, optional
            Filter by specific store
        category : str, optional
            Filter by product category
        top_n : int, default=20
            Number of top performers to return
        metric : str, default='revenue'
            Metric for ranking: 'revenue', 'quantity', 'profit', 'velocity'

        Returns
        -------
        List[BestsellerMetrics]
            List of bestseller metrics for top performers
        """
        # Input validation
        required_cols = ['product_id', 'store_id', 'date', 'quantity', 'revenue']
        missing_cols = [col for col in required_cols if col not in sales_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter data
        df = sales_data.copy()
        if store_id:
            df = df[df['store_id'] == store_id]
        if category:
            df = df[df['category'] == category]

        if len(df) == 0:
            warnings.warn("No data matching filters")
            return []

        # Aggregate by product
        agg_dict = {
            'revenue': 'sum',
            'quantity': 'sum',
            'date': ['min', 'max', 'count']
        }

        # Add optional columns
        if 'profit' in df.columns:
            agg_dict['profit'] = 'sum'
        if 'price' in df.columns:
            agg_dict['price'] = 'mean'
        if 'product_name' in df.columns:
            agg_dict['product_name'] = 'first'
        if 'category' in df.columns:
            agg_dict['category'] = 'first'

        product_stats = df.groupby(['product_id', 'store_id']).agg(agg_dict).reset_index()

        # Flatten column names
        product_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                 for col in product_stats.columns]

        # Calculate derived metrics
        product_stats['days_in_stock'] = (
            pd.to_datetime(product_stats['date_max']) -
            pd.to_datetime(product_stats['date_min'])
        ).dt.days + 1

        product_stats['daily_avg_sales'] = (
            product_stats['revenue_sum'] / product_stats['days_in_stock']
        )

        product_stats['weekly_avg_sales'] = product_stats['daily_avg_sales'] * 7
        product_stats['monthly_avg_sales'] = product_stats['daily_avg_sales'] * 30

        # Calculate profit margin if profit available
        if 'profit_sum' in product_stats.columns:
            product_stats['profit_margin'] = (
                product_stats['profit_sum'] / product_stats['revenue_sum'] * 100
            )
        else:
            product_stats['profit_margin'] = 0.0

        # Rank products
        rank_metric_map = {
            'revenue': 'revenue_sum',
            'quantity': 'quantity_sum',
            'profit': 'profit_sum' if 'profit_sum' in product_stats.columns else 'revenue_sum',
            'velocity': 'daily_avg_sales'
        }

        rank_col = rank_metric_map.get(metric, 'revenue_sum')
        product_stats['rank'] = product_stats[rank_col].rank(ascending=False, method='dense')

        # Get top N
        top_products = product_stats.nsmallest(top_n, 'rank')

        # Calculate revenue contribution
        total_revenue = product_stats['revenue_sum'].sum()
        top_products['revenue_contribution_pct'] = (
            top_products['revenue_sum'] / total_revenue * 100
        )

        # Categorize bestsellers
        top_products['bestseller_category'] = top_products.apply(
            lambda row: self._categorize_bestseller(
                row['rank'],
                len(product_stats)
            ),
            axis=1
        )

        # Calculate growth rate and velocity
        top_products['growth_rate'] = top_products.apply(
            lambda row: self._calculate_growth_rate(df, row['product_id'], row['store_id']),
            axis=1
        )

        top_products['velocity_category'] = top_products['growth_rate'].apply(
            self._categorize_velocity
        )

        # Calculate market share
        if store_id:
            store_total = total_revenue
        else:
            store_total = df.groupby('store_id')['revenue'].sum().get(
                top_products.iloc[0]['store_id'], total_revenue
            )

        top_products['market_share_pct'] = (
            top_products['revenue_sum'] / store_total * 100
        )

        # Detect seasonal patterns
        top_products['seasonal_pattern'] = top_products.apply(
            lambda row: self._detect_seasonal_pattern(df, row['product_id'], row['store_id']),
            axis=1
        )

        # Convert to BestsellerMetrics objects
        results = []
        for _, row in top_products.iterrows():
            metrics = BestsellerMetrics(
                product_id=row['product_id'],
                product_name=row.get('product_name_first', f"Product {row['product_id']}"),
                category=row.get('category_first', 'Unknown'),
                store_id=row['store_id'],
                total_revenue=float(row['revenue_sum']),
                total_quantity=int(row['quantity_sum']),
                avg_price=float(row.get('price_mean', row['revenue_sum'] / row['quantity_sum'])),
                profit_margin=float(row['profit_margin']),
                sales_rank=int(row['rank']),
                revenue_contribution_pct=float(row['revenue_contribution_pct']),
                category=row['bestseller_category'],
                daily_avg_sales=float(row['daily_avg_sales']),
                weekly_avg_sales=float(row['weekly_avg_sales']),
                monthly_avg_sales=float(row['monthly_avg_sales']),
                growth_rate=float(row['growth_rate']),
                velocity_category=row['velocity_category'],
                days_in_stock=int(row['days_in_stock']),
                stockout_days=0,  # Would need inventory data
                availability_rate=100.0,  # Would need inventory data
                market_share_pct=float(row['market_share_pct']),
                category_rank=int(row['rank']),
                store_rank=int(row['rank']),
                seasonal_pattern=row['seasonal_pattern'],
                peak_season_multiplier=1.0,
                metadata={
                    'analysis_date': datetime.now().isoformat(),
                    'metric_used': metric,
                    'sample_size': len(product_stats)
                }
            )
            results.append(metrics)

        return results

    def compare_across_stores(
        self,
        sales_data: pd.DataFrame,
        product_id: str,
        min_stores: int = 3
    ) -> CrossStoreComparison:
        """
        Compare product performance across multiple stores.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Sales data across stores
        product_id : str
            Product to analyze
        min_stores : int, default=3
            Minimum number of stores for comparison

        Returns
        -------
        CrossStoreComparison
            Cross-store comparison results
        """
        # Filter for specific product
        product_data = sales_data[sales_data['product_id'] == product_id].copy()

        if len(product_data) == 0:
            raise ValueError(f"No data found for product {product_id}")

        # Aggregate by store
        store_stats = product_data.groupby('store_id').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'date': 'count'
        }).reset_index()

        if len(store_stats) < min_stores:
            warnings.warn(
                f"Product only sold in {len(store_stats)} stores (minimum {min_stores})"
            )

        # Calculate ranks within each store
        all_products_by_store = sales_data.groupby(['store_id', 'product_id']).agg({
            'revenue': 'sum'
        }).reset_index()

        ranks = []
        for store in store_stats['store_id']:
            store_products = all_products_by_store[
                all_products_by_store['store_id'] == store
            ].copy()
            store_products['rank'] = store_products['revenue'].rank(
                ascending=False, method='dense'
            )
            product_rank = store_products[
                store_products['product_id'] == product_id
            ]['rank'].values[0]
            ranks.append({'store_id': store, 'rank': product_rank})

        ranks_df = pd.DataFrame(ranks)
        store_stats = store_stats.merge(ranks_df, on='store_id')

        # Performance metrics
        avg_rank = store_stats['rank'].mean()
        rank_variance = store_stats['rank'].var()
        revenue_variance = store_stats['revenue'].var()

        # Identify best and worst stores
        best_store = store_stats.nsmallest(1, 'rank')['store_id'].values[0]
        worst_store = store_stats.nlargest(1, 'rank')['store_id'].values[0]

        # Calculate performance consistency (inverse of coefficient of variation)
        cv = store_stats['revenue'].std() / store_stats['revenue'].mean()
        performance_consistency = 1.0 / (1.0 + cv)

        # Identify high potential stores (below median rank)
        median_rank = store_stats['rank'].median()
        high_potential = store_stats[store_stats['rank'] < median_rank]['store_id'].tolist()
        underutilized = store_stats[store_stats['rank'] > median_rank]['store_id'].tolist()

        # Statistical tests
        if len(store_stats) >= 3:
            # ANOVA test for revenue differences
            store_groups = [
                product_data[product_data['store_id'] == store]['revenue'].values
                for store in store_stats['store_id']
            ]
            f_stat, p_value = stats.f_oneway(*store_groups)

            statistical_tests = {
                'anova': {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < (1 - self.confidence_level)
                }
            }
        else:
            statistical_tests = {}

        # Get product name
        product_name = product_data['product_name'].iloc[0] if 'product_name' in product_data.columns else f"Product {product_id}"

        return CrossStoreComparison(
            product_id=product_id,
            product_name=product_name,
            stores_selling=store_stats['store_id'].tolist(),
            avg_rank=float(avg_rank),
            best_performing_store=best_store,
            worst_performing_store=worst_store,
            rank_variance=float(rank_variance),
            revenue_variance=float(revenue_variance),
            performance_consistency=float(performance_consistency),
            high_potential_stores=high_potential,
            underutilized_stores=underutilized,
            statistical_tests=statistical_tests
        )

    def what_if_analysis(
        self,
        sales_data: pd.DataFrame,
        target_store: str,
        store_features: pd.DataFrame,
        weather_data: Optional[pd.DataFrame] = None,
        top_n_similar: int = 5,
        min_similarity: float = 0.7
    ) -> WhatIfScenario:
        """
        Predict bestsellers for similar stores with matching weather/scale conditions.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Historical sales data
        target_store : str
            Store to generate predictions for
        store_features : pd.DataFrame
            Store characteristics (size, location, demographics, etc.)
        weather_data : pd.DataFrame, optional
            Weather patterns by store
        top_n_similar : int, default=5
            Number of similar stores to consider
        min_similarity : float, default=0.7
            Minimum similarity score (0-1)

        Returns
        -------
        WhatIfScenario
            What-if analysis results with predictions
        """
        # Find similar stores
        similar_stores = self._find_similar_stores(
            target_store,
            store_features,
            weather_data,
            top_n=top_n_similar,
            min_similarity=min_similarity
        )

        if len(similar_stores) == 0:
            raise ValueError(
                f"No similar stores found with similarity >= {min_similarity}"
            )

        # Get bestsellers from similar stores
        similar_store_ids = [store_id for store_id, _ in similar_stores]
        similar_bestsellers = sales_data[
            sales_data['store_id'].isin(similar_store_ids)
        ].copy()

        # Aggregate product performance across similar stores
        product_performance = similar_bestsellers.groupby('product_id').agg({
            'revenue': ['sum', 'mean', 'std'],
            'quantity': ['sum', 'mean'],
            'store_id': 'nunique'
        }).reset_index()

        product_performance.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                       for col in product_performance.columns]

        # Calculate confidence scores
        product_performance['confidence'] = (
            product_performance['store_id_nunique'] / len(similar_stores) *
            (1.0 - product_performance['revenue_std'].fillna(0) /
             product_performance['revenue_mean'].clip(lower=1))
        ).clip(0, 1)

        # Rank by revenue and confidence
        product_performance['score'] = (
            product_performance['revenue_sum'] *
            product_performance['confidence']
        )
        product_performance = product_performance.sort_values('score', ascending=False)

        # Get current products in target store
        target_products = set(
            sales_data[sales_data['store_id'] == target_store]['product_id'].unique()
        )

        # Identify products to add (not currently in store)
        products_to_add = product_performance[
            ~product_performance['product_id'].isin(target_products)
        ].head(10)['product_id'].tolist()

        # Identify products to promote (already in store but underperforming)
        products_to_promote = product_performance[
            product_performance['product_id'].isin(target_products)
        ].head(10)['product_id'].tolist()

        # Calculate expected impact
        if len(products_to_add) > 0:
            expected_revenue_lift = product_performance[
                product_performance['product_id'].isin(products_to_add)
            ]['revenue_mean'].sum()

            expected_sales_increase = int(product_performance[
                product_performance['product_id'].isin(products_to_add)
            ]['quantity_mean'].sum())
        else:
            expected_revenue_lift = 0.0
            expected_sales_increase = 0

        # Build predicted bestsellers list
        predicted_bestsellers = []
        for _, row in product_performance.head(20).iterrows():
            predicted_bestsellers.append({
                'product_id': row['product_id'],
                'expected_revenue': float(row['revenue_mean']),
                'expected_quantity': int(row['quantity_mean']),
                'confidence': float(row['confidence']),
                'selling_in_stores': int(row['store_id_nunique']),
                'already_in_store': row['product_id'] in target_products
            })

        # Confidence scores by product
        confidence_scores = {
            row['product_id']: float(row['confidence'])
            for _, row in product_performance.iterrows()
        }

        # Calculate similarity metrics
        target_features = store_features[store_features['store_id'] == target_store].iloc[0]

        weather_similarity = 1.0
        if weather_data is not None:
            weather_similarity = self._calculate_weather_similarity(
                target_store, similar_store_ids, weather_data
            )

        size_similarity = np.mean([score for _, score in similar_stores])

        # Expected impact details
        expected_impact = {
            'revenue_lift_pct': float(expected_revenue_lift /
                                     sales_data[sales_data['store_id'] == target_store]['revenue'].sum() * 100
                                     if len(sales_data[sales_data['store_id'] == target_store]) > 0 else 0),
            'new_products': len(products_to_add),
            'products_to_optimize': len(products_to_promote),
            'confidence_avg': float(product_performance.head(20)['confidence'].mean())
        }

        return WhatIfScenario(
            target_store=target_store,
            similar_stores=similar_stores,
            predicted_bestsellers=predicted_bestsellers,
            confidence_scores=confidence_scores,
            expected_revenue_lift=float(expected_revenue_lift),
            expected_sales_increase=expected_sales_increase,
            weather_similarity=float(weather_similarity),
            size_similarity=float(size_similarity),
            demographic_similarity=float(size_similarity),  # Simplified
            product_mix_similarity=float(size_similarity),  # Simplified
            products_to_add=products_to_add,
            products_to_promote=products_to_promote,
            expected_impact=expected_impact
        )

    def recommend_products(
        self,
        sales_data: pd.DataFrame,
        store_id: str,
        store_features: pd.DataFrame,
        top_n: int = 10,
        include_success_stories: bool = True
    ) -> ProductRecommendation:
        """
        Generate product recommendations for underperforming stores.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Historical sales data
        store_id : str
            Target store for recommendations
        store_features : pd.DataFrame
            Store characteristics
        top_n : int, default=10
            Number of products to recommend
        include_success_stories : bool, default=True
            Include success stories from similar stores

        Returns
        -------
        ProductRecommendation
            Product recommendations with expected impact
        """
        # Assess current performance
        store_sales = sales_data[sales_data['store_id'] == store_id]

        if len(store_sales) == 0:
            raise ValueError(f"No sales data for store {store_id}")

        current_performance = {
            'total_revenue': float(store_sales['revenue'].sum()),
            'total_products': int(store_sales['product_id'].nunique()),
            'avg_daily_revenue': float(store_sales.groupby('date')['revenue'].sum().mean()),
            'days_of_data': int(store_sales['date'].nunique())
        }

        # Find similar high-performing stores
        similar_stores = self._find_similar_stores(
            store_id,
            store_features,
            top_n=5,
            performance_filter='high'
        )

        # Get products from similar stores not in target store
        target_products = set(store_sales['product_id'].unique())
        similar_store_ids = [s[0] for s in similar_stores]

        similar_sales = sales_data[
            sales_data['store_id'].isin(similar_store_ids)
        ]

        # Aggregate product performance
        candidate_products = similar_sales.groupby('product_id').agg({
            'revenue': ['sum', 'mean'],
            'quantity': ['sum', 'mean'],
            'store_id': 'nunique'
        }).reset_index()

        candidate_products.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                      for col in candidate_products.columns]

        # Filter out products already in store
        new_products = candidate_products[
            ~candidate_products['product_id'].isin(target_products)
        ].copy()

        # Score products
        new_products['score'] = (
            new_products['revenue_mean'] *
            new_products['store_id_nunique']
        )

        # Top recommendations
        recommendations = new_products.nlargest(top_n, 'score')

        # Build recommendation list
        recommended_products = []
        for _, row in recommendations.iterrows():
            product_info = {
                'product_id': row['product_id'],
                'expected_monthly_revenue': float(row['revenue_mean'] * 30),
                'expected_monthly_quantity': int(row['quantity_mean'] * 30),
                'selling_in_similar_stores': int(row['store_id_nunique']),
                'confidence_score': float(row['store_id_nunique'] / len(similar_stores))
            }
            recommended_products.append(product_info)

        # Calculate potential impact
        revenue_potential = recommendations['revenue_mean'].sum() * 30  # Monthly
        profit_potential = revenue_potential * 0.25  # Assume 25% margin

        # Determine priority
        if revenue_potential > current_performance['avg_daily_revenue'] * 10:
            priority = 'high'
        elif revenue_potential > current_performance['avg_daily_revenue'] * 5:
            priority = 'medium'
        else:
            priority = 'low'

        # Success stories
        success_stories = []
        if include_success_stories:
            for similar_store_id, similarity in similar_stores[:3]:
                similar_store_sales = sales_data[sales_data['store_id'] == similar_store_id]

                # Find products that performed well
                top_products = similar_store_sales.groupby('product_id').agg({
                    'revenue': 'sum'
                }).nlargest(5, 'revenue')

                success_stories.append({
                    'store_id': similar_store_id,
                    'similarity_score': float(similarity),
                    'revenue_achieved': float(similar_store_sales['revenue'].sum()),
                    'top_products': top_products.index.tolist()
                })

        # Risk assessment
        risk_assessment = {
            'market_demand': 'high' if len(similar_stores) >= 3 else 'medium',
            'implementation_complexity': 'low',
            'inventory_risk': 'medium',
            'overall_risk': 'low' if priority == 'high' else 'medium'
        }

        return ProductRecommendation(
            store_id=store_id,
            current_performance=current_performance,
            recommended_products=recommended_products,
            revenue_potential=float(revenue_potential),
            profit_potential=float(profit_potential),
            implementation_priority=priority,
            success_stories=success_stories,
            market_demand={'score': float(len(similar_stores) / 5)},
            risk_assessment=risk_assessment
        )

    def calculate_sales_velocity(
        self,
        sales_data: pd.DataFrame,
        product_id: Optional[str] = None,
        store_id: Optional[str] = None,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculate sales velocity with trend analysis.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Sales data with date column
        product_id : str, optional
            Filter by product
        store_id : str, optional
            Filter by store
        window_days : int, default=30
            Rolling window for velocity calculation

        Returns
        -------
        pd.DataFrame
            Velocity metrics with trends
        """
        df = sales_data.copy()

        # Apply filters
        if product_id:
            df = df[df['product_id'] == product_id]
        if store_id:
            df = df[df['store_id'] == store_id]

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Group by date and product
        group_cols = ['date']
        if not product_id:
            group_cols.append('product_id')
        if not store_id:
            group_cols.append('store_id')

        daily_sales = df.groupby(group_cols).agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()

        # Calculate rolling metrics
        velocity_metrics = []

        for group_key in daily_sales.groupby([c for c in group_cols if c != 'date']):
            if len(group_cols) > 1:
                group_name, group_data = group_key
            else:
                group_data = daily_sales
                group_name = 'all'

            group_data = group_data.sort_values('date')

            # Rolling averages
            group_data['velocity_7d'] = group_data['revenue'].rolling(
                window=min(7, len(group_data)), min_periods=1
            ).mean()

            group_data['velocity_30d'] = group_data['revenue'].rolling(
                window=min(window_days, len(group_data)), min_periods=1
            ).mean()

            # Growth rate
            group_data['growth_rate'] = group_data['revenue'].pct_change(
                periods=min(7, len(group_data))
            )

            # Acceleration
            group_data['acceleration'] = group_data['velocity_7d'].pct_change()

            # Trend (linear regression slope)
            if len(group_data) >= 7:
                x = np.arange(len(group_data))
                slope, _, r_value, _, _ = stats.linregress(x, group_data['revenue'])
                group_data['trend_slope'] = slope
                group_data['trend_r2'] = r_value ** 2
            else:
                group_data['trend_slope'] = 0
                group_data['trend_r2'] = 0

            velocity_metrics.append(group_data)

        result = pd.concat(velocity_metrics, ignore_index=True)

        # Add velocity category
        result['velocity_category'] = result['growth_rate'].apply(
            self._categorize_velocity
        )

        return result

    def market_basket_analysis(
        self,
        transaction_data: pd.DataFrame,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
        max_length: int = 3
    ) -> MarketBasketInsights:
        """
        Perform market basket analysis to identify product associations.

        Parameters
        ----------
        transaction_data : pd.DataFrame
            Transaction data with transaction_id and product_id columns
        min_support : float, default=0.01
            Minimum support threshold (1%)
        min_confidence : float, default=0.1
            Minimum confidence threshold (10%)
        min_lift : float, default=1.0
            Minimum lift threshold
        max_length : int, default=3
            Maximum itemset length

        Returns
        -------
        MarketBasketInsights
            Market basket analysis results
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend is required for market basket analysis. "
                "Install with: pip install mlxtend>=0.19.0"
            )

        # Create transaction matrix
        basket = transaction_data.groupby(['transaction_id', 'product_id'])['quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # Find frequent itemsets
        frequent_itemsets = apriori(
            basket,
            min_support=min_support,
            max_len=max_length,
            use_colnames=True
        )

        if len(frequent_itemsets) == 0:
            warnings.warn(
                f"No frequent itemsets found with min_support={min_support}. "
                "Try lowering the threshold."
            )
            return MarketBasketInsights(
                product_pairs=[],
                frequent_itemsets=pd.DataFrame(),
                association_rules=pd.DataFrame(),
                recommended_bundles=[],
                cross_sell_opportunities={},
                avg_basket_size=0.0,
                basket_value_distribution={}
            )

        # Generate association rules
        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=min_lift
        )

        # Filter by confidence
        rules = rules[rules['confidence'] >= min_confidence]

        # Extract product pairs with lift
        product_pairs = []
        for _, rule in rules.iterrows():
            antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(rule['antecedents'])
            consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(rule['consequents'])

            product_pairs.append((
                str(antecedent),
                str(consequent),
                float(rule['lift'])
            ))

        # Sort by lift
        product_pairs.sort(key=lambda x: x[2], reverse=True)

        # Generate bundle recommendations
        recommended_bundles = []
        for itemset in frequent_itemsets.nlargest(10, 'support')['itemsets']:
            if len(itemset) >= 2:
                recommended_bundles.append({
                    'products': list(itemset),
                    'support': float(frequent_itemsets[
                        frequent_itemsets['itemsets'] == itemset
                    ]['support'].values[0]),
                    'bundle_type': 'frequent' if len(itemset) == 2 else 'multi-product'
                })

        # Cross-sell opportunities
        cross_sell = {}
        for _, rule in rules.iterrows():
            for antecedent in rule['antecedents']:
                if antecedent not in cross_sell:
                    cross_sell[antecedent] = []

                for consequent in rule['consequents']:
                    cross_sell[antecedent].append({
                        'product': consequent,
                        'confidence': float(rule['confidence']),
                        'lift': float(rule['lift'])
                    })

        # Calculate basket metrics
        basket_sizes = basket.sum(axis=1)
        avg_basket_size = float(basket_sizes.mean())

        basket_value_dist = {
            'min': float(basket_sizes.min()),
            'max': float(basket_sizes.max()),
            'mean': float(basket_sizes.mean()),
            'median': float(basket_sizes.median()),
            'std': float(basket_sizes.std())
        }

        return MarketBasketInsights(
            product_pairs=product_pairs[:20],  # Top 20 pairs
            frequent_itemsets=frequent_itemsets,
            association_rules=rules,
            recommended_bundles=recommended_bundles,
            cross_sell_opportunities=cross_sell,
            avg_basket_size=avg_basket_size,
            basket_value_distribution=basket_value_dist
        )

    def analyze_seasonal_trends(
        self,
        sales_data: pd.DataFrame,
        product_id: Optional[str] = None,
        min_cycles: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze seasonal bestseller trends and patterns.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Sales data with date column
        product_id : str, optional
            Analyze specific product
        min_cycles : int, default=2
            Minimum seasonal cycles required

        Returns
        -------
        Dict[str, Any]
            Seasonal analysis results
        """
        df = sales_data.copy()
        df['date'] = pd.to_datetime(df['date'])

        if product_id:
            df = df[df['product_id'] == product_id]

        # Add time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week'] = df['date'].dt.isocalendar().week
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Monthly patterns
        monthly_sales = df.groupby('month')['revenue'].agg(['mean', 'std', 'count'])
        monthly_sales['cv'] = monthly_sales['std'] / monthly_sales['mean']

        # Identify peak months
        peak_month = monthly_sales['mean'].idxmax()
        low_month = monthly_sales['mean'].idxmin()

        # Seasonal pattern detection
        seasonal_strength = monthly_sales['mean'].std() / monthly_sales['mean'].mean()

        if seasonal_strength > 0.5:
            # Strong seasonality
            if peak_month in [6, 7, 8]:
                pattern = SeasonalPattern.SUMMER_PEAK
            elif peak_month in [12, 1, 2]:
                pattern = SeasonalPattern.WINTER_PEAK
            elif peak_month in [3, 4, 5]:
                pattern = SeasonalPattern.SPRING_PEAK
            elif peak_month in [9, 10, 11]:
                pattern = SeasonalPattern.FALL_PEAK
            else:
                pattern = SeasonalPattern.YEAR_ROUND
        elif df.groupby('is_weekend')['revenue'].mean()[1] > df.groupby('is_weekend')['revenue'].mean()[0] * 1.2:
            pattern = SeasonalPattern.WEEKEND_DRIVEN
        else:
            pattern = SeasonalPattern.YEAR_ROUND

        # Quarter analysis
        quarterly_sales = df.groupby('quarter')['revenue'].agg(['mean', 'std'])

        # Weekend vs weekday
        weekend_analysis = df.groupby('is_weekend')['revenue'].agg(['mean', 'sum', 'count'])

        # Year-over-year growth
        yearly_sales = df.groupby('year')['revenue'].sum()
        yoy_growth = yearly_sales.pct_change().mean() if len(yearly_sales) > 1 else 0

        return {
            'seasonal_pattern': pattern,
            'seasonal_strength': float(seasonal_strength),
            'peak_month': int(peak_month),
            'low_month': int(low_month),
            'peak_multiplier': float(monthly_sales.loc[peak_month, 'mean'] / monthly_sales['mean'].mean()),
            'monthly_patterns': monthly_sales.to_dict('index'),
            'quarterly_patterns': quarterly_sales.to_dict('index'),
            'weekend_effect': {
                'weekend_avg': float(weekend_analysis.loc[1, 'mean']) if 1 in weekend_analysis.index else 0,
                'weekday_avg': float(weekend_analysis.loc[0, 'mean']) if 0 in weekend_analysis.index else 0,
                'weekend_premium_pct': float(
                    (weekend_analysis.loc[1, 'mean'] / weekend_analysis.loc[0, 'mean'] - 1) * 100
                    if 0 in weekend_analysis.index and 1 in weekend_analysis.index else 0
                )
            },
            'yoy_growth_rate': float(yoy_growth),
            'data_span_years': float((df['date'].max() - df['date'].min()).days / 365.25),
            'sufficient_history': (df['date'].max() - df['date'].min()).days >= 365 * min_cycles
        }

    def train_bestseller_predictor(
        self,
        sales_data: pd.DataFrame,
        feature_columns: List[str],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train machine learning model to predict bestsellers.

        Parameters
        ----------
        sales_data : pd.DataFrame
            Training data with features and target
        feature_columns : List[str]
            Columns to use as features
        test_size : float, default=0.2
            Proportion of data for testing

        Returns
        -------
        Dict[str, Any]
            Training results and model performance metrics
        """
        # Prepare features
        X = sales_data[feature_columns].copy()

        # Create target variable (bestseller classification)
        sales_data['is_bestseller'] = (
            sales_data['revenue'].rank(pct=True) >= (1 - self.bestseller_thresholds['bestseller'])
        ).astype(int)

        y = sales_data['is_bestseller']

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Train model
        if self.use_xgboost and self.ml_model_type == 'xgboost':
            self.ml_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.ml_model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.ml_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )

        # Train
        self.ml_model.fit(X_train, y_train)

        # Evaluate
        train_score = self.ml_model.score(X_train, y_train)
        test_score = self.ml_model.score(X_test, y_test)

        # Predictions
        y_pred = self.ml_model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self.ml_model, X_scaled, y, cv=5, scoring='accuracy'
        )

        # Feature importance
        if hasattr(self.ml_model, 'feature_importances_'):
            feature_importance = dict(zip(
                feature_columns,
                self.ml_model.feature_importances_
            ))
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        else:
            feature_importance = {}

        return {
            'model_type': self.ml_model_type,
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'class_distribution': {
                'train': dict(pd.Series(y_train).value_counts()),
                'test': dict(pd.Series(y_test).value_counts())
            }
        }

    def predict_bestseller_probability(
        self,
        product_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict probability of products becoming bestsellers.

        Parameters
        ----------
        product_features : pd.DataFrame
            Product features for prediction

        Returns
        -------
        pd.DataFrame
            Predictions with probability scores
        """
        if self.ml_model is None:
            raise ValueError("Model not trained. Call train_bestseller_predictor first.")

        X = product_features.copy()

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                # Handle unknown categories
                X[col] = 0

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.ml_model.predict(X_scaled)
        probabilities = self.ml_model.predict_proba(X_scaled)[:, 1]

        # Create results dataframe
        results = product_features.copy()
        results['is_bestseller_predicted'] = predictions
        results['bestseller_probability'] = probabilities
        results['confidence_level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        )

        return results

    # Helper methods

    def _categorize_bestseller(self, rank: int, total_products: int) -> BestsellerCategory:
        """Categorize product as bestseller based on rank."""
        percentile = rank / total_products

        if percentile <= self.bestseller_thresholds['superstar']:
            return BestsellerCategory.SUPERSTAR
        elif percentile <= self.bestseller_thresholds['bestseller']:
            return BestsellerCategory.BESTSELLER
        elif percentile <= self.bestseller_thresholds['performer']:
            return BestsellerCategory.PERFORMER
        elif percentile <= self.bestseller_thresholds['average']:
            return BestsellerCategory.AVERAGE
        else:
            return BestsellerCategory.UNDERPERFORMER

    def _categorize_velocity(self, growth_rate: float) -> VelocityCategory:
        """Categorize velocity based on growth rate."""
        if pd.isna(growth_rate):
            return VelocityCategory.MODERATE

        if growth_rate >= self.velocity_thresholds['explosive']:
            return VelocityCategory.EXPLOSIVE
        elif growth_rate >= self.velocity_thresholds['fast']:
            return VelocityCategory.FAST
        elif growth_rate >= self.velocity_thresholds['moderate']:
            return VelocityCategory.MODERATE
        elif growth_rate >= self.velocity_thresholds['slow']:
            return VelocityCategory.SLOW
        else:
            return VelocityCategory.DECLINING

    def _calculate_growth_rate(
        self,
        df: pd.DataFrame,
        product_id: str,
        store_id: str,
        periods: int = 7
    ) -> float:
        """Calculate growth rate for a product."""
        product_data = df[
            (df['product_id'] == product_id) &
            (df['store_id'] == store_id)
        ].copy()

        if len(product_data) < periods * 2:
            return 0.0

        product_data['date'] = pd.to_datetime(product_data['date'])
        product_data = product_data.sort_values('date')

        # Recent vs previous period
        recent = product_data.tail(periods)['revenue'].sum()
        previous = product_data.iloc[-periods*2:-periods]['revenue'].sum()

        if previous == 0:
            return 0.0

        return (recent - previous) / previous

    def _detect_seasonal_pattern(
        self,
        df: pd.DataFrame,
        product_id: str,
        store_id: str
    ) -> Optional[SeasonalPattern]:
        """Detect seasonal pattern for a product."""
        product_data = df[
            (df['product_id'] == product_id) &
            (df['store_id'] == store_id)
        ].copy()

        if len(product_data) < 90:  # Need at least 3 months
            return None

        product_data['date'] = pd.to_datetime(product_data['date'])
        product_data['month'] = product_data['date'].dt.month

        monthly_avg = product_data.groupby('month')['revenue'].mean()

        if len(monthly_avg) < 3:
            return SeasonalPattern.YEAR_ROUND

        peak_month = monthly_avg.idxmax()

        # Determine pattern based on peak month
        if peak_month in [6, 7, 8]:
            return SeasonalPattern.SUMMER_PEAK
        elif peak_month in [12, 1, 2]:
            return SeasonalPattern.WINTER_PEAK
        elif peak_month in [3, 4, 5]:
            return SeasonalPattern.SPRING_PEAK
        elif peak_month in [9, 10, 11]:
            return SeasonalPattern.FALL_PEAK
        else:
            return SeasonalPattern.YEAR_ROUND

    def _find_similar_stores(
        self,
        target_store: str,
        store_features: pd.DataFrame,
        weather_data: Optional[pd.DataFrame] = None,
        top_n: int = 5,
        min_similarity: float = 0.7,
        performance_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Find similar stores based on features."""
        if target_store not in store_features['store_id'].values:
            raise ValueError(f"Store {target_store} not found in features")

        # Prepare features for similarity
        feature_cols = [col for col in store_features.columns
                       if col not in ['store_id', 'store_name']]

        X = store_features[feature_cols].copy()

        # Encode categorical
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))

        # Calculate similarity
        target_idx = store_features[store_features['store_id'] == target_store].index[0]
        target_vector = X_scaled[target_idx].reshape(1, -1)

        similarities = cosine_similarity(target_vector, X_scaled)[0]

        # Get similar stores
        store_ids = store_features['store_id'].values
        similar = [(store_ids[i], similarities[i])
                  for i in range(len(store_ids))
                  if store_ids[i] != target_store and similarities[i] >= min_similarity]

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)

        return similar[:top_n]

    def _calculate_weather_similarity(
        self,
        target_store: str,
        similar_stores: List[str],
        weather_data: pd.DataFrame
    ) -> float:
        """Calculate weather pattern similarity."""
        # Simplified weather similarity
        # In production, would use actual weather pattern matching
        return 0.85  # Placeholder


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']
    products = [f'P{i:03d}' for i in range(1, 51)]

    # Generate synthetic sales data
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
                    'revenue': quantity * price
                })

    sales_df = pd.DataFrame(data)

    # Initialize system
    analyzer = BestsellerAnalysisSystem()

    # Test top performers identification
    print("=" * 80)
    print("TOP PERFORMERS ANALYSIS")
    print("=" * 80)

    top_performers = analyzer.identify_top_performers(
        sales_df,
        store_id='S001',
        top_n=10,
        metric='revenue'
    )

    for i, performer in enumerate(top_performers[:5], 1):
        print(f"\n{i}. {performer.product_name}")
        print(f"   Revenue: ${performer.total_revenue:,.2f}")
        print(f"   Category: {performer.category.value}")
        print(f"   Velocity: {performer.velocity_category.value}")
        print(f"   Growth Rate: {performer.growth_rate:.1%}")
        print(f"   Market Share: {performer.market_share_pct:.2f}%")

    print("\n" + "=" * 80)
    print("BESTSELLER ANALYSIS SYSTEM - Test Complete")
    print("=" * 80)
