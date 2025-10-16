"""
Cross-Store Comparison Accuracy Tests
======================================

Tests accuracy of store comparison metrics,
benchmarking calculations, and ranking algorithms.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats


@pytest.mark.comparison
class TestStoreMetricsCalculation:
    """Test calculation of store performance metrics."""

    def test_total_sales_by_store(self, sample_data):
        """Test total sales calculation per store."""
        sales_by_store = sample_data.groupby('Store')['Sales'].sum()

        assert len(sales_by_store) == sample_data['Store'].nunique()
        assert all(sales_by_store > 0)
        assert sales_by_store.sum() == sample_data['Sales'].sum()

    def test_average_sales_by_store(self, sample_data):
        """Test average sales calculation per store."""
        avg_sales = sample_data.groupby('Store')['Sales'].mean()

        assert len(avg_sales) == sample_data['Store'].nunique()
        assert all(avg_sales > 0)

    def test_sales_per_customer_metric(self, sample_data):
        """Test sales per customer metric."""
        metrics = sample_data.groupby('Store').apply(
            lambda x: x['Sales'].sum() / x['Customers'].sum()
        )

        assert len(metrics) > 0
        assert all(metrics > 0)
        assert all(np.isfinite(metrics))

    def test_conversion_rate_metric(self, sample_data):
        """Test customer conversion rate."""
        # Assuming conversion is customers / potential customers
        sample_data['PotentialCustomers'] = sample_data['Customers'] * 1.5

        conversion = sample_data.groupby('Store').apply(
            lambda x: (x['Customers'].sum() / x['PotentialCustomers'].sum()) * 100
        )

        assert len(conversion) > 0
        assert all(conversion >= 0)
        assert all(conversion <= 100)

    def test_growth_rate_calculation(self, sample_data):
        """Test year-over-year growth rate calculation."""
        sample_data['Year'] = sample_data['Date'].dt.year

        # Calculate YoY growth
        yearly_sales = sample_data.groupby(['Store', 'Year'])['Sales'].sum().unstack()

        if len(yearly_sales.columns) >= 2:
            growth = ((yearly_sales.iloc[:, -1] - yearly_sales.iloc[:, -2]) /
                     yearly_sales.iloc[:, -2] * 100)

            assert len(growth) > 0
            assert all(np.isfinite(growth))


@pytest.mark.comparison
class TestStoreRanking:
    """Test store ranking algorithms."""

    def test_simple_ranking_by_sales(self, sample_data):
        """Test ranking stores by total sales."""
        sales_by_store = sample_data.groupby('Store')['Sales'].sum()
        ranking = sales_by_store.rank(ascending=False, method='min')

        assert len(ranking) == len(sales_by_store)
        assert ranking.min() == 1  # Top rank
        assert ranking.max() <= len(sales_by_store)
        assert len(ranking.unique()) <= len(ranking)

    def test_composite_score_ranking(self, sample_data):
        """Test ranking by composite score."""
        # Calculate multiple metrics
        metrics = sample_data.groupby('Store').agg({
            'Sales': 'sum',
            'Customers': 'sum',
            'Promo': 'mean'
        })

        # Normalize metrics to 0-100 scale
        normalized = (metrics - metrics.min()) / (metrics.max() - metrics.min()) * 100

        # Composite score (weighted average)
        weights = {'Sales': 0.5, 'Customers': 0.3, 'Promo': 0.2}
        composite = sum(normalized[col] * weights[col] for col in weights.keys())

        ranking = composite.rank(ascending=False)

        assert len(ranking) == len(metrics)
        assert all(composite >= 0)
        assert all(composite <= 100)

    def test_percentile_ranking(self, sample_data):
        """Test percentile-based ranking."""
        sales_by_store = sample_data.groupby('Store')['Sales'].sum()

        percentiles = sales_by_store.rank(pct=True) * 100

        assert len(percentiles) == len(sales_by_store)
        assert all(percentiles >= 0)
        assert all(percentiles <= 100)

    def test_tier_classification(self, sample_data):
        """Test classification into performance tiers."""
        sales_by_store = sample_data.groupby('Store')['Sales'].sum()
        percentiles = sales_by_store.rank(pct=True)

        def classify_tier(percentile):
            if percentile >= 0.75:
                return 'Top'
            elif percentile >= 0.5:
                return 'High'
            elif percentile >= 0.25:
                return 'Medium'
            else:
                return 'Low'

        tiers = percentiles.apply(classify_tier)

        assert len(tiers) == len(sales_by_store)
        assert set(tiers.unique()).issubset({'Top', 'High', 'Medium', 'Low'})


@pytest.mark.comparison
class TestBenchmarking:
    """Test benchmarking calculations."""

    def test_benchmark_vs_average(self, sample_data):
        """Test comparison against average performance."""
        store_sales = sample_data.groupby('Store')['Sales'].sum()
        average = store_sales.mean()

        variance = ((store_sales - average) / average * 100)

        assert all(np.isfinite(variance))
        assert variance.mean() == pytest.approx(0, abs=1e-10)  # Should balance out

    def test_benchmark_vs_top_performer(self, sample_data):
        """Test comparison against top performer."""
        store_sales = sample_data.groupby('Store')['Sales'].sum()
        top_performer = store_sales.max()

        gap = ((top_performer - store_sales) / top_performer * 100)

        assert all(gap >= 0)
        assert all(gap <= 100)
        assert gap.min() == 0  # Top performer has 0 gap

    def test_benchmark_vs_target(self):
        """Test comparison against target."""
        actual = pd.Series([8000, 12000, 15000, 9500, 11000])
        target = pd.Series([10000, 10000, 10000, 10000, 10000])

        achievement = (actual / target * 100)
        variance = actual - target

        assert len(achievement) == len(actual)
        assert all(achievement > 0)
        assert sum(variance > 0) > 0  # Some above target
        assert sum(variance < 0) > 0  # Some below target

    def test_peer_group_benchmarking(self, sample_data):
        """Test benchmarking within peer groups."""
        # Group stores by size (based on average customers)
        store_size = sample_data.groupby('Store')['Customers'].mean()
        size_quartiles = pd.qcut(store_size, q=4, labels=['Small', 'Medium', 'Large', 'XLarge'])

        # Calculate performance within each peer group
        sample_data['StoreSize'] = sample_data['Store'].map(size_quartiles)

        peer_benchmarks = sample_data.groupby('StoreSize')['Sales'].agg(['mean', 'median', 'std'])

        assert len(peer_benchmarks) <= 4
        assert all(peer_benchmarks['mean'] > 0)


@pytest.mark.comparison
class TestStatisticalSignificance:
    """Test statistical significance of comparisons."""

    def test_ttest_store_comparison(self, sample_data):
        """Test t-test for comparing two stores."""
        store1_sales = sample_data[sample_data['Store'] == 1]['Sales']
        store2_sales = sample_data[sample_data['Store'] == 2]['Sales']

        t_stat, p_value = stats.ttest_ind(store1_sales, store2_sales)

        assert np.isfinite(t_stat)
        assert 0 <= p_value <= 1

        # Significant if p < 0.05
        is_significant = p_value < 0.05

        assert isinstance(is_significant, (bool, np.bool_))

    def test_anova_multiple_stores(self, sample_data):
        """Test ANOVA for comparing multiple stores."""
        # Get sales for first 5 stores
        store_groups = [
            sample_data[sample_data['Store'] == i]['Sales'].values
            for i in range(1, 6)
        ]

        f_stat, p_value = stats.f_oneway(*store_groups)

        assert np.isfinite(f_stat)
        assert 0 <= p_value <= 1

    def test_confidence_intervals(self, sample_data):
        """Test confidence interval calculation."""
        store_sales = sample_data.groupby('Store')['Sales'].apply(list)

        def calc_ci(data, confidence=0.95):
            mean = np.mean(data)
            sem = stats.sem(data)
            interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
            return mean - interval, mean + interval

        confidence_intervals = store_sales.apply(calc_ci)

        assert len(confidence_intervals) > 0
        assert all(ci[0] < ci[1] for ci in confidence_intervals)

    def test_effect_size_calculation(self, sample_data):
        """Test effect size (Cohen's d) calculation."""
        store1_sales = sample_data[sample_data['Store'] == 1]['Sales']
        store2_sales = sample_data[sample_data['Store'] == 2]['Sales']

        mean1, mean2 = store1_sales.mean(), store2_sales.mean()
        std1, std2 = store1_sales.std(), store2_sales.std()

        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std

        assert np.isfinite(cohens_d)

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect = 'small'
        elif abs(cohens_d) < 0.8:
            effect = 'medium'
        else:
            effect = 'large'

        assert effect in ['small', 'medium', 'large']


@pytest.mark.comparison
class TestMultiDimensionalComparison:
    """Test multi-dimensional store comparisons."""

    def test_correlation_analysis(self, sample_data):
        """Test correlation between metrics across stores."""
        store_metrics = sample_data.groupby('Store').agg({
            'Sales': 'sum',
            'Customers': 'sum',
            'Promo': 'mean'
        })

        correlation = store_metrics.corr()

        assert correlation.shape == (3, 3)
        assert all(correlation.iloc[i, i] == pytest.approx(1.0) for i in range(3))
        assert all(abs(correlation.values.flatten()) <= 1.0)

    def test_principal_component_analysis(self, sample_data):
        """Test PCA for dimensionality reduction."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        store_metrics = sample_data.groupby('Store').agg({
            'Sales': ['sum', 'mean', 'std'],
            'Customers': ['sum', 'mean', 'std'],
            'Promo': 'mean'
        })

        # Flatten multi-level columns
        store_metrics.columns = ['_'.join(col).strip() for col in store_metrics.columns]

        # Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(store_metrics)

        # PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)

        assert components.shape == (len(store_metrics), 2)
        assert 0 <= sum(pca.explained_variance_ratio_) <= 1

    def test_clustering_analysis(self, sample_data):
        """Test clustering stores by performance."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        store_metrics = sample_data.groupby('Store').agg({
            'Sales': 'sum',
            'Customers': 'sum',
            'Promo': 'mean'
        })

        # Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(store_metrics)

        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled)

        assert len(clusters) == len(store_metrics)
        assert len(np.unique(clusters)) <= 3
        assert all(0 <= c < 3 for c in clusters)


@pytest.mark.comparison
class TestAccuracyValidation:
    """Test accuracy of comparison calculations."""

    def test_sum_consistency(self, sample_data):
        """Test that aggregated sums match totals."""
        total_sales = sample_data['Sales'].sum()
        store_sales_sum = sample_data.groupby('Store')['Sales'].sum().sum()

        assert total_sales == pytest.approx(store_sales_sum, rel=1e-9)

    def test_weighted_average_accuracy(self, sample_data):
        """Test weighted average calculation."""
        # Calculate weighted average sales per customer
        total_sales = sample_data['Sales'].sum()
        total_customers = sample_data['Customers'].sum()
        overall_avg = total_sales / total_customers

        # Calculate from store averages weighted by customers
        store_metrics = sample_data.groupby('Store').agg({
            'Sales': 'sum',
            'Customers': 'sum'
        })

        store_metrics['AvgPerCustomer'] = store_metrics['Sales'] / store_metrics['Customers']
        weighted_avg = (
            (store_metrics['AvgPerCustomer'] * store_metrics['Customers']).sum() /
            store_metrics['Customers'].sum()
        )

        assert overall_avg == pytest.approx(weighted_avg, rel=1e-9)

    def test_percentage_calculation_accuracy(self, sample_data):
        """Test percentage calculations sum to 100%."""
        store_sales = sample_data.groupby('Store')['Sales'].sum()
        total_sales = store_sales.sum()

        percentages = (store_sales / total_sales * 100)

        assert percentages.sum() == pytest.approx(100.0, rel=1e-9)

    def test_ranking_consistency(self, sample_data):
        """Test ranking consistency."""
        store_sales = sample_data.groupby('Store')['Sales'].sum()

        # Different ranking methods should be consistent for unique values
        rank_min = store_sales.rank(method='min')
        rank_max = store_sales.rank(method='max')

        # For unique values, min and max should be equal
        unique_sales = store_sales.drop_duplicates()
        if len(unique_sales) == len(store_sales):
            pd.testing.assert_series_equal(rank_min, rank_max)


@pytest.mark.comparison
class TestComparativeVisualizationData:
    """Test data preparation for comparative visualizations."""

    def test_side_by_side_comparison_data(self, sample_data):
        """Test data for side-by-side comparison charts."""
        comparison = sample_data.groupby('Store').agg({
            'Sales': 'sum',
            'Customers': 'sum'
        }).reset_index()

        # Reshape for plotting
        melted = comparison.melt(
            id_vars='Store',
            value_vars=['Sales', 'Customers'],
            var_name='Metric',
            value_name='Value'
        )

        assert len(melted) == len(comparison) * 2
        assert set(melted['Metric'].unique()) == {'Sales', 'Customers'}

    def test_normalized_comparison_data(self, sample_data):
        """Test normalized data for fair comparison."""
        store_sales = sample_data.groupby('Store')['Sales'].sum()

        # Min-max normalization
        normalized = (store_sales - store_sales.min()) / (store_sales.max() - store_sales.min())

        assert all(normalized >= 0)
        assert all(normalized <= 1)
        assert normalized.min() == 0
        assert normalized.max() == 1

    def test_sparkline_data(self, sample_data):
        """Test data for sparkline visualizations."""
        # Get daily sales for each store
        sparklines = {}

        for store in sample_data['Store'].unique():
            store_data = sample_data[sample_data['Store'] == store]
            daily_sales = store_data.groupby('Date')['Sales'].sum().values[:30]  # Last 30 days
            sparklines[store] = daily_sales.tolist()

        assert len(sparklines) == sample_data['Store'].nunique()
        assert all(len(v) > 0 for v in sparklines.values())
