"""
Store Comparison and Benchmarking System

This module provides comprehensive cross-store comparison capabilities including:
- Store similarity matching based on multiple dimensions
- Benchmark store identification
- Performance gap analysis with statistical significance
- Best practice identification from top performers
- Product mix comparison
- What-if scenario analysis
- Store clustering and segmentation

Author: PyCaret Development Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy import stats
from scipy.spatial.distance import cdist
import warnings


@dataclass
class StoreProfile:
    """Store profile for similarity matching"""
    store_id: str
    size_sqm: float
    location_type: str
    customer_segment: str
    avg_daily_sales: float
    avg_transaction_value: float
    avg_customer_count: float
    product_categories: List[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from store comparison"""
    target_store: str
    similar_stores: List[Tuple[str, float]]  # (store_id, similarity_score)
    benchmark_store: str
    performance_gap: Dict[str, float]
    statistical_significance: Dict[str, Dict[str, float]]
    best_practices: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class ClusteringResult:
    """Results from store clustering"""
    cluster_assignments: Dict[str, int]
    cluster_profiles: Dict[int, Dict[str, Any]]
    silhouette_score: float
    inertia: Optional[float] = None


class StoreComparison:
    """
    Comprehensive store comparison and benchmarking system

    Features:
    - Multi-dimensional similarity matching
    - Statistical performance analysis
    - Clustering and segmentation
    - What-if scenario modeling
    """

    def __init__(
        self,
        similarity_weights: Optional[Dict[str, float]] = None,
        min_sample_size: int = 30,
        significance_level: float = 0.05
    ):
        """
        Initialize store comparison system

        Parameters
        ----------
        similarity_weights : dict, optional
            Weights for different similarity dimensions
            Keys: 'size', 'location', 'customer_segment', 'performance'
        min_sample_size : int, default=30
            Minimum sample size for statistical tests
        significance_level : float, default=0.05
            P-value threshold for statistical significance
        """
        self.similarity_weights = similarity_weights or {
            'size': 0.3,
            'location': 0.2,
            'customer_segment': 0.2,
            'performance': 0.3
        }
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

        self.scaler = StandardScaler()
        self.store_profiles: Dict[str, StoreProfile] = {}
        self.feature_matrix: Optional[np.ndarray] = None
        self.store_ids: List[str] = []

    def add_store_profile(self, profile: StoreProfile) -> None:
        """Add a store profile to the comparison database"""
        self.store_profiles[profile.store_id] = profile

    def add_store_profiles_from_dataframe(
        self,
        df: pd.DataFrame,
        store_id_col: str = 'store_id',
        size_col: str = 'size_sqm',
        location_col: str = 'location_type',
        customer_segment_col: str = 'customer_segment'
    ) -> None:
        """
        Bulk add store profiles from a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing store information
        store_id_col : str
            Column name for store ID
        size_col : str
            Column name for store size in square meters
        location_col : str
            Column name for location type
        customer_segment_col : str
            Column name for customer segment
        """
        for _, row in df.iterrows():
            profile = StoreProfile(
                store_id=str(row[store_id_col]),
                size_sqm=float(row[size_col]),
                location_type=str(row[location_col]),
                customer_segment=str(row[customer_segment_col]),
                avg_daily_sales=float(row.get('avg_daily_sales', 0)),
                avg_transaction_value=float(row.get('avg_transaction_value', 0)),
                avg_customer_count=float(row.get('avg_customer_count', 0)),
                product_categories=row.get('product_categories', []),
                latitude=row.get('latitude'),
                longitude=row.get('longitude'),
                metadata={k: v for k, v in row.items()
                         if k not in [store_id_col, size_col, location_col, customer_segment_col]}
            )
            self.add_store_profile(profile)

    def _build_feature_matrix(self) -> np.ndarray:
        """Build feature matrix for similarity calculations"""
        if not self.store_profiles:
            raise ValueError("No store profiles available. Add profiles first.")

        self.store_ids = list(self.store_profiles.keys())
        features = []

        for store_id in self.store_ids:
            profile = self.store_profiles[store_id]

            # Numerical features
            feature_vector = [
                profile.size_sqm,
                profile.avg_daily_sales,
                profile.avg_transaction_value,
                profile.avg_customer_count
            ]

            # Add location coordinates if available
            if profile.latitude is not None and profile.longitude is not None:
                feature_vector.extend([profile.latitude, profile.longitude])

            features.append(feature_vector)

        # Standardize features
        self.feature_matrix = self.scaler.fit_transform(np.array(features))
        return self.feature_matrix

    def find_similar_stores(
        self,
        target_store_id: str,
        n_similar: int = 5,
        method: str = 'cosine',
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find stores similar to target store

        Parameters
        ----------
        target_store_id : str
            ID of target store
        n_similar : int, default=5
            Number of similar stores to return
        method : str, default='cosine'
            Similarity method: 'cosine', 'euclidean', 'manhattan'
        filter_criteria : dict, optional
            Criteria to filter candidate stores
            Example: {'location_type': 'urban', 'size_range': (100, 500)}

        Returns
        -------
        list of tuples
            List of (store_id, similarity_score) sorted by similarity
        """
        if self.feature_matrix is None:
            self._build_feature_matrix()

        if target_store_id not in self.store_ids:
            raise ValueError(f"Store {target_store_id} not found in profiles")

        target_idx = self.store_ids.index(target_store_id)
        target_vector = self.feature_matrix[target_idx].reshape(1, -1)

        # Apply filters
        candidate_indices = list(range(len(self.store_ids)))
        if filter_criteria:
            candidate_indices = self._apply_filters(filter_criteria, exclude_idx=target_idx)
        else:
            candidate_indices.remove(target_idx)

        if not candidate_indices:
            return []

        candidate_features = self.feature_matrix[candidate_indices]

        # Calculate similarity
        if method == 'cosine':
            similarities = cosine_similarity(target_vector, candidate_features)[0]
        elif method == 'euclidean':
            distances = euclidean_distances(target_vector, candidate_features)[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        elif method == 'manhattan':
            distances = cdist(target_vector, candidate_features, metric='cityblock')[0]
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][:n_similar]

        results = [
            (self.store_ids[candidate_indices[idx]], float(similarities[idx]))
            for idx in sorted_indices
        ]

        return results

    def _apply_filters(
        self,
        filter_criteria: Dict[str, Any],
        exclude_idx: Optional[int] = None
    ) -> List[int]:
        """Apply filtering criteria to store selection"""
        indices = []

        for idx, store_id in enumerate(self.store_ids):
            if exclude_idx is not None and idx == exclude_idx:
                continue

            profile = self.store_profiles[store_id]
            match = True

            # Check each filter criterion
            if 'location_type' in filter_criteria:
                if profile.location_type != filter_criteria['location_type']:
                    match = False

            if 'customer_segment' in filter_criteria:
                if profile.customer_segment != filter_criteria['customer_segment']:
                    match = False

            if 'size_range' in filter_criteria:
                min_size, max_size = filter_criteria['size_range']
                if not (min_size <= profile.size_sqm <= max_size):
                    match = False

            if 'sales_range' in filter_criteria:
                min_sales, max_sales = filter_criteria['sales_range']
                if not (min_sales <= profile.avg_daily_sales <= max_sales):
                    match = False

            if match:
                indices.append(idx)

        return indices

    def identify_benchmark_store(
        self,
        target_store_id: str,
        metric: str = 'avg_daily_sales',
        similar_stores: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Identify benchmark (best performing) store among similar stores

        Parameters
        ----------
        target_store_id : str
            ID of target store
        metric : str, default='avg_daily_sales'
            Performance metric to use for benchmarking
        similar_stores : list, optional
            List of similar store IDs. If None, will find similar stores first

        Returns
        -------
        tuple
            (benchmark_store_id, performance_value)
        """
        if similar_stores is None:
            similar_results = self.find_similar_stores(target_store_id, n_similar=10)
            similar_stores = [store_id for store_id, _ in similar_results]

        if not similar_stores:
            raise ValueError("No similar stores found for benchmarking")

        # Find best performer
        best_store = None
        best_value = -np.inf

        for store_id in similar_stores:
            profile = self.store_profiles[store_id]
            value = getattr(profile, metric, None)

            if value is not None and value > best_value:
                best_value = value
                best_store = store_id

        if best_store is None:
            raise ValueError(f"Could not find benchmark store using metric: {metric}")

        return best_store, float(best_value)

    def analyze_performance_gap(
        self,
        target_store_id: str,
        benchmark_store_id: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze performance gap between target and benchmark store

        Parameters
        ----------
        target_store_id : str
            ID of target store
        benchmark_store_id : str
            ID of benchmark store
        metrics : list, optional
            List of metrics to compare. If None, uses default metrics

        Returns
        -------
        dict
            Dictionary of metric: gap_percentage
        """
        if metrics is None:
            metrics = ['avg_daily_sales', 'avg_transaction_value', 'avg_customer_count']

        target = self.store_profiles[target_store_id]
        benchmark = self.store_profiles[benchmark_store_id]

        gaps = {}
        for metric in metrics:
            target_value = getattr(target, metric, None)
            benchmark_value = getattr(benchmark, metric, None)

            if target_value is not None and benchmark_value is not None and benchmark_value != 0:
                gap_pct = ((benchmark_value - target_value) / benchmark_value) * 100
                gaps[metric] = float(gap_pct)

        return gaps

    def test_statistical_significance(
        self,
        store_a_data: pd.Series,
        store_b_data: pd.Series,
        test_type: str = 'auto'
    ) -> Dict[str, float]:
        """
        Test statistical significance between two stores' performance

        Parameters
        ----------
        store_a_data : pd.Series
            Performance data for store A
        store_b_data : pd.Series
            Performance data for store B
        test_type : str, default='auto'
            Type of test: 'auto', 't-test', 'mann-whitney', 'welch'

        Returns
        -------
        dict
            Dictionary containing:
            - statistic: Test statistic value
            - p_value: P-value
            - significant: Boolean indicating significance
            - test_used: Name of test performed
        """
        # Remove NaN values
        a_clean = store_a_data.dropna()
        b_clean = store_b_data.dropna()

        if len(a_clean) < self.min_sample_size or len(b_clean) < self.min_sample_size:
            warnings.warn(f"Sample size below minimum ({self.min_sample_size}). Results may be unreliable.")

        # Auto-select test
        if test_type == 'auto':
            # Check normality using Shapiro-Wilk test
            _, p_norm_a = stats.shapiro(a_clean) if len(a_clean) <= 5000 else (0, 0.05)
            _, p_norm_b = stats.shapiro(b_clean) if len(b_clean) <= 5000 else (0, 0.05)

            # Check variance homogeneity using Levene's test
            _, p_var = stats.levene(a_clean, b_clean)

            if p_norm_a > 0.05 and p_norm_b > 0.05:
                if p_var > 0.05:
                    test_type = 't-test'
                else:
                    test_type = 'welch'
            else:
                test_type = 'mann-whitney'

        # Perform test
        if test_type == 't-test':
            statistic, p_value = stats.ttest_ind(a_clean, b_clean)
            test_name = "Independent t-test"
        elif test_type == 'welch':
            statistic, p_value = stats.ttest_ind(a_clean, b_clean, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(a_clean, b_clean, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'test_used': test_name,
            'effect_size': float(self._calculate_effect_size(a_clean, b_clean))
        }

    def _calculate_effect_size(self, a: pd.Series, b: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        mean_diff = a.mean() - b.mean()
        pooled_std = np.sqrt(((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2) / (len(a) + len(b) - 2))

        if pooled_std == 0:
            return 0.0

        return mean_diff / pooled_std

    def identify_best_practices(
        self,
        target_store_id: str,
        benchmark_stores: List[str],
        sales_data: pd.DataFrame,
        min_improvement: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Identify best practices from top-performing stores

        Parameters
        ----------
        target_store_id : str
            ID of target store
        benchmark_stores : list
            List of benchmark store IDs
        sales_data : pd.DataFrame
            Sales data with columns: store_id, date, sales, product_category, etc.
        min_improvement : float, default=10.0
            Minimum percentage improvement to consider as best practice

        Returns
        -------
        list of dict
            List of identified best practices with evidence
        """
        best_practices = []
        target_profile = self.store_profiles[target_store_id]

        # Analyze product mix
        target_mix = self._get_product_mix(target_store_id, sales_data)

        for benchmark_id in benchmark_stores:
            benchmark_profile = self.store_profiles[benchmark_id]
            benchmark_mix = self._get_product_mix(benchmark_id, sales_data)

            # Compare product categories
            for category in benchmark_mix.index:
                if category not in target_mix.index:
                    continue

                benchmark_sales = benchmark_mix.loc[category, 'total_sales']
                target_sales = target_mix.loc[category, 'total_sales']

                if benchmark_sales > 0 and target_sales > 0:
                    improvement_pct = ((benchmark_sales - target_sales) / target_sales) * 100

                    if improvement_pct >= min_improvement:
                        best_practices.append({
                            'practice_type': 'product_mix',
                            'category': category,
                            'benchmark_store': benchmark_id,
                            'improvement_potential': improvement_pct,
                            'benchmark_sales': benchmark_sales,
                            'current_sales': target_sales,
                            'recommendation': f"Increase focus on {category} category"
                        })

        # Analyze sales patterns
        target_patterns = self._analyze_sales_patterns(target_store_id, sales_data)

        for benchmark_id in benchmark_stores:
            benchmark_patterns = self._analyze_sales_patterns(benchmark_id, sales_data)

            # Compare peak hours
            if 'peak_hours' in benchmark_patterns and 'peak_hours' in target_patterns:
                if benchmark_patterns['peak_hour_sales'] > target_patterns['peak_hour_sales']:
                    improvement_pct = ((benchmark_patterns['peak_hour_sales'] -
                                      target_patterns['peak_hour_sales']) /
                                     target_patterns['peak_hour_sales']) * 100

                    if improvement_pct >= min_improvement:
                        best_practices.append({
                            'practice_type': 'operational',
                            'aspect': 'peak_hour_optimization',
                            'benchmark_store': benchmark_id,
                            'improvement_potential': improvement_pct,
                            'recommendation': f"Optimize staffing and inventory for peak hours: {benchmark_patterns['peak_hours']}"
                        })

        return sorted(best_practices, key=lambda x: x['improvement_potential'], reverse=True)

    def _get_product_mix(self, store_id: str, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Get product mix analysis for a store"""
        store_data = sales_data[sales_data['store_id'] == store_id]

        if 'product_category' in store_data.columns:
            mix = store_data.groupby('product_category').agg({
                'sales': ['sum', 'mean', 'count']
            })
            mix.columns = ['total_sales', 'avg_sales', 'transaction_count']
            return mix

        return pd.DataFrame()

    def _analyze_sales_patterns(self, store_id: str, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sales patterns for a store"""
        store_data = sales_data[sales_data['store_id'] == store_id].copy()
        patterns = {}

        if 'date' in store_data.columns:
            store_data['date'] = pd.to_datetime(store_data['date'])
            store_data['hour'] = store_data['date'].dt.hour
            store_data['day_of_week'] = store_data['date'].dt.dayofweek

            # Peak hours
            hourly_sales = store_data.groupby('hour')['sales'].sum()
            patterns['peak_hours'] = hourly_sales.nlargest(3).index.tolist()
            patterns['peak_hour_sales'] = hourly_sales.max()

            # Best days
            daily_sales = store_data.groupby('day_of_week')['sales'].mean()
            patterns['best_days'] = daily_sales.nlargest(2).index.tolist()

        return patterns

    def compare_product_mix(
        self,
        store_a_id: str,
        store_b_id: str,
        sales_data: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare product mix between two stores

        Parameters
        ----------
        store_a_id : str
            First store ID
        store_b_id : str
            Second store ID
        sales_data : pd.DataFrame
            Sales data with product information
        top_n : int, default=10
            Number of top products to compare

        Returns
        -------
        pd.DataFrame
            Comparison table with products and sales from both stores
        """
        mix_a = self._get_product_mix(store_a_id, sales_data)
        mix_b = self._get_product_mix(store_b_id, sales_data)

        # Merge and compare
        comparison = pd.DataFrame({
            f'{store_a_id}_sales': mix_a['total_sales'],
            f'{store_b_id}_sales': mix_b['total_sales']
        })

        comparison['difference'] = comparison[f'{store_b_id}_sales'] - comparison[f'{store_a_id}_sales']
        comparison['pct_difference'] = (comparison['difference'] / comparison[f'{store_a_id}_sales']) * 100

        return comparison.nlargest(top_n, 'pct_difference')

    def what_if_scenario(
        self,
        target_store_id: str,
        scenario_params: Dict[str, Any],
        model: Any,
        baseline_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Run what-if scenario analysis

        Parameters
        ----------
        target_store_id : str
            ID of target store
        scenario_params : dict
            Scenario parameters to modify
            Example: {'weather': 'sunny', 'day_of_week': 'saturday', 'promotion': True}
        model : object
            Trained prediction model with predict method
        baseline_data : pd.DataFrame
            Baseline data for the store

        Returns
        -------
        dict
            Predicted outcomes under scenario
        """
        # Create scenario data
        scenario_data = baseline_data[baseline_data['store_id'] == target_store_id].copy()

        # Apply scenario modifications
        for param, value in scenario_params.items():
            if param in scenario_data.columns:
                scenario_data[param] = value

        # Make predictions
        predictions = model.predict(scenario_data)

        # Calculate aggregated metrics
        results = {
            'predicted_total_sales': float(predictions.sum()),
            'predicted_avg_sales': float(predictions.mean()),
            'predicted_max_sales': float(predictions.max()),
            'predicted_min_sales': float(predictions.min()),
            'scenario_params': scenario_params
        }

        # Compare with baseline
        baseline_predictions = model.predict(baseline_data[baseline_data['store_id'] == target_store_id])
        results['baseline_total_sales'] = float(baseline_predictions.sum())
        results['uplift_pct'] = ((results['predicted_total_sales'] - results['baseline_total_sales']) /
                                results['baseline_total_sales']) * 100

        return results

    def cluster_stores(
        self,
        n_clusters: Optional[int] = None,
        method: str = 'kmeans',
        features: Optional[List[str]] = None
    ) -> ClusteringResult:
        """
        Cluster stores based on characteristics

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (required for kmeans)
        method : str, default='kmeans'
            Clustering method: 'kmeans', 'dbscan', 'hierarchical'
        features : list, optional
            List of features to use for clustering

        Returns
        -------
        ClusteringResult
            Clustering results with assignments and profiles
        """
        if self.feature_matrix is None:
            self._build_feature_matrix()

        X = self.feature_matrix

        # Select clustering method
        if method == 'kmeans':
            if n_clusters is None:
                # Use elbow method to find optimal k
                n_clusters = self._find_optimal_k(X)

            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
            inertia = clusterer.inertia_

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(X)
            inertia = None

        elif method == 'hierarchical':
            if n_clusters is None:
                n_clusters = 5

            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X)
            inertia = None

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Create cluster assignments
        assignments = {
            store_id: int(labels[idx])
            for idx, store_id in enumerate(self.store_ids)
        }

        # Create cluster profiles
        profiles = self._create_cluster_profiles(assignments, labels)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette = float(silhouette_score(X, labels))

        return ClusteringResult(
            cluster_assignments=assignments,
            cluster_profiles=profiles,
            silhouette_score=silhouette,
            inertia=inertia
        )

    def _find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        k_range = range(2, min(max_k + 1, len(X)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Find elbow point (simplified)
        if len(inertias) < 2:
            return 3

        diffs = np.diff(inertias)
        elbow_idx = np.argmax(diffs) + 2  # +2 because range starts at 2 and diff reduces length by 1

        return elbow_idx

    def _create_cluster_profiles(
        self,
        assignments: Dict[str, int],
        labels: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """Create descriptive profiles for each cluster"""
        profiles = {}

        for cluster_id in np.unique(labels):
            cluster_stores = [
                store_id for store_id, label in assignments.items()
                if label == cluster_id
            ]

            # Aggregate statistics
            sizes = [self.store_profiles[s].size_sqm for s in cluster_stores]
            sales = [self.store_profiles[s].avg_daily_sales for s in cluster_stores]

            profiles[int(cluster_id)] = {
                'size': len(cluster_stores),
                'store_ids': cluster_stores,
                'avg_size_sqm': float(np.mean(sizes)),
                'avg_daily_sales': float(np.mean(sales)),
                'total_sales': float(np.sum(sales)),
                'characteristics': self._get_cluster_characteristics(cluster_stores)
            }

        return profiles

    def _get_cluster_characteristics(self, store_ids: List[str]) -> Dict[str, Any]:
        """Extract common characteristics of stores in a cluster"""
        profiles = [self.store_profiles[s] for s in store_ids]

        # Most common location type
        location_types = [p.location_type for p in profiles]
        most_common_location = max(set(location_types), key=location_types.count)

        # Most common customer segment
        customer_segments = [p.customer_segment for p in profiles]
        most_common_segment = max(set(customer_segments), key=customer_segments.count)

        return {
            'dominant_location_type': most_common_location,
            'dominant_customer_segment': most_common_segment,
            'location_diversity': len(set(location_types)) / len(location_types),
            'segment_diversity': len(set(customer_segments)) / len(customer_segments)
        }

    def segment_stores(
        self,
        segmentation_criteria: Dict[str, List[Any]]
    ) -> Dict[str, List[str]]:
        """
        Segment stores based on predefined criteria

        Parameters
        ----------
        segmentation_criteria : dict
            Criteria for segmentation
            Example: {
                'size': [(0, 100, 'small'), (100, 300, 'medium'), (300, 1000, 'large')],
                'location_type': ['urban', 'suburban', 'rural']
            }

        Returns
        -------
        dict
            Dictionary mapping segment names to store IDs
        """
        segments = {}

        for store_id, profile in self.store_profiles.items():
            segment_tags = []

            # Apply each criterion
            for criterion, values in segmentation_criteria.items():
                if criterion == 'size':
                    for min_val, max_val, tag in values:
                        if min_val <= profile.size_sqm < max_val:
                            segment_tags.append(tag)
                            break

                elif criterion == 'location_type':
                    if profile.location_type in values:
                        segment_tags.append(profile.location_type)

                elif criterion == 'customer_segment':
                    if profile.customer_segment in values:
                        segment_tags.append(profile.customer_segment)

            # Create segment key
            segment_key = '_'.join(segment_tags) if segment_tags else 'unclassified'

            if segment_key not in segments:
                segments[segment_key] = []
            segments[segment_key].append(store_id)

        return segments

    def generate_comparison_report(
        self,
        target_store_id: str,
        sales_data: pd.DataFrame,
        n_similar: int = 5
    ) -> ComparisonResult:
        """
        Generate comprehensive comparison report for a target store

        Parameters
        ----------
        target_store_id : str
            ID of target store
        sales_data : pd.DataFrame
            Historical sales data for analysis
        n_similar : int, default=5
            Number of similar stores to include

        Returns
        -------
        ComparisonResult
            Comprehensive comparison results
        """
        # Find similar stores
        similar_stores = self.find_similar_stores(target_store_id, n_similar=n_similar)
        similar_store_ids = [store_id for store_id, _ in similar_stores]

        # Identify benchmark
        benchmark_store, _ = self.identify_benchmark_store(
            target_store_id,
            similar_stores=similar_store_ids
        )

        # Analyze performance gap
        performance_gap = self.analyze_performance_gap(target_store_id, benchmark_store)

        # Statistical significance testing
        significance_results = {}
        target_sales = sales_data[sales_data['store_id'] == target_store_id]['sales']

        for store_id in [benchmark_store]:
            store_sales = sales_data[sales_data['store_id'] == store_id]['sales']
            if len(store_sales) >= self.min_sample_size:
                significance_results[store_id] = self.test_statistical_significance(
                    target_sales,
                    store_sales
                )

        # Identify best practices
        best_practices = self.identify_best_practices(
            target_store_id,
            similar_store_ids,
            sales_data
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            performance_gap,
            best_practices,
            significance_results
        )

        return ComparisonResult(
            target_store=target_store_id,
            similar_stores=similar_stores,
            benchmark_store=benchmark_store,
            performance_gap=performance_gap,
            statistical_significance=significance_results,
            best_practices=best_practices[:5],  # Top 5
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        performance_gap: Dict[str, float],
        best_practices: List[Dict[str, Any]],
        significance: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Performance gap recommendations
        for metric, gap_pct in performance_gap.items():
            if gap_pct > 10:
                recommendations.append(
                    f"Focus on improving {metric.replace('_', ' ')} - currently {gap_pct:.1f}% below benchmark"
                )

        # Best practice recommendations
        for practice in best_practices[:3]:
            if practice['practice_type'] == 'product_mix':
                recommendations.append(
                    f"Consider expanding {practice['category']} category "
                    f"(potential {practice['improvement_potential']:.1f}% improvement)"
                )
            elif practice['practice_type'] == 'operational':
                recommendations.append(practice['recommendation'])

        # Statistical significance insights
        for store_id, sig_result in significance.items():
            if sig_result['significant']:
                effect = abs(sig_result['effect_size'])
                if effect > 0.8:
                    size_desc = "large"
                elif effect > 0.5:
                    size_desc = "medium"
                else:
                    size_desc = "small"

                recommendations.append(
                    f"Performance difference with benchmark store is statistically significant "
                    f"(p={sig_result['p_value']:.4f}, {size_desc} effect size)"
                )

        return recommendations


# Convenience functions
def create_store_comparison_system(
    store_data: pd.DataFrame,
    **kwargs
) -> StoreComparison:
    """
    Create and initialize a store comparison system

    Parameters
    ----------
    store_data : pd.DataFrame
        DataFrame with store information
    **kwargs : dict
        Additional parameters for StoreComparison initialization

    Returns
    -------
    StoreComparison
        Initialized comparison system
    """
    system = StoreComparison(**kwargs)
    system.add_store_profiles_from_dataframe(store_data)
    return system


def quick_store_comparison(
    target_store: str,
    store_data: pd.DataFrame,
    sales_data: pd.DataFrame,
    n_similar: int = 5
) -> ComparisonResult:
    """
    Quick store comparison analysis

    Parameters
    ----------
    target_store : str
        Target store ID
    store_data : pd.DataFrame
        Store profile data
    sales_data : pd.DataFrame
        Historical sales data
    n_similar : int, default=5
        Number of similar stores to compare

    Returns
    -------
    ComparisonResult
        Comparison analysis results
    """
    system = create_store_comparison_system(store_data)
    return system.generate_comparison_report(target_store, sales_data, n_similar)
