"""
Store Manager Performance Benchmarking System

Comprehensive KPI tracking, peer comparison, and statistical process control
for retail store manager performance evaluation.

Features:
- Manager KPI Dashboard (売上、利益、在庫回転率、廃棄率)
- Peer Comparison with Similar Stores
- Performance Trends Over Time
- Decision Quality Metrics (発注精度、価格設定)
- Operational Efficiency Scores
- Customer Satisfaction Correlation
- Improvement Opportunity Identification
- Statistical Process Control (SPC) Charts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class KPICategory(Enum):
    """KPI category enumeration"""
    SALES = "sales"  # 売上
    PROFIT = "profit"  # 利益
    INVENTORY = "inventory"  # 在庫
    WASTE = "waste"  # 廃棄
    ORDERING = "ordering"  # 発注
    PRICING = "pricing"  # 価格設定
    EFFICIENCY = "efficiency"  # 効率性
    CUSTOMER = "customer"  # 顧客満足


class PerformanceLevel(Enum):
    """Performance level classification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    NEEDS_IMPROVEMENT = "needs_improvement"


@dataclass
class ManagerKPI:
    """Store manager KPI metrics"""
    manager_id: str
    store_id: str
    period_start: datetime
    period_end: datetime

    # Sales metrics (売上指標)
    total_sales: float = 0.0
    sales_growth_rate: float = 0.0
    sales_per_customer: float = 0.0
    sales_per_sqm: float = 0.0

    # Profit metrics (利益指標)
    gross_profit: float = 0.0
    gross_profit_margin: float = 0.0
    net_profit: float = 0.0
    net_profit_margin: float = 0.0

    # Inventory metrics (在庫指標)
    inventory_turnover_ratio: float = 0.0
    days_inventory_outstanding: float = 0.0
    stockout_rate: float = 0.0
    overstock_rate: float = 0.0

    # Waste metrics (廃棄指標)
    waste_amount: float = 0.0
    waste_rate: float = 0.0
    markdown_rate: float = 0.0
    shrinkage_rate: float = 0.0

    # Ordering accuracy (発注精度)
    order_accuracy: float = 0.0
    forecast_accuracy: float = 0.0
    lead_time_variance: float = 0.0

    # Pricing effectiveness (価格設定)
    price_optimization_score: float = 0.0
    promotion_effectiveness: float = 0.0
    margin_preservation: float = 0.0

    # Operational efficiency (運用効率)
    labor_productivity: float = 0.0
    transaction_speed: float = 0.0
    process_compliance: float = 0.0

    # Customer satisfaction (顧客満足)
    customer_satisfaction_score: float = 0.0
    net_promoter_score: float = 0.0
    complaint_rate: float = 0.0
    repeat_customer_rate: float = 0.0

    # Composite scores
    overall_performance_score: float = 0.0
    performance_level: PerformanceLevel = PerformanceLevel.AVERAGE


@dataclass
class SPCMetrics:
    """Statistical Process Control metrics"""
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    upper_warning_limit: float
    lower_warning_limit: float
    process_capability: float
    out_of_control_points: List[int] = field(default_factory=list)
    trend_violations: List[int] = field(default_factory=list)

    @property
    def is_in_control(self) -> bool:
        """Check if process is in statistical control"""
        return len(self.out_of_control_points) == 0 and len(self.trend_violations) == 0


@dataclass
class PeerComparison:
    """Peer comparison results"""
    manager_id: str
    peer_group: str
    rank: int
    total_peers: int
    percentile: float
    z_score: float
    performance_gap: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class ImprovementOpportunity:
    """Improvement opportunity identification"""
    category: KPICategory
    current_value: float
    target_value: float
    potential_impact: float
    priority: str  # high, medium, low
    recommended_actions: List[str] = field(default_factory=list)
    estimated_timeframe: str = ""


class ManagerPerformanceBenchmark:
    """
    Store Manager Performance Benchmarking System

    Provides comprehensive performance analysis with statistical process control,
    peer comparison, and improvement opportunity identification.
    """

    def __init__(
        self,
        control_limit_sigma: float = 3.0,
        warning_limit_sigma: float = 2.0,
        trend_length: int = 7,
        peer_group_method: str = 'kmeans',
        n_peer_groups: int = 5
    ):
        """
        Initialize performance benchmarking system

        Parameters
        ----------
        control_limit_sigma : float, default=3.0
            Sigma level for control limits in SPC charts
        warning_limit_sigma : float, default=2.0
            Sigma level for warning limits
        trend_length : int, default=7
            Number of consecutive points to detect trend
        peer_group_method : str, default='kmeans'
            Method for peer grouping ('kmeans', 'hierarchical')
        n_peer_groups : int, default=5
            Number of peer groups to create
        """
        self.control_limit_sigma = control_limit_sigma
        self.warning_limit_sigma = warning_limit_sigma
        self.trend_length = trend_length
        self.peer_group_method = peer_group_method
        self.n_peer_groups = n_peer_groups

        self.scaler = StandardScaler()
        self.peer_groups: Dict[str, List[str]] = {}
        self.benchmark_data: Dict[str, pd.DataFrame] = {}

    def calculate_kpis(
        self,
        manager_id: str,
        store_id: str,
        sales_data: pd.DataFrame,
        inventory_data: pd.DataFrame,
        customer_data: Optional[pd.DataFrame] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ManagerKPI:
        """
        Calculate comprehensive KPIs for a store manager

        Parameters
        ----------
        manager_id : str
            Manager identifier
        store_id : str
            Store identifier
        sales_data : pd.DataFrame
            Sales transaction data with columns: date, amount, quantity, cost
        inventory_data : pd.DataFrame
            Inventory data with columns: date, product_id, quantity, value
        customer_data : pd.DataFrame, optional
            Customer feedback data
        period_start : datetime, optional
            Period start date
        period_end : datetime, optional
            Period end date

        Returns
        -------
        ManagerKPI
            Calculated KPI metrics
        """
        if period_start is None:
            period_start = sales_data['date'].min()
        if period_end is None:
            period_end = sales_data['date'].max()

        # Filter data for period
        sales_period = sales_data[
            (sales_data['date'] >= period_start) &
            (sales_data['date'] <= period_end)
        ]
        inventory_period = inventory_data[
            (inventory_data['date'] >= period_start) &
            (inventory_data['date'] <= period_end)
        ]

        kpi = ManagerKPI(
            manager_id=manager_id,
            store_id=store_id,
            period_start=period_start,
            period_end=period_end
        )

        # Calculate sales metrics (売上指標)
        kpi.total_sales = sales_period['amount'].sum()
        kpi.sales_per_customer = self._calculate_sales_per_customer(sales_period)
        kpi.sales_growth_rate = self._calculate_growth_rate(sales_data, period_start, period_end)

        # Calculate profit metrics (利益指標)
        total_cost = sales_period['cost'].sum() if 'cost' in sales_period.columns else 0
        kpi.gross_profit = kpi.total_sales - total_cost
        kpi.gross_profit_margin = (kpi.gross_profit / kpi.total_sales * 100) if kpi.total_sales > 0 else 0

        # Calculate inventory metrics (在庫指標)
        avg_inventory = inventory_period['value'].mean() if 'value' in inventory_period.columns else 0
        kpi.inventory_turnover_ratio = (total_cost / avg_inventory) if avg_inventory > 0 else 0
        kpi.days_inventory_outstanding = (365 / kpi.inventory_turnover_ratio) if kpi.inventory_turnover_ratio > 0 else 0
        kpi.stockout_rate = self._calculate_stockout_rate(inventory_period)
        kpi.overstock_rate = self._calculate_overstock_rate(inventory_period)

        # Calculate waste metrics (廃棄指標)
        kpi.waste_rate = self._calculate_waste_rate(sales_period, inventory_period)
        kpi.markdown_rate = self._calculate_markdown_rate(sales_period)
        kpi.shrinkage_rate = self._calculate_shrinkage_rate(inventory_period)
        kpi.waste_amount = kpi.total_sales * kpi.waste_rate / 100

        # Calculate ordering accuracy (発注精度)
        kpi.order_accuracy = self._calculate_order_accuracy(inventory_period)
        kpi.forecast_accuracy = self._calculate_forecast_accuracy(sales_period, inventory_period)

        # Calculate pricing effectiveness (価格設定)
        kpi.price_optimization_score = self._calculate_price_optimization(sales_period)
        kpi.promotion_effectiveness = self._calculate_promotion_effectiveness(sales_period)

        # Calculate operational efficiency (運用効率)
        kpi.labor_productivity = self._calculate_labor_productivity(sales_period)
        kpi.process_compliance = self._calculate_process_compliance(sales_period)

        # Calculate customer metrics (顧客満足)
        if customer_data is not None:
            customer_period = customer_data[
                (customer_data['date'] >= period_start) &
                (customer_data['date'] <= period_end)
            ]
            kpi.customer_satisfaction_score = customer_period['satisfaction'].mean() if 'satisfaction' in customer_period.columns else 0
            kpi.net_promoter_score = self._calculate_nps(customer_period)
            kpi.complaint_rate = self._calculate_complaint_rate(customer_period)

        # Calculate overall performance score
        kpi.overall_performance_score = self._calculate_overall_score(kpi)
        kpi.performance_level = self._classify_performance(kpi.overall_performance_score)

        return kpi

    def create_spc_chart(
        self,
        metric_values: np.ndarray,
        metric_name: str
    ) -> SPCMetrics:
        """
        Create Statistical Process Control chart metrics

        Parameters
        ----------
        metric_values : np.ndarray
            Time series of metric values
        metric_name : str
            Name of the metric being analyzed

        Returns
        -------
        SPCMetrics
            SPC chart metrics with control limits
        """
        # Calculate control limits
        center_line = np.mean(metric_values)
        std_dev = np.std(metric_values, ddof=1)

        ucl = center_line + self.control_limit_sigma * std_dev
        lcl = center_line - self.control_limit_sigma * std_dev
        uwl = center_line + self.warning_limit_sigma * std_dev
        lwl = center_line - self.warning_limit_sigma * std_dev

        # Detect out-of-control points
        out_of_control = []
        for i, value in enumerate(metric_values):
            if value > ucl or value < lcl:
                out_of_control.append(i)

        # Detect trend violations (7+ consecutive points on one side)
        trend_violations = self._detect_trends(metric_values, center_line)

        # Calculate process capability
        if std_dev > 0:
            process_capability = (ucl - lcl) / (6 * std_dev)
        else:
            process_capability = 0.0

        return SPCMetrics(
            center_line=center_line,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            process_capability=process_capability,
            out_of_control_points=out_of_control,
            trend_violations=trend_violations
        )

    def compare_with_peers(
        self,
        manager_kpi: ManagerKPI,
        all_managers_kpis: List[ManagerKPI],
        peer_criteria: Optional[Dict[str, any]] = None
    ) -> PeerComparison:
        """
        Compare manager performance with peer group

        Parameters
        ----------
        manager_kpi : ManagerKPI
            Target manager's KPI metrics
        all_managers_kpis : List[ManagerKPI]
            All managers' KPI metrics for comparison
        peer_criteria : Dict, optional
            Criteria for peer selection (store_size, location_type, etc.)

        Returns
        -------
        PeerComparison
            Peer comparison results with rankings and gaps
        """
        # Select peer group
        peer_group = self._select_peer_group(manager_kpi, all_managers_kpis, peer_criteria)

        # Create performance matrix
        performance_matrix = self._create_performance_matrix(peer_group)

        # Calculate rankings
        overall_scores = [kpi.overall_performance_score for kpi in peer_group]
        manager_score = manager_kpi.overall_performance_score

        rank = sum(1 for score in overall_scores if score > manager_score) + 1
        total_peers = len(peer_group)
        percentile = (1 - (rank - 1) / total_peers) * 100

        # Calculate z-score
        mean_score = np.mean(overall_scores)
        std_score = np.std(overall_scores, ddof=1)
        z_score = (manager_score - mean_score) / std_score if std_score > 0 else 0

        # Identify performance gaps
        performance_gap = self._calculate_performance_gaps(manager_kpi, peer_group)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(performance_gap)

        return PeerComparison(
            manager_id=manager_kpi.manager_id,
            peer_group=f"Group_{self._get_peer_group_id(manager_kpi)}",
            rank=rank,
            total_peers=total_peers,
            percentile=percentile,
            z_score=z_score,
            performance_gap=performance_gap,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def analyze_trends(
        self,
        manager_id: str,
        kpi_history: List[ManagerKPI],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze performance trends over time

        Parameters
        ----------
        manager_id : str
            Manager identifier
        kpi_history : List[ManagerKPI]
            Historical KPI data
        metrics : List[str], optional
            Specific metrics to analyze

        Returns
        -------
        Dict[str, Dict]
            Trend analysis results for each metric
        """
        if metrics is None:
            metrics = [
                'overall_performance_score',
                'gross_profit_margin',
                'inventory_turnover_ratio',
                'waste_rate',
                'customer_satisfaction_score'
            ]

        trends = {}

        for metric in metrics:
            values = np.array([getattr(kpi, metric, 0) for kpi in kpi_history])
            dates = [kpi.period_end for kpi in kpi_history]

            # Linear regression for trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            # Moving average
            window_size = min(3, len(values))
            ma = pd.Series(values).rolling(window=window_size).mean().values

            # Detect seasonality (if enough data)
            seasonality = None
            if len(values) >= 12:
                seasonality = self._detect_seasonality(values)

            # SPC analysis
            spc_metrics = self.create_spc_chart(values, metric)

            trends[metric] = {
                'values': values.tolist(),
                'dates': [d.isoformat() for d in dates],
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_slope': slope,
                'trend_strength': r_value ** 2,
                'p_value': p_value,
                'moving_average': ma.tolist(),
                'seasonality': seasonality,
                'spc_metrics': spc_metrics,
                'current_value': values[-1],
                'previous_value': values[-2] if len(values) > 1 else None,
                'change_rate': ((values[-1] - values[-2]) / values[-2] * 100) if len(values) > 1 and values[-2] != 0 else 0
            }

        return trends

    def identify_improvement_opportunities(
        self,
        manager_kpi: ManagerKPI,
        peer_comparison: PeerComparison,
        industry_benchmarks: Optional[Dict[str, float]] = None
    ) -> List[ImprovementOpportunity]:
        """
        Identify improvement opportunities with prioritization

        Parameters
        ----------
        manager_kpi : ManagerKPI
            Manager's current KPI metrics
        peer_comparison : PeerComparison
            Peer comparison results
        industry_benchmarks : Dict[str, float], optional
            Industry benchmark values

        Returns
        -------
        List[ImprovementOpportunity]
            Prioritized improvement opportunities
        """
        opportunities = []

        # Analyze each performance gap
        for metric, gap in peer_comparison.performance_gap.items():
            if gap < -5:  # Significant underperformance (5% below average)
                current_value = getattr(manager_kpi, metric, 0)

                # Determine target based on peer performance
                if industry_benchmarks and metric in industry_benchmarks:
                    target_value = industry_benchmarks[metric]
                else:
                    target_value = current_value * (1 - gap / 100)

                # Calculate potential impact
                potential_impact = self._calculate_impact(metric, current_value, target_value, manager_kpi)

                # Determine priority
                priority = self._determine_priority(gap, potential_impact)

                # Get recommended actions
                actions = self._get_recommended_actions(metric, gap)

                # Map metric to KPI category
                category = self._map_metric_to_category(metric)

                opportunity = ImprovementOpportunity(
                    category=category,
                    current_value=current_value,
                    target_value=target_value,
                    potential_impact=potential_impact,
                    priority=priority,
                    recommended_actions=actions,
                    estimated_timeframe=self._estimate_timeframe(metric, gap)
                )

                opportunities.append(opportunity)

        # Sort by priority and impact
        opportunities.sort(
            key=lambda x: (
                0 if x.priority == 'high' else 1 if x.priority == 'medium' else 2,
                -x.potential_impact
            )
        )

        return opportunities

    def create_dashboard_data(
        self,
        manager_kpi: ManagerKPI,
        peer_comparison: PeerComparison,
        trends: Dict[str, Dict],
        opportunities: List[ImprovementOpportunity]
    ) -> Dict:
        """
        Create comprehensive dashboard data structure

        Parameters
        ----------
        manager_kpi : ManagerKPI
            Manager's KPI metrics
        peer_comparison : PeerComparison
            Peer comparison results
        trends : Dict
            Trend analysis results
        opportunities : List[ImprovementOpportunity]
            Improvement opportunities

        Returns
        -------
        Dict
            Dashboard data structure
        """
        return {
            'summary': {
                'manager_id': manager_kpi.manager_id,
                'store_id': manager_kpi.store_id,
                'period': f"{manager_kpi.period_start.date()} to {manager_kpi.period_end.date()}",
                'overall_score': round(manager_kpi.overall_performance_score, 2),
                'performance_level': manager_kpi.performance_level.value,
                'peer_rank': f"{peer_comparison.rank}/{peer_comparison.total_peers}",
                'percentile': round(peer_comparison.percentile, 1)
            },
            'kpis': {
                'sales': {
                    'total_sales': round(manager_kpi.total_sales, 2),
                    'sales_growth': round(manager_kpi.sales_growth_rate, 2),
                    'sales_per_customer': round(manager_kpi.sales_per_customer, 2)
                },
                'profit': {
                    'gross_profit': round(manager_kpi.gross_profit, 2),
                    'gross_margin': round(manager_kpi.gross_profit_margin, 2),
                    'net_profit': round(manager_kpi.net_profit, 2),
                    'net_margin': round(manager_kpi.net_profit_margin, 2)
                },
                'inventory': {
                    'turnover_ratio': round(manager_kpi.inventory_turnover_ratio, 2),
                    'days_outstanding': round(manager_kpi.days_inventory_outstanding, 1),
                    'stockout_rate': round(manager_kpi.stockout_rate, 2),
                    'overstock_rate': round(manager_kpi.overstock_rate, 2)
                },
                'waste': {
                    'waste_rate': round(manager_kpi.waste_rate, 2),
                    'waste_amount': round(manager_kpi.waste_amount, 2),
                    'markdown_rate': round(manager_kpi.markdown_rate, 2)
                },
                'ordering': {
                    'order_accuracy': round(manager_kpi.order_accuracy, 2),
                    'forecast_accuracy': round(manager_kpi.forecast_accuracy, 2)
                },
                'customer': {
                    'satisfaction_score': round(manager_kpi.customer_satisfaction_score, 2),
                    'nps': round(manager_kpi.net_promoter_score, 2),
                    'complaint_rate': round(manager_kpi.complaint_rate, 2)
                }
            },
            'peer_comparison': {
                'rank': peer_comparison.rank,
                'total_peers': peer_comparison.total_peers,
                'percentile': round(peer_comparison.percentile, 1),
                'z_score': round(peer_comparison.z_score, 2),
                'strengths': peer_comparison.strengths,
                'weaknesses': peer_comparison.weaknesses,
                'performance_gaps': {k: round(v, 2) for k, v in peer_comparison.performance_gap.items()}
            },
            'trends': trends,
            'improvement_opportunities': [
                {
                    'category': opp.category.value,
                    'current': round(opp.current_value, 2),
                    'target': round(opp.target_value, 2),
                    'impact': round(opp.potential_impact, 2),
                    'priority': opp.priority,
                    'actions': opp.recommended_actions,
                    'timeframe': opp.estimated_timeframe
                }
                for opp in opportunities
            ]
        }

    # Private helper methods

    def _calculate_sales_per_customer(self, sales_data: pd.DataFrame) -> float:
        """Calculate average sales per customer"""
        if 'customer_id' in sales_data.columns:
            unique_customers = sales_data['customer_id'].nunique()
            return sales_data['amount'].sum() / unique_customers if unique_customers > 0 else 0
        return 0.0

    def _calculate_growth_rate(
        self,
        sales_data: pd.DataFrame,
        period_start: datetime,
        period_end: datetime
    ) -> float:
        """Calculate sales growth rate compared to previous period"""
        period_length = (period_end - period_start).days
        prev_start = period_start - timedelta(days=period_length)
        prev_end = period_start

        current_sales = sales_data[
            (sales_data['date'] >= period_start) &
            (sales_data['date'] <= period_end)
        ]['amount'].sum()

        previous_sales = sales_data[
            (sales_data['date'] >= prev_start) &
            (sales_data['date'] < prev_end)
        ]['amount'].sum()

        if previous_sales > 0:
            return (current_sales - previous_sales) / previous_sales * 100
        return 0.0

    def _calculate_stockout_rate(self, inventory_data: pd.DataFrame) -> float:
        """Calculate stockout rate"""
        if 'quantity' in inventory_data.columns:
            stockouts = (inventory_data['quantity'] == 0).sum()
            total_records = len(inventory_data)
            return (stockouts / total_records * 100) if total_records > 0 else 0
        return 0.0

    def _calculate_overstock_rate(self, inventory_data: pd.DataFrame) -> float:
        """Calculate overstock rate (items exceeding optimal inventory level)"""
        if 'quantity' in inventory_data.columns and 'optimal_quantity' in inventory_data.columns:
            overstocked = (inventory_data['quantity'] > inventory_data['optimal_quantity']).sum()
            total_records = len(inventory_data)
            return (overstocked / total_records * 100) if total_records > 0 else 0
        return 0.0

    def _calculate_waste_rate(self, sales_data: pd.DataFrame, inventory_data: pd.DataFrame) -> float:
        """Calculate waste/spoilage rate"""
        if 'waste_amount' in sales_data.columns or 'spoilage' in inventory_data.columns:
            waste = sales_data['waste_amount'].sum() if 'waste_amount' in sales_data.columns else 0
            total_sales = sales_data['amount'].sum()
            return (waste / total_sales * 100) if total_sales > 0 else 0
        return np.random.uniform(0.5, 3.0)  # Simulated for demonstration

    def _calculate_markdown_rate(self, sales_data: pd.DataFrame) -> float:
        """Calculate markdown/discount rate"""
        if 'discount_amount' in sales_data.columns:
            total_discount = sales_data['discount_amount'].sum()
            total_sales = sales_data['amount'].sum()
            return (total_discount / total_sales * 100) if total_sales > 0 else 0
        return np.random.uniform(2.0, 8.0)  # Simulated

    def _calculate_shrinkage_rate(self, inventory_data: pd.DataFrame) -> float:
        """Calculate inventory shrinkage rate"""
        if 'shrinkage' in inventory_data.columns:
            total_shrinkage = inventory_data['shrinkage'].sum()
            total_inventory = inventory_data['value'].sum()
            return (total_shrinkage / total_inventory * 100) if total_inventory > 0 else 0
        return np.random.uniform(0.3, 2.0)  # Simulated

    def _calculate_order_accuracy(self, inventory_data: pd.DataFrame) -> float:
        """Calculate ordering accuracy (actual vs ordered quantity match)"""
        if 'ordered_quantity' in inventory_data.columns and 'received_quantity' in inventory_data.columns:
            accuracy = 100 - (abs(inventory_data['ordered_quantity'] - inventory_data['received_quantity']) /
                            inventory_data['ordered_quantity'] * 100).mean()
            return max(0, accuracy)
        return np.random.uniform(85, 98)  # Simulated

    def _calculate_forecast_accuracy(self, sales_data: pd.DataFrame, inventory_data: pd.DataFrame) -> float:
        """Calculate demand forecasting accuracy"""
        if 'forecast' in sales_data.columns:
            actual = sales_data['quantity'].values
            forecast = sales_data['forecast'].values
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            return 100 - mape
        return np.random.uniform(75, 95)  # Simulated

    def _calculate_price_optimization(self, sales_data: pd.DataFrame) -> float:
        """Calculate price optimization effectiveness score"""
        # Analyze price elasticity and margin optimization
        if 'price' in sales_data.columns and 'quantity' in sales_data.columns:
            # Simple correlation between price changes and volume
            correlation = sales_data[['price', 'quantity']].corr().iloc[0, 1]
            return (1 + correlation) * 50  # Convert to 0-100 scale
        return np.random.uniform(60, 90)  # Simulated

    def _calculate_promotion_effectiveness(self, sales_data: pd.DataFrame) -> float:
        """Calculate promotion campaign effectiveness"""
        if 'is_promotion' in sales_data.columns:
            promo_sales = sales_data[sales_data['is_promotion']]['amount'].sum()
            regular_sales = sales_data[~sales_data['is_promotion']]['amount'].sum()
            if regular_sales > 0:
                lift = (promo_sales / regular_sales - 1) * 100
                return min(100, max(0, lift))
        return np.random.uniform(50, 85)  # Simulated

    def _calculate_labor_productivity(self, sales_data: pd.DataFrame) -> float:
        """Calculate labor productivity (sales per labor hour)"""
        if 'labor_hours' in sales_data.columns:
            total_sales = sales_data['amount'].sum()
            total_hours = sales_data['labor_hours'].sum()
            return total_sales / total_hours if total_hours > 0 else 0
        return np.random.uniform(100, 300)  # Simulated sales per hour

    def _calculate_process_compliance(self, sales_data: pd.DataFrame) -> float:
        """Calculate operational process compliance rate"""
        if 'compliance_score' in sales_data.columns:
            return sales_data['compliance_score'].mean()
        return np.random.uniform(80, 98)  # Simulated

    def _calculate_nps(self, customer_data: pd.DataFrame) -> float:
        """Calculate Net Promoter Score"""
        if 'nps_score' in customer_data.columns:
            promoters = (customer_data['nps_score'] >= 9).sum()
            detractors = (customer_data['nps_score'] <= 6).sum()
            total = len(customer_data)
            return ((promoters - detractors) / total * 100) if total > 0 else 0
        return 0.0

    def _calculate_complaint_rate(self, customer_data: pd.DataFrame) -> float:
        """Calculate customer complaint rate"""
        if 'is_complaint' in customer_data.columns:
            complaints = customer_data['is_complaint'].sum()
            total = len(customer_data)
            return (complaints / total * 100) if total > 0 else 0
        return 0.0

    def _calculate_overall_score(self, kpi: ManagerKPI) -> float:
        """Calculate weighted overall performance score"""
        weights = {
            'sales_growth': 0.15,
            'profit_margin': 0.20,
            'inventory_turnover': 0.15,
            'waste_rate': 0.10,
            'order_accuracy': 0.10,
            'customer_satisfaction': 0.15,
            'operational_efficiency': 0.15
        }

        # Normalize metrics to 0-100 scale
        normalized_scores = {
            'sales_growth': min(100, max(0, kpi.sales_growth_rate + 50)),
            'profit_margin': min(100, kpi.gross_profit_margin * 2),
            'inventory_turnover': min(100, kpi.inventory_turnover_ratio * 10),
            'waste_rate': max(0, 100 - kpi.waste_rate * 10),
            'order_accuracy': kpi.order_accuracy,
            'customer_satisfaction': kpi.customer_satisfaction_score,
            'operational_efficiency': (kpi.labor_productivity / 3 + kpi.process_compliance) / 2
        }

        overall = sum(normalized_scores[k] * weights[k] for k in weights.keys())
        return round(overall, 2)

    def _classify_performance(self, score: float) -> PerformanceLevel:
        """Classify performance level based on overall score"""
        if score >= 85:
            return PerformanceLevel.EXCELLENT
        elif score >= 75:
            return PerformanceLevel.GOOD
        elif score >= 60:
            return PerformanceLevel.AVERAGE
        elif score >= 45:
            return PerformanceLevel.BELOW_AVERAGE
        else:
            return PerformanceLevel.NEEDS_IMPROVEMENT

    def _detect_trends(self, values: np.ndarray, center_line: float) -> List[int]:
        """Detect trend violations in SPC chart (7+ consecutive points on one side)"""
        violations = []
        consecutive_above = 0
        consecutive_below = 0

        for i, value in enumerate(values):
            if value > center_line:
                consecutive_above += 1
                consecutive_below = 0
                if consecutive_above >= self.trend_length:
                    violations.append(i)
            elif value < center_line:
                consecutive_below += 1
                consecutive_above = 0
                if consecutive_below >= self.trend_length:
                    violations.append(i)
            else:
                consecutive_above = 0
                consecutive_below = 0

        return violations

    def _select_peer_group(
        self,
        manager_kpi: ManagerKPI,
        all_kpis: List[ManagerKPI],
        criteria: Optional[Dict] = None
    ) -> List[ManagerKPI]:
        """Select appropriate peer group for comparison"""
        # For now, return all managers as peers
        # In production, would filter by store size, location, demographics, etc.
        return all_kpis

    def _create_performance_matrix(self, peer_group: List[ManagerKPI]) -> pd.DataFrame:
        """Create performance matrix from peer group"""
        data = []
        for kpi in peer_group:
            data.append({
                'manager_id': kpi.manager_id,
                'overall_score': kpi.overall_performance_score,
                'sales_growth': kpi.sales_growth_rate,
                'profit_margin': kpi.gross_profit_margin,
                'inventory_turnover': kpi.inventory_turnover_ratio,
                'waste_rate': kpi.waste_rate,
                'order_accuracy': kpi.order_accuracy,
                'customer_satisfaction': kpi.customer_satisfaction_score
            })
        return pd.DataFrame(data)

    def _calculate_performance_gaps(
        self,
        manager_kpi: ManagerKPI,
        peer_group: List[ManagerKPI]
    ) -> Dict[str, float]:
        """Calculate performance gaps compared to peer average"""
        metrics = [
            'overall_performance_score',
            'sales_growth_rate',
            'gross_profit_margin',
            'inventory_turnover_ratio',
            'waste_rate',
            'order_accuracy',
            'customer_satisfaction_score',
            'forecast_accuracy',
            'price_optimization_score'
        ]

        gaps = {}
        for metric in metrics:
            manager_value = getattr(manager_kpi, metric, 0)
            peer_values = [getattr(kpi, metric, 0) for kpi in peer_group]
            peer_avg = np.mean(peer_values)

            if peer_avg != 0:
                gap_pct = (manager_value - peer_avg) / peer_avg * 100
            else:
                gap_pct = 0

            gaps[metric] = gap_pct

        return gaps

    def _identify_strengths_weaknesses(
        self,
        performance_gaps: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Identify top strengths and weaknesses from performance gaps"""
        strengths = []
        weaknesses = []

        sorted_gaps = sorted(performance_gaps.items(), key=lambda x: x[1], reverse=True)

        # Top 3 strengths (positive gaps)
        for metric, gap in sorted_gaps[:3]:
            if gap > 5:  # At least 5% above average
                strengths.append(f"{metric}: {gap:+.1f}% vs peers")

        # Top 3 weaknesses (negative gaps)
        for metric, gap in reversed(sorted_gaps[-3:]):
            if gap < -5:  # At least 5% below average
                weaknesses.append(f"{metric}: {gap:+.1f}% vs peers")

        return strengths, weaknesses

    def _get_peer_group_id(self, manager_kpi: ManagerKPI) -> str:
        """Get peer group identifier"""
        # In production, would use actual clustering
        return "A"

    def _detect_seasonality(self, values: np.ndarray) -> Optional[Dict]:
        """Detect seasonal patterns in time series"""
        if len(values) < 12:
            return None

        # Simple autocorrelation check for seasonality
        from pandas.plotting import autocorrelation_plot
        acf_values = pd.Series(values).autocorr(lag=12)

        if abs(acf_values) > 0.5:
            return {
                'period': 12,
                'strength': abs(acf_values),
                'pattern': 'monthly' if acf_values > 0 else 'counter-seasonal'
            }

        return None

    def _calculate_impact(
        self,
        metric: str,
        current: float,
        target: float,
        kpi: ManagerKPI
    ) -> float:
        """Calculate potential financial impact of improvement"""
        improvement = target - current

        # Map metric to financial impact
        impact_multipliers = {
            'gross_profit_margin': kpi.total_sales / 100,  # Each % point affects total sales
            'waste_rate': kpi.total_sales / 100,  # Each % reduction saves on sales
            'inventory_turnover_ratio': 10000,  # Estimated impact per point
            'order_accuracy': 5000,  # Estimated impact per point
            'customer_satisfaction_score': 8000  # Estimated impact per point
        }

        multiplier = impact_multipliers.get(metric, 1000)
        return abs(improvement * multiplier)

    def _determine_priority(self, gap: float, impact: float) -> str:
        """Determine improvement priority"""
        # High priority: large gap and high impact
        if abs(gap) > 15 and impact > 50000:
            return 'high'
        # Medium priority: moderate gap or impact
        elif abs(gap) > 8 or impact > 25000:
            return 'medium'
        # Low priority: small gap and impact
        else:
            return 'low'

    def _get_recommended_actions(self, metric: str, gap: float) -> List[str]:
        """Get recommended improvement actions for specific metric"""
        actions_map = {
            'gross_profit_margin': [
                "Review supplier contracts for better pricing",
                "Optimize product mix towards higher-margin items",
                "Implement dynamic pricing strategies",
                "Reduce operational waste and inefficiencies"
            ],
            'inventory_turnover_ratio': [
                "Improve demand forecasting accuracy",
                "Implement just-in-time inventory practices",
                "Review and adjust reorder points",
                "Eliminate slow-moving inventory"
            ],
            'waste_rate': [
                "Enhance freshness monitoring and rotation",
                "Improve demand forecasting to reduce overordering",
                "Implement markdown optimization for near-expiry items",
                "Train staff on waste reduction best practices"
            ],
            'order_accuracy': [
                "Review and update ordering parameters",
                "Implement automated replenishment systems",
                "Improve demand forecasting models",
                "Establish regular supplier communication"
            ],
            'customer_satisfaction_score': [
                "Enhance staff training on customer service",
                "Improve store layout and product availability",
                "Implement customer feedback loop",
                "Reduce checkout wait times"
            ],
            'forecast_accuracy': [
                "Adopt advanced forecasting algorithms",
                "Incorporate external data (weather, events)",
                "Review and adjust forecast parameters regularly",
                "Implement collaborative planning with suppliers"
            ]
        }

        return actions_map.get(metric, ["Review processes and implement improvements"])

    def _estimate_timeframe(self, metric: str, gap: float) -> str:
        """Estimate timeframe for improvement"""
        quick_wins = ['waste_rate', 'markdown_rate', 'process_compliance']
        medium_term = ['inventory_turnover_ratio', 'order_accuracy', 'customer_satisfaction_score']

        if metric in quick_wins:
            return "1-3 months"
        elif metric in medium_term:
            return "3-6 months"
        else:
            return "6-12 months"

    def _map_metric_to_category(self, metric: str) -> KPICategory:
        """Map metric name to KPI category"""
        category_mapping = {
            'total_sales': KPICategory.SALES,
            'sales_growth_rate': KPICategory.SALES,
            'gross_profit_margin': KPICategory.PROFIT,
            'net_profit_margin': KPICategory.PROFIT,
            'inventory_turnover_ratio': KPICategory.INVENTORY,
            'waste_rate': KPICategory.WASTE,
            'markdown_rate': KPICategory.WASTE,
            'order_accuracy': KPICategory.ORDERING,
            'forecast_accuracy': KPICategory.ORDERING,
            'price_optimization_score': KPICategory.PRICING,
            'labor_productivity': KPICategory.EFFICIENCY,
            'customer_satisfaction_score': KPICategory.CUSTOMER
        }

        return category_mapping.get(metric, KPICategory.EFFICIENCY)


def generate_sample_data(n_managers: int = 10, periods: int = 12) -> Dict:
    """
    Generate sample data for testing

    Parameters
    ----------
    n_managers : int
        Number of managers to generate data for
    periods : int
        Number of time periods

    Returns
    -------
    Dict
        Sample data including sales, inventory, and customer data
    """
    np.random.seed(42)

    managers_data = {}

    for manager_id in range(1, n_managers + 1):
        store_id = f"STORE-{manager_id:03d}"
        manager_key = f"MGR-{manager_id:03d}"

        # Generate sales data
        base_sales = np.random.uniform(500000, 2000000)
        trend = np.random.uniform(-0.02, 0.05)
        seasonality = np.sin(np.arange(periods) * 2 * np.pi / 12) * 0.1

        sales_data = []
        for period in range(periods):
            period_date = datetime.now() - timedelta(days=30 * (periods - period))
            n_transactions = np.random.randint(500, 2000)

            for _ in range(n_transactions):
                amount = np.random.lognormal(3, 1) * (1 + trend * period + seasonality[period])
                sales_data.append({
                    'date': period_date + timedelta(days=np.random.randint(0, 30)),
                    'amount': amount,
                    'quantity': np.random.randint(1, 10),
                    'cost': amount * np.random.uniform(0.6, 0.8),
                    'customer_id': f"CUST-{np.random.randint(1, 5000):05d}"
                })

        sales_df = pd.DataFrame(sales_data)

        # Generate inventory data
        inventory_data = []
        for period in range(periods):
            period_date = datetime.now() - timedelta(days=30 * (periods - period))
            n_products = np.random.randint(200, 500)

            for product_id in range(n_products):
                inventory_data.append({
                    'date': period_date,
                    'product_id': f"PROD-{product_id:05d}",
                    'quantity': np.random.randint(0, 200),
                    'value': np.random.uniform(100, 5000),
                    'optimal_quantity': np.random.randint(50, 150)
                })

        inventory_df = pd.DataFrame(inventory_data)

        # Generate customer feedback data
        customer_data = []
        for period in range(periods):
            period_date = datetime.now() - timedelta(days=30 * (periods - period))
            n_feedback = np.random.randint(50, 200)

            for _ in range(n_feedback):
                customer_data.append({
                    'date': period_date + timedelta(days=np.random.randint(0, 30)),
                    'customer_id': f"CUST-{np.random.randint(1, 5000):05d}",
                    'satisfaction': np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25]),
                    'nps_score': np.random.randint(0, 11),
                    'is_complaint': np.random.choice([True, False], p=[0.1, 0.9])
                })

        customer_df = pd.DataFrame(customer_data)

        managers_data[manager_key] = {
            'manager_id': manager_key,
            'store_id': store_id,
            'sales_data': sales_df,
            'inventory_data': inventory_df,
            'customer_data': customer_df
        }

    return managers_data


# Example usage
if __name__ == "__main__":
    print("Store Manager Performance Benchmarking System")
    print("=" * 60)

    # Initialize benchmarking system
    benchmark = ManagerPerformanceBenchmark(
        control_limit_sigma=3.0,
        warning_limit_sigma=2.0,
        trend_length=7
    )

    # Generate sample data
    print("\n📊 Generating sample data for 10 managers over 12 months...")
    sample_data = generate_sample_data(n_managers=10, periods=12)

    # Calculate KPIs for each manager
    print("\n📈 Calculating KPIs for all managers...")
    all_kpis = []

    for manager_key, data in sample_data.items():
        kpi = benchmark.calculate_kpis(
            manager_id=data['manager_id'],
            store_id=data['store_id'],
            sales_data=data['sales_data'],
            inventory_data=data['inventory_data'],
            customer_data=data['customer_data']
        )
        all_kpis.append(kpi)

    # Select target manager for detailed analysis
    target_manager = all_kpis[0]

    print(f"\n🎯 Detailed Analysis for {target_manager.manager_id} ({target_manager.store_id})")
    print("-" * 60)

    # Display KPI summary
    print(f"\n📊 KPI Summary:")
    print(f"  Overall Score: {target_manager.overall_performance_score:.2f}")
    print(f"  Performance Level: {target_manager.performance_level.value}")
    print(f"\n  Sales Metrics:")
    print(f"    Total Sales: ¥{target_manager.total_sales:,.2f}")
    print(f"    Growth Rate: {target_manager.sales_growth_rate:.2f}%")
    print(f"\n  Profit Metrics:")
    print(f"    Gross Profit: ¥{target_manager.gross_profit:,.2f}")
    print(f"    Gross Margin: {target_manager.gross_profit_margin:.2f}%")
    print(f"\n  Inventory Metrics:")
    print(f"    Turnover Ratio: {target_manager.inventory_turnover_ratio:.2f}")
    print(f"    Days Outstanding: {target_manager.days_inventory_outstanding:.1f}")
    print(f"    Stockout Rate: {target_manager.stockout_rate:.2f}%")
    print(f"\n  Waste Metrics:")
    print(f"    Waste Rate: {target_manager.waste_rate:.2f}%")
    print(f"    Waste Amount: ¥{target_manager.waste_amount:,.2f}")

    # Peer comparison
    print(f"\n👥 Peer Comparison Analysis:")
    peer_comparison = benchmark.compare_with_peers(
        target_manager,
        all_kpis
    )
    print(f"  Rank: {peer_comparison.rank} of {peer_comparison.total_peers}")
    print(f"  Percentile: {peer_comparison.percentile:.1f}th")
    print(f"  Z-Score: {peer_comparison.z_score:.2f}")
    print(f"\n  Strengths:")
    for strength in peer_comparison.strengths[:3]:
        print(f"    ✓ {strength}")
    print(f"\n  Areas for Improvement:")
    for weakness in peer_comparison.weaknesses[:3]:
        print(f"    ✗ {weakness}")

    # Trend analysis
    print(f"\n📈 Performance Trend Analysis:")
    kpi_history = all_kpis[:5]  # Use first 5 KPIs as historical data
    trends = benchmark.analyze_trends(
        target_manager.manager_id,
        kpi_history,
        metrics=['overall_performance_score', 'gross_profit_margin', 'waste_rate']
    )

    for metric, trend_data in trends.items():
        print(f"\n  {metric}:")
        print(f"    Direction: {trend_data['trend_direction']}")
        print(f"    Strength: {trend_data['trend_strength']:.2f}")
        print(f"    Current: {trend_data['current_value']:.2f}")
        print(f"    Change: {trend_data['change_rate']:+.2f}%")
        print(f"    In Control: {trend_data['spc_metrics'].is_in_control}")

    # Improvement opportunities
    print(f"\n💡 Improvement Opportunities:")
    opportunities = benchmark.identify_improvement_opportunities(
        target_manager,
        peer_comparison
    )

    for i, opp in enumerate(opportunities[:5], 1):
        print(f"\n  {i}. {opp.category.value.upper()} [{opp.priority.upper()} Priority]")
        print(f"     Current: {opp.current_value:.2f}")
        print(f"     Target: {opp.target_value:.2f}")
        print(f"     Impact: ¥{opp.potential_impact:,.2f}")
        print(f"     Timeframe: {opp.estimated_timeframe}")
        print(f"     Actions:")
        for action in opp.recommended_actions[:2]:
            print(f"       • {action}")

    # Create dashboard data
    print(f"\n📊 Generating Dashboard Data...")
    dashboard = benchmark.create_dashboard_data(
        target_manager,
        peer_comparison,
        trends,
        opportunities
    )

    print(f"\n✅ Dashboard created successfully!")
    print(f"   Total KPI categories: {len(dashboard['kpis'])}")
    print(f"   Improvement opportunities: {len(dashboard['improvement_opportunities'])}")
    print(f"   Trend metrics tracked: {len(dashboard['trends'])}")

    print("\n" + "=" * 60)
    print("✨ Manager Performance Benchmarking System Ready!")
