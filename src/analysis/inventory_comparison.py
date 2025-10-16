"""
Inventory Management Comparison Module

This module provides comprehensive inventory management analysis and benchmarking
across stores, including turnover rates, stock-out analysis, shrinkage tracking,
and working capital efficiency metrics.

Author: PyCaret Development Team
Date: 2025-10-08
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class IndustryBenchmarks:
    """Industry benchmark data for retail inventory metrics."""

    # Inventory Turnover (times per year)
    turnover_excellent: float = 12.0  # >12 times/year
    turnover_good: float = 8.0        # 8-12 times/year
    turnover_average: float = 6.0     # 6-8 times/year
    turnover_poor: float = 4.0        # <4 times/year

    # Days of Inventory on Hand (DOH)
    doh_excellent: float = 30.0       # <30 days
    doh_good: float = 45.0            # 30-45 days
    doh_average: float = 60.0         # 45-60 days
    doh_poor: float = 90.0            # >90 days

    # Stock-out Rate (%)
    stockout_excellent: float = 2.0   # <2%
    stockout_good: float = 5.0        # 2-5%
    stockout_average: float = 8.0     # 5-8%
    stockout_poor: float = 10.0       # >10%

    # Shrinkage Rate (%)
    shrinkage_excellent: float = 0.5  # <0.5%
    shrinkage_good: float = 1.0       # 0.5-1%
    shrinkage_average: float = 1.5    # 1-1.5%
    shrinkage_poor: float = 2.0       # >2%

    # Fill Rate (%)
    fill_rate_excellent: float = 98.0 # >98%
    fill_rate_good: float = 95.0      # 95-98%
    fill_rate_average: float = 92.0   # 92-95%
    fill_rate_poor: float = 90.0      # <90%

    # Working Capital Efficiency (inventory/sales ratio)
    wc_excellent: float = 0.15        # <15%
    wc_good: float = 0.20             # 15-20%
    wc_average: float = 0.25          # 20-25%
    wc_poor: float = 0.30             # >30%


class InventoryComparisonAnalyzer:
    """
    Comprehensive inventory management comparison and benchmarking analyzer.

    This class provides methods for analyzing inventory performance across stores,
    comparing against industry benchmarks, and identifying optimization opportunities.
    """

    def __init__(self, benchmarks: Optional[IndustryBenchmarks] = None):
        """
        Initialize the Inventory Comparison Analyzer.

        Parameters
        ----------
        benchmarks : IndustryBenchmarks, optional
            Custom industry benchmarks. If None, uses default benchmarks.
        """
        self.benchmarks = benchmarks or IndustryBenchmarks()
        self.results = {}

    def calculate_inventory_turnover(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        sales_col: str = 'sales',
        inventory_col: str = 'inventory_value',
        period_days: int = 365
    ) -> pd.DataFrame:
        """
        Calculate inventory turnover rate for each store.

        Inventory Turnover = Cost of Goods Sold / Average Inventory Value
        Higher values indicate more efficient inventory management.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with store, sales, and inventory information
        store_col : str
            Column name for store identifier
        sales_col : str
            Column name for sales/COGS
        inventory_col : str
            Column name for inventory value
        period_days : int
            Analysis period in days (default: 365)

        Returns
        -------
        pd.DataFrame
            Turnover metrics by store with benchmark comparison
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            # Calculate metrics
            total_cogs = store_data[sales_col].sum()
            avg_inventory = store_data[inventory_col].mean()

            if avg_inventory > 0:
                # Annual turnover rate
                turnover_rate = (total_cogs / avg_inventory) * (365 / period_days)

                # Benchmark classification
                if turnover_rate >= self.benchmarks.turnover_excellent:
                    performance = 'Excellent'
                elif turnover_rate >= self.benchmarks.turnover_good:
                    performance = 'Good'
                elif turnover_rate >= self.benchmarks.turnover_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                results.append({
                    'store_id': store,
                    'turnover_rate': round(turnover_rate, 2),
                    'total_cogs': round(total_cogs, 2),
                    'avg_inventory_value': round(avg_inventory, 2),
                    'performance': performance,
                    'vs_excellent': round(turnover_rate - self.benchmarks.turnover_excellent, 2),
                    'vs_industry_avg': round(turnover_rate - self.benchmarks.turnover_average, 2)
                })

        result_df = pd.DataFrame(results)
        self.results['inventory_turnover'] = result_df
        return result_df

    def calculate_days_on_hand(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        inventory_col: str = 'inventory_value',
        daily_sales_col: str = 'daily_sales'
    ) -> pd.DataFrame:
        """
        Calculate Days of Inventory on Hand (DOH) for each store.

        DOH = Average Inventory Value / Average Daily Sales
        Lower values indicate faster inventory movement.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with store and inventory information
        store_col : str
            Column name for store identifier
        inventory_col : str
            Column name for inventory value
        daily_sales_col : str
            Column name for daily sales

        Returns
        -------
        pd.DataFrame
            DOH metrics by store with benchmark comparison
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            avg_inventory = store_data[inventory_col].mean()
            avg_daily_sales = store_data[daily_sales_col].mean()

            if avg_daily_sales > 0:
                doh = avg_inventory / avg_daily_sales

                # Benchmark classification
                if doh <= self.benchmarks.doh_excellent:
                    performance = 'Excellent'
                elif doh <= self.benchmarks.doh_good:
                    performance = 'Good'
                elif doh <= self.benchmarks.doh_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                results.append({
                    'store_id': store,
                    'days_on_hand': round(doh, 1),
                    'avg_inventory_value': round(avg_inventory, 2),
                    'avg_daily_sales': round(avg_daily_sales, 2),
                    'performance': performance,
                    'vs_excellent': round(doh - self.benchmarks.doh_excellent, 1),
                    'improvement_potential_days': round(max(0, doh - self.benchmarks.doh_excellent), 1)
                })

        result_df = pd.DataFrame(results)
        self.results['days_on_hand'] = result_df
        return result_df

    def analyze_stockout_frequency(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        product_col: str = 'product_id',
        inventory_col: str = 'inventory_qty',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Analyze stock-out frequency and patterns across stores.

        Stock-out Rate = (Days with Stock-outs / Total Days) * 100

        Parameters
        ----------
        data : pd.DataFrame
            Input data with inventory levels over time
        store_col : str
            Column name for store identifier
        product_col : str
            Column name for product identifier
        inventory_col : str
            Column name for inventory quantity
        date_col : str
            Column name for date

        Returns
        -------
        pd.DataFrame
            Stock-out analysis by store with benchmark comparison
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            # Calculate stock-out metrics
            total_observations = len(store_data)
            stockout_count = len(store_data[store_data[inventory_col] == 0])

            if total_observations > 0:
                stockout_rate = (stockout_count / total_observations) * 100

                # Benchmark classification
                if stockout_rate <= self.benchmarks.stockout_excellent:
                    performance = 'Excellent'
                elif stockout_rate <= self.benchmarks.stockout_good:
                    performance = 'Good'
                elif stockout_rate <= self.benchmarks.stockout_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                # Product-level analysis
                products_tracked = store_data[product_col].nunique()
                products_with_stockouts = store_data[
                    store_data[inventory_col] == 0
                ][product_col].nunique()

                results.append({
                    'store_id': store,
                    'stockout_rate_pct': round(stockout_rate, 2),
                    'stockout_incidents': stockout_count,
                    'total_observations': total_observations,
                    'products_tracked': products_tracked,
                    'products_with_stockouts': products_with_stockouts,
                    'stockout_product_pct': round((products_with_stockouts / products_tracked) * 100, 2),
                    'performance': performance,
                    'vs_excellent': round(stockout_rate - self.benchmarks.stockout_excellent, 2)
                })

        result_df = pd.DataFrame(results)
        self.results['stockout_analysis'] = result_df
        return result_df

    def analyze_shrinkage_waste(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        expected_inventory_col: str = 'expected_inventory',
        actual_inventory_col: str = 'actual_inventory',
        waste_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Benchmark shrinkage and waste rates across stores.

        Shrinkage Rate = (Expected - Actual) / Expected * 100

        Parameters
        ----------
        data : pd.DataFrame
            Input data with inventory and waste information
        store_col : str
            Column name for store identifier
        expected_inventory_col : str
            Column name for expected inventory value
        actual_inventory_col : str
            Column name for actual inventory value
        waste_col : str, optional
            Column name for waste value

        Returns
        -------
        pd.DataFrame
            Shrinkage and waste metrics by store with benchmarks
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            total_expected = store_data[expected_inventory_col].sum()
            total_actual = store_data[actual_inventory_col].sum()

            if total_expected > 0:
                shrinkage = total_expected - total_actual
                shrinkage_rate = (shrinkage / total_expected) * 100

                # Benchmark classification
                if shrinkage_rate <= self.benchmarks.shrinkage_excellent:
                    performance = 'Excellent'
                elif shrinkage_rate <= self.benchmarks.shrinkage_good:
                    performance = 'Good'
                elif shrinkage_rate <= self.benchmarks.shrinkage_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                result = {
                    'store_id': store,
                    'shrinkage_rate_pct': round(shrinkage_rate, 2),
                    'shrinkage_value': round(shrinkage, 2),
                    'expected_inventory': round(total_expected, 2),
                    'actual_inventory': round(total_actual, 2),
                    'performance': performance,
                    'vs_excellent': round(shrinkage_rate - self.benchmarks.shrinkage_excellent, 2),
                    'potential_savings': round(shrinkage * (shrinkage_rate - self.benchmarks.shrinkage_excellent) / 100, 2)
                }

                # Add waste analysis if available
                if waste_col and waste_col in store_data.columns:
                    total_waste = store_data[waste_col].sum()
                    waste_rate = (total_waste / total_expected) * 100
                    result['waste_value'] = round(total_waste, 2)
                    result['waste_rate_pct'] = round(waste_rate, 2)
                    result['total_loss_rate_pct'] = round(shrinkage_rate + waste_rate, 2)

                results.append(result)

        result_df = pd.DataFrame(results)
        self.results['shrinkage_waste'] = result_df
        return result_df

    def analyze_order_frequency(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        order_date_col: str = 'order_date',
        product_col: str = 'product_id',
        order_qty_col: str = 'order_quantity'
    ) -> pd.DataFrame:
        """
        Analyze order frequency patterns across stores.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with order information
        store_col : str
            Column name for store identifier
        order_date_col : str
            Column name for order date
        product_col : str
            Column name for product identifier
        order_qty_col : str
            Column name for order quantity

        Returns
        -------
        pd.DataFrame
            Order frequency analysis by store
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()
            store_data[order_date_col] = pd.to_datetime(store_data[order_date_col])

            # Calculate order metrics
            total_orders = len(store_data)
            unique_products = store_data[product_col].nunique()

            # Date range analysis
            date_range = (store_data[order_date_col].max() -
                         store_data[order_date_col].min()).days

            if date_range > 0:
                orders_per_day = total_orders / date_range

                # Calculate average order interval
                store_data = store_data.sort_values(order_date_col)
                order_intervals = store_data[order_date_col].diff().dt.days.dropna()
                avg_order_interval = order_intervals.mean()

                # Order size analysis
                avg_order_qty = store_data[order_qty_col].mean()
                total_order_qty = store_data[order_qty_col].sum()

                # Order consistency (coefficient of variation)
                order_cv = (order_intervals.std() / avg_order_interval) * 100 if avg_order_interval > 0 else 0

                results.append({
                    'store_id': store,
                    'total_orders': total_orders,
                    'unique_products_ordered': unique_products,
                    'avg_order_interval_days': round(avg_order_interval, 1),
                    'orders_per_day': round(orders_per_day, 2),
                    'avg_order_quantity': round(avg_order_qty, 2),
                    'total_order_quantity': round(total_order_qty, 2),
                    'order_consistency_cv': round(order_cv, 2),
                    'pattern': 'Consistent' if order_cv < 30 else 'Variable' if order_cv < 60 else 'Erratic'
                })

        result_df = pd.DataFrame(results)
        self.results['order_frequency'] = result_df
        return result_df

    def calculate_fill_rate(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        ordered_qty_col: str = 'ordered_quantity',
        received_qty_col: str = 'received_quantity'
    ) -> pd.DataFrame:
        """
        Analyze fill rate (order fulfillment) across stores.

        Fill Rate = (Received Quantity / Ordered Quantity) * 100

        Parameters
        ----------
        data : pd.DataFrame
            Input data with order and receipt information
        store_col : str
            Column name for store identifier
        ordered_qty_col : str
            Column name for ordered quantity
        received_qty_col : str
            Column name for received quantity

        Returns
        -------
        pd.DataFrame
            Fill rate analysis by store with benchmarks
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            total_ordered = store_data[ordered_qty_col].sum()
            total_received = store_data[received_qty_col].sum()

            if total_ordered > 0:
                fill_rate = (total_received / total_ordered) * 100

                # Benchmark classification
                if fill_rate >= self.benchmarks.fill_rate_excellent:
                    performance = 'Excellent'
                elif fill_rate >= self.benchmarks.fill_rate_good:
                    performance = 'Good'
                elif fill_rate >= self.benchmarks.fill_rate_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                # Calculate unfulfilled quantity
                unfulfilled_qty = total_ordered - total_received
                unfulfilled_rate = 100 - fill_rate

                results.append({
                    'store_id': store,
                    'fill_rate_pct': round(fill_rate, 2),
                    'total_ordered': round(total_ordered, 2),
                    'total_received': round(total_received, 2),
                    'unfulfilled_quantity': round(unfulfilled_qty, 2),
                    'unfulfilled_rate_pct': round(unfulfilled_rate, 2),
                    'performance': performance,
                    'vs_excellent': round(fill_rate - self.benchmarks.fill_rate_excellent, 2)
                })

        result_df = pd.DataFrame(results)
        self.results['fill_rate'] = result_df
        return result_df

    def calculate_working_capital_efficiency(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        inventory_col: str = 'inventory_value',
        sales_col: str = 'sales',
        period_days: int = 365
    ) -> pd.DataFrame:
        """
        Analyze working capital efficiency across stores.

        WC Efficiency = Average Inventory / Sales
        Lower ratios indicate better working capital management.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with inventory and sales information
        store_col : str
            Column name for store identifier
        inventory_col : str
            Column name for inventory value
        sales_col : str
            Column name for sales
        period_days : int
            Analysis period in days

        Returns
        -------
        pd.DataFrame
            Working capital efficiency by store with benchmarks
        """
        results = []

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store].copy()

            avg_inventory = store_data[inventory_col].mean()
            total_sales = store_data[sales_col].sum()

            # Annualize sales if needed
            annualized_sales = total_sales * (365 / period_days)

            if annualized_sales > 0:
                wc_ratio = avg_inventory / annualized_sales

                # Benchmark classification
                if wc_ratio <= self.benchmarks.wc_excellent:
                    performance = 'Excellent'
                elif wc_ratio <= self.benchmarks.wc_good:
                    performance = 'Good'
                elif wc_ratio <= self.benchmarks.wc_average:
                    performance = 'Average'
                else:
                    performance = 'Poor'

                # Calculate tied-up capital
                excess_inventory = max(0, avg_inventory - (annualized_sales * self.benchmarks.wc_excellent))

                results.append({
                    'store_id': store,
                    'wc_efficiency_ratio': round(wc_ratio, 3),
                    'wc_efficiency_pct': round(wc_ratio * 100, 2),
                    'avg_inventory_value': round(avg_inventory, 2),
                    'annualized_sales': round(annualized_sales, 2),
                    'excess_inventory_value': round(excess_inventory, 2),
                    'performance': performance,
                    'vs_excellent': round((wc_ratio - self.benchmarks.wc_excellent) * 100, 2),
                    'potential_capital_release': round(excess_inventory, 2)
                })

        result_df = pd.DataFrame(results)
        self.results['working_capital'] = result_df
        return result_df

    def generate_comprehensive_report(
        self,
        data: pd.DataFrame,
        store_col: str = 'store_id',
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a comprehensive inventory management report.

        Runs all analysis methods and returns combined results.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with all required columns
        store_col : str
            Column name for store identifier
        **kwargs
            Additional parameters for specific analysis methods

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all analysis results
        """
        report = {}

        try:
            # Inventory Turnover
            if all(col in data.columns for col in [store_col, 'sales', 'inventory_value']):
                report['inventory_turnover'] = self.calculate_inventory_turnover(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['sales_col', 'inventory_col', 'period_days']}
                )
        except Exception as e:
            warnings.warn(f"Inventory turnover calculation failed: {str(e)}")

        try:
            # Days on Hand
            if all(col in data.columns for col in [store_col, 'inventory_value', 'daily_sales']):
                report['days_on_hand'] = self.calculate_days_on_hand(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['inventory_col', 'daily_sales_col']}
                )
        except Exception as e:
            warnings.warn(f"Days on hand calculation failed: {str(e)}")

        try:
            # Stock-out Analysis
            if all(col in data.columns for col in [store_col, 'product_id', 'inventory_qty', 'date']):
                report['stockout_analysis'] = self.analyze_stockout_frequency(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['product_col', 'inventory_col', 'date_col']}
                )
        except Exception as e:
            warnings.warn(f"Stock-out analysis failed: {str(e)}")

        try:
            # Shrinkage and Waste
            if all(col in data.columns for col in [store_col, 'expected_inventory', 'actual_inventory']):
                report['shrinkage_waste'] = self.analyze_shrinkage_waste(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['expected_inventory_col', 'actual_inventory_col', 'waste_col']}
                )
        except Exception as e:
            warnings.warn(f"Shrinkage analysis failed: {str(e)}")

        try:
            # Order Frequency
            if all(col in data.columns for col in [store_col, 'order_date', 'product_id', 'order_quantity']):
                report['order_frequency'] = self.analyze_order_frequency(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['order_date_col', 'product_col', 'order_qty_col']}
                )
        except Exception as e:
            warnings.warn(f"Order frequency analysis failed: {str(e)}")

        try:
            # Fill Rate
            if all(col in data.columns for col in [store_col, 'ordered_quantity', 'received_quantity']):
                report['fill_rate'] = self.calculate_fill_rate(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['ordered_qty_col', 'received_qty_col']}
                )
        except Exception as e:
            warnings.warn(f"Fill rate calculation failed: {str(e)}")

        try:
            # Working Capital Efficiency
            if all(col in data.columns for col in [store_col, 'inventory_value', 'sales']):
                report['working_capital'] = self.calculate_working_capital_efficiency(
                    data, store_col=store_col,
                    **{k: v for k, v in kwargs.items() if k in ['inventory_col', 'sales_col', 'period_days']}
                )
        except Exception as e:
            warnings.warn(f"Working capital analysis failed: {str(e)}")

        self.results['comprehensive_report'] = report
        return report

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Generate a summary of overall performance across all metrics.

        Returns
        -------
        pd.DataFrame
            Summary of performance by store across all analyzed metrics
        """
        if not self.results:
            raise ValueError("No analysis results available. Run analysis methods first.")

        summary_data = []

        # Get all stores
        all_stores = set()
        for result_df in self.results.values():
            if isinstance(result_df, pd.DataFrame) and 'store_id' in result_df.columns:
                all_stores.update(result_df['store_id'].unique())

        for store in all_stores:
            store_summary = {'store_id': store}

            # Collect performance ratings from each analysis
            for metric_name, result_df in self.results.items():
                if isinstance(result_df, pd.DataFrame) and 'store_id' in result_df.columns:
                    store_result = result_df[result_df['store_id'] == store]
                    if not store_result.empty and 'performance' in store_result.columns:
                        store_summary[f'{metric_name}_performance'] = store_result['performance'].values[0]

            summary_data.append(store_summary)

        return pd.DataFrame(summary_data)


def calculate_inventory_metrics(
    data: pd.DataFrame,
    store_col: str = 'store_id',
    custom_benchmarks: Optional[IndustryBenchmarks] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to calculate all inventory metrics at once.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with required columns
    store_col : str
        Column name for store identifier
    custom_benchmarks : IndustryBenchmarks, optional
        Custom benchmark values

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of all analysis results
    """
    analyzer = InventoryComparisonAnalyzer(benchmarks=custom_benchmarks)
    return analyzer.generate_comprehensive_report(data, store_col=store_col)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_stores = 5
    n_days = 90

    sample_data = []
    for store_id in range(1, n_stores + 1):
        for day in range(n_days):
            sample_data.append({
                'store_id': f'Store_{store_id}',
                'date': datetime(2025, 1, 1) + timedelta(days=day),
                'sales': np.random.uniform(5000, 15000),
                'inventory_value': np.random.uniform(50000, 100000),
                'daily_sales': np.random.uniform(5000, 15000),
                'inventory_qty': np.random.randint(0, 1000),
                'product_id': f'P{np.random.randint(1, 50)}',
                'expected_inventory': np.random.uniform(50000, 100000),
                'actual_inventory': np.random.uniform(48000, 99000),
                'order_date': datetime(2025, 1, 1) + timedelta(days=day),
                'order_quantity': np.random.randint(100, 500),
                'ordered_quantity': np.random.randint(100, 500),
                'received_quantity': np.random.randint(90, 500)
            })

    df = pd.DataFrame(sample_data)

    # Initialize analyzer
    analyzer = InventoryComparisonAnalyzer()

    # Run comprehensive analysis
    print("=" * 80)
    print("INVENTORY MANAGEMENT COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    report = analyzer.generate_comprehensive_report(df, store_col='store_id')

    for metric_name, result_df in report.items():
        print(f"\n{metric_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(result_df.to_string(index=False))
        print()

    # Performance summary
    print("\nPERFORMANCE SUMMARY")
    print("-" * 80)
    summary = analyzer.get_performance_summary()
    print(summary.to_string(index=False))
