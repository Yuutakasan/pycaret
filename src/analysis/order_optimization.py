"""
Order Optimization Analysis Module

This module provides comprehensive inventory order optimization analysis including:
- Economic Order Quantity (EOQ) calculation
- Reorder point determination
- Safety stock optimization
- Lead time analysis
- Stock-out risk assessment
- Overstock detection
- Cost-benefit analysis
- Inventory turnover metrics
- ROI calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar


class ServiceLevel(Enum):
    """Service level targets for safety stock calculations"""
    LOW = 0.90  # 90% service level
    MEDIUM = 0.95  # 95% service level
    HIGH = 0.99  # 99% service level
    CRITICAL = 0.999  # 99.9% service level


@dataclass
class InventoryMetrics:
    """Container for inventory performance metrics"""
    turnover_ratio: float
    days_of_inventory: float
    stock_to_sales_ratio: float
    carrying_cost_percentage: float
    stockout_rate: float
    fill_rate: float


@dataclass
class OrderOptimizationResult:
    """Container for order optimization results"""
    eoq: float
    reorder_point: float
    safety_stock: float
    order_frequency: float
    total_annual_cost: float
    ordering_cost: float
    holding_cost: float
    stockout_cost: float
    roi: float
    metrics: InventoryMetrics
    recommendations: List[str]


class OrderOptimizationAnalyzer:
    """
    Advanced order optimization analyzer for inventory management.

    This class implements various inventory optimization techniques including
    EOQ, reorder point calculation, safety stock optimization, and comprehensive
    cost-benefit analysis.
    """

    def __init__(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_rate: float,
        unit_cost: float,
        lead_time: float,
        demand_std: Optional[float] = None,
        service_level: Union[float, ServiceLevel] = ServiceLevel.MEDIUM,
        stockout_cost: Optional[float] = None,
        working_days: int = 365
    ):
        """
        Initialize the Order Optimization Analyzer.

        Parameters
        ----------
        annual_demand : float
            Total annual demand in units
        ordering_cost : float
            Fixed cost per order
        holding_cost_rate : float
            Annual holding cost as percentage of unit cost (e.g., 0.25 for 25%)
        unit_cost : float
            Cost per unit
        lead_time : float
            Lead time in days
        demand_std : float, optional
            Standard deviation of daily demand
        service_level : float or ServiceLevel, optional
            Desired service level (0-1) or ServiceLevel enum
        stockout_cost : float, optional
            Cost per stockout unit (if None, uses 2x unit_cost)
        working_days : int, optional
            Number of working days per year (default: 365)
        """
        self.annual_demand = annual_demand
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
        self.unit_cost = unit_cost
        self.lead_time = lead_time
        self.working_days = working_days

        # Calculate daily demand
        self.daily_demand = annual_demand / working_days

        # Handle demand variability
        if demand_std is None:
            # Estimate as 20% of mean if not provided
            self.demand_std = self.daily_demand * 0.20
            warnings.warn(
                "Demand standard deviation not provided. Using estimated value of 20% of mean demand.",
                UserWarning
            )
        else:
            self.demand_std = demand_std

        # Handle service level
        if isinstance(service_level, ServiceLevel):
            self.service_level = service_level.value
        else:
            self.service_level = service_level

        # Calculate stockout cost
        if stockout_cost is None:
            self.stockout_cost = 2 * unit_cost
        else:
            self.stockout_cost = stockout_cost

        # Calculate holding cost per unit
        self.holding_cost = holding_cost_rate * unit_cost

    def calculate_eoq(self) -> float:
        """
        Calculate Economic Order Quantity (EOQ).

        The EOQ is the optimal order quantity that minimizes total inventory costs.

        Returns
        -------
        float
            Economic Order Quantity
        """
        eoq = np.sqrt((2 * self.annual_demand * self.ordering_cost) / self.holding_cost)
        return eoq

    def calculate_safety_stock(self, service_level: Optional[float] = None) -> float:
        """
        Calculate optimal safety stock based on service level.

        Parameters
        ----------
        service_level : float, optional
            Desired service level (0-1). If None, uses instance service_level.

        Returns
        -------
        float
            Safety stock quantity
        """
        if service_level is None:
            service_level = self.service_level

        # Calculate z-score for desired service level
        z_score = stats.norm.ppf(service_level)

        # Safety stock = z * σ * √L where L is lead time
        safety_stock = z_score * self.demand_std * np.sqrt(self.lead_time)

        return max(0, safety_stock)

    def calculate_reorder_point(self, safety_stock: Optional[float] = None) -> float:
        """
        Calculate reorder point.

        ROP = (Daily Demand × Lead Time) + Safety Stock

        Parameters
        ----------
        safety_stock : float, optional
            Safety stock quantity. If None, calculated automatically.

        Returns
        -------
        float
            Reorder point
        """
        if safety_stock is None:
            safety_stock = self.calculate_safety_stock()

        rop = (self.daily_demand * self.lead_time) + safety_stock
        return rop

    def calculate_order_frequency(self, order_quantity: Optional[float] = None) -> float:
        """
        Calculate annual order frequency.

        Parameters
        ----------
        order_quantity : float, optional
            Order quantity. If None, uses EOQ.

        Returns
        -------
        float
            Number of orders per year
        """
        if order_quantity is None:
            order_quantity = self.calculate_eoq()

        return self.annual_demand / order_quantity

    def calculate_total_cost(
        self,
        order_quantity: float,
        safety_stock: Optional[float] = None,
        include_stockout_cost: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total annual inventory cost.

        Parameters
        ----------
        order_quantity : float
            Order quantity
        safety_stock : float, optional
            Safety stock quantity
        include_stockout_cost : bool, optional
            Whether to include expected stockout costs

        Returns
        -------
        dict
            Dictionary with cost breakdown
        """
        if safety_stock is None:
            safety_stock = self.calculate_safety_stock()

        # Ordering cost
        num_orders = self.annual_demand / order_quantity
        annual_ordering_cost = num_orders * self.ordering_cost

        # Holding cost (average inventory + safety stock)
        average_inventory = (order_quantity / 2) + safety_stock
        annual_holding_cost = average_inventory * self.holding_cost

        # Stockout cost (expected)
        annual_stockout_cost = 0
        if include_stockout_cost:
            # Calculate expected stockouts per cycle
            stockout_probability = 1 - self.service_level
            expected_stockout_units = self._calculate_expected_shortage(safety_stock)
            annual_stockout_cost = (
                num_orders * stockout_probability * expected_stockout_units * self.stockout_cost
            )

        total_cost = annual_ordering_cost + annual_holding_cost + annual_stockout_cost

        return {
            'ordering_cost': annual_ordering_cost,
            'holding_cost': annual_holding_cost,
            'stockout_cost': annual_stockout_cost,
            'total_cost': total_cost
        }

    def _calculate_expected_shortage(self, safety_stock: float) -> float:
        """
        Calculate expected shortage per order cycle.

        Parameters
        ----------
        safety_stock : float
            Current safety stock level

        Returns
        -------
        float
            Expected shortage in units
        """
        # Using normal distribution for demand during lead time
        z = safety_stock / (self.demand_std * np.sqrt(self.lead_time))

        # Expected shortage formula for normal distribution
        expected_shortage = self.demand_std * np.sqrt(self.lead_time) * (
            stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
        )

        return max(0, expected_shortage)

    def optimize_safety_stock(self) -> Tuple[float, Dict[str, float]]:
        """
        Optimize safety stock to minimize total cost including stockout costs.

        Returns
        -------
        tuple
            (optimal_safety_stock, cost_breakdown)
        """
        eoq = self.calculate_eoq()

        def cost_function(ss):
            costs = self.calculate_total_cost(eoq, ss, include_stockout_cost=True)
            return costs['total_cost']

        # Optimize safety stock
        result = minimize_scalar(
            cost_function,
            bounds=(0, self.daily_demand * self.lead_time * 3),
            method='bounded'
        )

        optimal_ss = result.x
        costs = self.calculate_total_cost(eoq, optimal_ss, include_stockout_cost=True)

        return optimal_ss, costs

    def assess_stockout_risk(self, current_inventory: float, pending_orders: float = 0) -> Dict[str, float]:
        """
        Assess risk of stockout given current inventory levels.

        Parameters
        ----------
        current_inventory : float
            Current inventory on hand
        pending_orders : float, optional
            Units on order but not yet received

        Returns
        -------
        dict
            Risk assessment metrics
        """
        rop = self.calculate_reorder_point()
        safety_stock = self.calculate_safety_stock()

        # Effective inventory
        effective_inventory = current_inventory + pending_orders

        # Days of inventory remaining
        days_remaining = effective_inventory / self.daily_demand if self.daily_demand > 0 else float('inf')

        # Stockout probability during lead time
        if effective_inventory < rop:
            # Calculate z-score for current inventory level
            expected_demand = self.daily_demand * self.lead_time
            demand_std_leadtime = self.demand_std * np.sqrt(self.lead_time)

            if demand_std_leadtime > 0:
                z_score = (effective_inventory - expected_demand) / demand_std_leadtime
                stockout_probability = 1 - stats.norm.cdf(z_score)
            else:
                stockout_probability = 1.0 if effective_inventory < expected_demand else 0.0
        else:
            stockout_probability = 1 - self.service_level

        # Risk level categorization
        if stockout_probability > 0.50:
            risk_level = "CRITICAL"
        elif stockout_probability > 0.25:
            risk_level = "HIGH"
        elif stockout_probability > 0.10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            'current_inventory': current_inventory,
            'pending_orders': pending_orders,
            'effective_inventory': effective_inventory,
            'reorder_point': rop,
            'safety_stock': safety_stock,
            'days_remaining': days_remaining,
            'stockout_probability': stockout_probability,
            'risk_level': risk_level,
            'should_reorder': effective_inventory <= rop
        }

    def detect_overstock(self, current_inventory: float, threshold_multiplier: float = 2.0) -> Dict[str, Union[bool, float, str]]:
        """
        Detect if current inventory levels indicate overstocking.

        Parameters
        ----------
        current_inventory : float
            Current inventory on hand
        threshold_multiplier : float, optional
            Multiplier for EOQ to determine overstock threshold

        Returns
        -------
        dict
            Overstock analysis
        """
        eoq = self.calculate_eoq()
        rop = self.calculate_reorder_point()
        max_inventory = rop + eoq

        # Overstock threshold (typically 2x EOQ above ROP)
        overstock_threshold = rop + (eoq * threshold_multiplier)

        is_overstocked = current_inventory > overstock_threshold

        # Calculate excess inventory
        excess_inventory = max(0, current_inventory - max_inventory)
        excess_holding_cost = excess_inventory * self.holding_cost

        # Days to consume excess
        days_to_normal = excess_inventory / self.daily_demand if self.daily_demand > 0 else 0

        # Severity level
        if current_inventory > overstock_threshold * 1.5:
            severity = "SEVERE"
        elif is_overstocked:
            severity = "MODERATE"
        else:
            severity = "NONE"

        return {
            'is_overstocked': is_overstocked,
            'current_inventory': current_inventory,
            'optimal_max_inventory': max_inventory,
            'overstock_threshold': overstock_threshold,
            'excess_inventory': excess_inventory,
            'excess_holding_cost': excess_holding_cost,
            'days_to_normal': days_to_normal,
            'severity': severity
        }

    def analyze_lead_time(self, historical_lead_times: List[float]) -> Dict[str, float]:
        """
        Analyze lead time variability and its impact.

        Parameters
        ----------
        historical_lead_times : list
            Historical lead times in days

        Returns
        -------
        dict
            Lead time analysis metrics
        """
        lead_times = np.array(historical_lead_times)

        mean_lead_time = np.mean(lead_times)
        std_lead_time = np.std(lead_times)
        min_lead_time = np.min(lead_times)
        max_lead_time = np.max(lead_times)
        cv_lead_time = std_lead_time / mean_lead_time if mean_lead_time > 0 else 0

        # Calculate impact on safety stock
        base_safety_stock = self.calculate_safety_stock()

        # Adjusted safety stock with lead time variability
        # Using combined variance approach
        demand_variance = self.demand_std ** 2
        lead_time_variance = std_lead_time ** 2

        z_score = stats.norm.ppf(self.service_level)
        adjusted_safety_stock = z_score * np.sqrt(
            mean_lead_time * demand_variance +
            (self.daily_demand ** 2) * lead_time_variance
        )

        safety_stock_increase = adjusted_safety_stock - base_safety_stock
        additional_holding_cost = safety_stock_increase * self.holding_cost

        # Lead time reliability score
        reliability_score = 1 - min(cv_lead_time, 1.0)

        return {
            'mean_lead_time': mean_lead_time,
            'std_lead_time': std_lead_time,
            'min_lead_time': min_lead_time,
            'max_lead_time': max_lead_time,
            'cv_lead_time': cv_lead_time,
            'reliability_score': reliability_score,
            'base_safety_stock': base_safety_stock,
            'adjusted_safety_stock': adjusted_safety_stock,
            'safety_stock_increase': safety_stock_increase,
            'additional_holding_cost': additional_holding_cost
        }

    def calculate_inventory_turnover(
        self,
        annual_sales: Optional[float] = None,
        average_inventory: Optional[float] = None
    ) -> InventoryMetrics:
        """
        Calculate comprehensive inventory turnover metrics.

        Parameters
        ----------
        annual_sales : float, optional
            Annual sales in units (defaults to annual_demand)
        average_inventory : float, optional
            Average inventory level (calculated from EOQ if not provided)

        Returns
        -------
        InventoryMetrics
            Comprehensive inventory performance metrics
        """
        if annual_sales is None:
            annual_sales = self.annual_demand

        if average_inventory is None:
            eoq = self.calculate_eoq()
            safety_stock = self.calculate_safety_stock()
            average_inventory = (eoq / 2) + safety_stock

        # Inventory turnover ratio
        turnover_ratio = annual_sales / average_inventory if average_inventory > 0 else 0

        # Days of inventory
        days_of_inventory = self.working_days / turnover_ratio if turnover_ratio > 0 else float('inf')

        # Stock to sales ratio
        stock_to_sales_ratio = average_inventory / (annual_sales / self.working_days) if annual_sales > 0 else 0

        # Carrying cost percentage
        carrying_cost_percentage = self.holding_cost_rate * 100

        # Stockout rate (based on service level)
        stockout_rate = (1 - self.service_level) * 100

        # Fill rate
        fill_rate = self.service_level * 100

        return InventoryMetrics(
            turnover_ratio=turnover_ratio,
            days_of_inventory=days_of_inventory,
            stock_to_sales_ratio=stock_to_sales_ratio,
            carrying_cost_percentage=carrying_cost_percentage,
            stockout_rate=stockout_rate,
            fill_rate=fill_rate
        )

    def calculate_roi(
        self,
        current_order_quantity: float,
        current_safety_stock: float,
        proposed_order_quantity: Optional[float] = None,
        proposed_safety_stock: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate ROI of implementing optimized ordering policy.

        Parameters
        ----------
        current_order_quantity : float
            Current order quantity
        current_safety_stock : float
            Current safety stock level
        proposed_order_quantity : float, optional
            Proposed order quantity (uses EOQ if not provided)
        proposed_safety_stock : float, optional
            Proposed safety stock (uses optimized if not provided)

        Returns
        -------
        dict
            ROI analysis including cost savings and payback period
        """
        if proposed_order_quantity is None:
            proposed_order_quantity = self.calculate_eoq()

        if proposed_safety_stock is None:
            proposed_safety_stock, _ = self.optimize_safety_stock()

        # Current costs
        current_costs = self.calculate_total_cost(current_order_quantity, current_safety_stock)
        current_total = current_costs['total_cost']

        # Proposed costs
        proposed_costs = self.calculate_total_cost(proposed_order_quantity, proposed_safety_stock)
        proposed_total = proposed_costs['total_cost']

        # Savings
        annual_savings = current_total - proposed_total
        savings_percentage = (annual_savings / current_total * 100) if current_total > 0 else 0

        # Implementation cost estimate (one-time)
        implementation_cost = self.ordering_cost * 2  # Assume 2 orders worth of setup cost

        # ROI calculation
        roi = (annual_savings / implementation_cost * 100) if implementation_cost > 0 else 0

        # Payback period (months)
        payback_months = (implementation_cost / (annual_savings / 12)) if annual_savings > 0 else float('inf')

        return {
            'current_total_cost': current_total,
            'proposed_total_cost': proposed_total,
            'annual_savings': annual_savings,
            'savings_percentage': savings_percentage,
            'implementation_cost': implementation_cost,
            'roi_percentage': roi,
            'payback_months': payback_months,
            'npv_3_years': annual_savings * 3 - implementation_cost
        }

    def generate_recommendations(
        self,
        current_inventory: float,
        current_order_quantity: float,
        current_safety_stock: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Parameters
        ----------
        current_inventory : float
            Current inventory level
        current_order_quantity : float
            Current order quantity
        current_safety_stock : float
            Current safety stock level

        Returns
        -------
        list
            List of recommendations
        """
        recommendations = []

        # EOQ comparison
        eoq = self.calculate_eoq()
        if abs(current_order_quantity - eoq) / eoq > 0.10:
            diff_pct = ((eoq - current_order_quantity) / current_order_quantity) * 100
            recommendations.append(
                f"Adjust order quantity from {current_order_quantity:.0f} to {eoq:.0f} units "
                f"({diff_pct:+.1f}%) to minimize total costs"
            )

        # Safety stock comparison
        optimal_ss, _ = self.optimize_safety_stock()
        if abs(current_safety_stock - optimal_ss) / optimal_ss > 0.10:
            diff_pct = ((optimal_ss - current_safety_stock) / current_safety_stock) * 100
            recommendations.append(
                f"Adjust safety stock from {current_safety_stock:.0f} to {optimal_ss:.0f} units "
                f"({diff_pct:+.1f}%) to balance service level and costs"
            )

        # Stockout risk
        risk = self.assess_stockout_risk(current_inventory)
        if risk['risk_level'] in ['HIGH', 'CRITICAL']:
            recommendations.append(
                f"URGENT: Place order immediately. Stockout risk is {risk['risk_level']} "
                f"with {risk['days_remaining']:.1f} days of inventory remaining"
            )
        elif risk['should_reorder']:
            recommendations.append(
                f"Reorder point reached. Place order for {eoq:.0f} units"
            )

        # Overstock detection
        overstock = self.detect_overstock(current_inventory)
        if overstock['is_overstocked']:
            recommendations.append(
                f"Reduce inventory by {overstock['excess_inventory']:.0f} units to avoid "
                f"excess holding costs of ${overstock['excess_holding_cost']:.2f}/year"
            )

        # Turnover metrics
        metrics = self.calculate_inventory_turnover()
        if metrics.turnover_ratio < 4:
            recommendations.append(
                f"Low inventory turnover ({metrics.turnover_ratio:.2f}x/year). "
                f"Consider reducing order quantities or safety stock"
            )
        elif metrics.turnover_ratio > 12:
            recommendations.append(
                f"High inventory turnover ({metrics.turnover_ratio:.2f}x/year). "
                f"Monitor for potential stockouts and consider increasing safety stock"
            )

        # Service level
        if metrics.stockout_rate > 10:
            recommendations.append(
                f"Stockout rate of {metrics.stockout_rate:.1f}% exceeds 10% threshold. "
                f"Increase safety stock or service level target"
            )

        return recommendations

    def optimize(
        self,
        current_inventory: float,
        current_order_quantity: float,
        current_safety_stock: float
    ) -> OrderOptimizationResult:
        """
        Perform complete order optimization analysis.

        Parameters
        ----------
        current_inventory : float
            Current inventory level
        current_order_quantity : float
            Current order quantity
        current_safety_stock : float
            Current safety stock level

        Returns
        -------
        OrderOptimizationResult
            Complete optimization results with recommendations
        """
        # Calculate optimal values
        eoq = self.calculate_eoq()
        optimal_ss, costs = self.optimize_safety_stock()
        rop = self.calculate_reorder_point(optimal_ss)
        order_freq = self.calculate_order_frequency(eoq)

        # Calculate metrics
        metrics = self.calculate_inventory_turnover()

        # Calculate ROI
        roi_analysis = self.calculate_roi(
            current_order_quantity,
            current_safety_stock,
            eoq,
            optimal_ss
        )

        # Generate recommendations
        recommendations = self.generate_recommendations(
            current_inventory,
            current_order_quantity,
            current_safety_stock
        )

        return OrderOptimizationResult(
            eoq=eoq,
            reorder_point=rop,
            safety_stock=optimal_ss,
            order_frequency=order_freq,
            total_annual_cost=costs['total_cost'],
            ordering_cost=costs['ordering_cost'],
            holding_cost=costs['holding_cost'],
            stockout_cost=costs['stockout_cost'],
            roi=roi_analysis['roi_percentage'],
            metrics=metrics,
            recommendations=recommendations
        )


def analyze_multi_product_optimization(
    products_data: pd.DataFrame,
    ordering_cost: float,
    holding_cost_rate: float,
    service_level: Union[float, ServiceLevel] = ServiceLevel.MEDIUM
) -> pd.DataFrame:
    """
    Perform order optimization analysis for multiple products.

    Parameters
    ----------
    products_data : pd.DataFrame
        DataFrame with columns: product_id, annual_demand, unit_cost, lead_time,
        demand_std, current_inventory, current_order_qty, current_safety_stock
    ordering_cost : float
        Fixed cost per order
    holding_cost_rate : float
        Annual holding cost rate
    service_level : float or ServiceLevel
        Desired service level

    Returns
    -------
    pd.DataFrame
        Optimization results for all products
    """
    results = []

    for _, row in products_data.iterrows():
        analyzer = OrderOptimizationAnalyzer(
            annual_demand=row['annual_demand'],
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_cost_rate,
            unit_cost=row['unit_cost'],
            lead_time=row['lead_time'],
            demand_std=row.get('demand_std'),
            service_level=service_level
        )

        optimization = analyzer.optimize(
            current_inventory=row['current_inventory'],
            current_order_quantity=row['current_order_qty'],
            current_safety_stock=row['current_safety_stock']
        )

        results.append({
            'product_id': row['product_id'],
            'eoq': optimization.eoq,
            'reorder_point': optimization.reorder_point,
            'safety_stock': optimization.safety_stock,
            'order_frequency': optimization.order_frequency,
            'total_annual_cost': optimization.total_annual_cost,
            'roi_percentage': optimization.roi,
            'turnover_ratio': optimization.metrics.turnover_ratio,
            'days_of_inventory': optimization.metrics.days_of_inventory,
            'fill_rate': optimization.metrics.fill_rate,
            'recommendations': '; '.join(optimization.recommendations)
        })

    return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Single product optimization
    print("=" * 80)
    print("EXAMPLE 1: Single Product Optimization")
    print("=" * 80)

    analyzer = OrderOptimizationAnalyzer(
        annual_demand=10000,
        ordering_cost=50,
        holding_cost_rate=0.25,
        unit_cost=20,
        lead_time=7,
        demand_std=5,
        service_level=ServiceLevel.HIGH
    )

    result = analyzer.optimize(
        current_inventory=500,
        current_order_quantity=1000,
        current_safety_stock=100
    )

    print(f"\nOptimal Order Quantity (EOQ): {result.eoq:.0f} units")
    print(f"Reorder Point: {result.reorder_point:.0f} units")
    print(f"Safety Stock: {result.safety_stock:.0f} units")
    print(f"Order Frequency: {result.order_frequency:.1f} times/year")
    print(f"\nCost Breakdown:")
    print(f"  Ordering Cost: ${result.ordering_cost:,.2f}")
    print(f"  Holding Cost: ${result.holding_cost:,.2f}")
    print(f"  Stockout Cost: ${result.stockout_cost:,.2f}")
    print(f"  Total Annual Cost: ${result.total_annual_cost:,.2f}")
    print(f"\nROI: {result.roi:.1f}%")
    print(f"\nInventory Metrics:")
    print(f"  Turnover Ratio: {result.metrics.turnover_ratio:.2f}x")
    print(f"  Days of Inventory: {result.metrics.days_of_inventory:.1f} days")
    print(f"  Fill Rate: {result.metrics.fill_rate:.1f}%")
    print(f"\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")

    # Example 2: Stockout risk assessment
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Stockout Risk Assessment")
    print("=" * 80)

    risk = analyzer.assess_stockout_risk(current_inventory=150, pending_orders=200)
    print(f"\nCurrent Inventory: {risk['current_inventory']:.0f} units")
    print(f"Pending Orders: {risk['pending_orders']:.0f} units")
    print(f"Effective Inventory: {risk['effective_inventory']:.0f} units")
    print(f"Reorder Point: {risk['reorder_point']:.0f} units")
    print(f"Days Remaining: {risk['days_remaining']:.1f} days")
    print(f"Stockout Probability: {risk['stockout_probability']:.1%}")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Should Reorder: {'YES' if risk['should_reorder'] else 'NO'}")

    # Example 3: Lead time analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Lead Time Variability Analysis")
    print("=" * 80)

    historical_lead_times = [6, 7, 8, 7, 9, 6, 7, 10, 7, 8]
    lead_time_analysis = analyzer.analyze_lead_time(historical_lead_times)

    print(f"\nMean Lead Time: {lead_time_analysis['mean_lead_time']:.1f} days")
    print(f"Std Dev: {lead_time_analysis['std_lead_time']:.2f} days")
    print(f"Range: {lead_time_analysis['min_lead_time']:.0f} - {lead_time_analysis['max_lead_time']:.0f} days")
    print(f"Coefficient of Variation: {lead_time_analysis['cv_lead_time']:.2%}")
    print(f"Reliability Score: {lead_time_analysis['reliability_score']:.2%}")
    print(f"Additional Safety Stock Needed: {lead_time_analysis['safety_stock_increase']:.0f} units")
    print(f"Additional Holding Cost: ${lead_time_analysis['additional_holding_cost']:.2f}/year")
