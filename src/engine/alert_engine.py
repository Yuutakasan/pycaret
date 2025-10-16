"""
Intelligent Alert and Recommendation Engine

Features:
- Rule-based alerts (inventory shortage, waste risk, sales anomalies)
- ML-based anomaly detection (Isolation Forest, Statistical methods)
- Threshold monitoring with escalation
- Priority ranking system
- Actionable recommendation generation
- Alert aggregation and summarization
- Notification scheduling
- Alert history tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import sqlite3
from pathlib import Path


class AlertType(Enum):
    """Alert type classification"""
    INVENTORY_SHORTAGE = "inventory_shortage"
    WASTE_RISK = "waste_risk"
    SALES_ANOMALY = "sales_anomaly"
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    EXPIRY_WARNING = "expiry_warning"
    COST_ANOMALY = "cost_anomaly"
    SUPPLY_CHAIN = "supply_chain"
    QUALITY_ISSUE = "quality_issue"
    SEASONAL_PATTERN = "seasonal_pattern"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = 4  # Immediate action required
    HIGH = 3      # Action required within hours
    MEDIUM = 2    # Action required within days
    LOW = 1       # Informational, monitor
    INFO = 0      # General information


class AlertStatus(Enum):
    """Alert lifecycle status"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    entity_id: str  # Product ID, Store ID, etc.
    entity_type: str  # "product", "store", "category"
    timestamp: datetime
    status: AlertStatus = AlertStatus.NEW
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    escalation_count: int = 0
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "recommendations": self.recommendations,
            "escalation_count": self.escalation_count,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AlertRule:
    """Rule-based alert configuration"""
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Python expression
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    lookback_period: timedelta
    cooldown_period: timedelta = timedelta(hours=1)
    enabled: bool = True

    def evaluate(self, value: float) -> bool:
        """Evaluate rule condition"""
        if not self.enabled:
            return False

        comparisons = {
            "gt": value > self.threshold,
            "gte": value >= self.threshold,
            "lt": value < self.threshold,
            "lte": value <= self.threshold,
            "eq": abs(value - self.threshold) < 1e-6
        }
        return comparisons.get(self.comparison, False)


class AnomalyDetector:
    """ML-based anomaly detection"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        """Train anomaly detection model"""
        if len(data) < 10:
            return

        scaled_data = self.scaler.fit_transform(data)
        self.isolation_forest.fit(scaled_data)
        self.is_fitted = True

    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies
        Returns: (predictions, anomaly_scores)
        predictions: -1 for anomaly, 1 for normal
        """
        if not self.is_fitted:
            return np.ones(len(data)), np.zeros(len(data))

        scaled_data = self.scaler.transform(data)
        predictions = self.isolation_forest.predict(scaled_data)
        scores = self.isolation_forest.score_samples(scaled_data)
        return predictions, scores

    def detect_statistical(self, values: np.ndarray,
                          z_threshold: float = 3.0) -> np.ndarray:
        """Statistical anomaly detection using z-score"""
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool)

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros(len(values), dtype=bool)

        z_scores = np.abs((values - mean) / std)
        return z_scores > z_threshold


class AlertEngine:
    """Main alert and recommendation engine"""

    def __init__(self, db_path: str = ".swarm/alerts.db"):
        self.db_path = db_path
        self.rules: Dict[str, AlertRule] = {}
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, datetime] = {}

        # Initialize database
        self._init_database()

        # Load default rules
        self._load_default_rules()

    def _init_database(self):
        """Initialize SQLite database for alert history"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                alert_type TEXT NOT NULL,
                severity INTEGER NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                metadata TEXT,
                recommendations TEXT,
                escalation_count INTEGER DEFAULT 0,
                acknowledged_at TEXT,
                resolved_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alert_type
            ON alerts(alert_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity
            ON alerts(entity_id, entity_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON alerts(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON alerts(status)
        """)

        conn.commit()
        conn.close()

    def _load_default_rules(self):
        """Load default alert rules"""

        # Inventory shortage rules
        self.add_rule(AlertRule(
            rule_id="inventory_critical",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            condition="stock_level < safety_stock * 0.5",
            threshold=0.5,
            comparison="lt",
            lookback_period=timedelta(hours=1),
            cooldown_period=timedelta(hours=6)
        ))

        self.add_rule(AlertRule(
            rule_id="inventory_low",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.HIGH,
            condition="stock_level < safety_stock",
            threshold=1.0,
            comparison="lt",
            lookback_period=timedelta(hours=6),
            cooldown_period=timedelta(hours=12)
        ))

        # Waste risk rules
        self.add_rule(AlertRule(
            rule_id="waste_critical",
            alert_type=AlertType.WASTE_RISK,
            severity=AlertSeverity.CRITICAL,
            condition="days_to_expiry < 2",
            threshold=2.0,
            comparison="lt",
            lookback_period=timedelta(days=1),
            cooldown_period=timedelta(hours=12)
        ))

        self.add_rule(AlertRule(
            rule_id="waste_warning",
            alert_type=AlertType.WASTE_RISK,
            severity=AlertSeverity.MEDIUM,
            condition="days_to_expiry < 7",
            threshold=7.0,
            comparison="lt",
            lookback_period=timedelta(days=1),
            cooldown_period=timedelta(days=1)
        ))

        # Demand spike rule
        self.add_rule(AlertRule(
            rule_id="demand_spike",
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.HIGH,
            condition="current_demand > avg_demand * 2.0",
            threshold=2.0,
            comparison="gt",
            lookback_period=timedelta(hours=4),
            cooldown_period=timedelta(hours=8)
        ))

        # Demand drop rule
        self.add_rule(AlertRule(
            rule_id="demand_drop",
            alert_type=AlertType.DEMAND_DROP,
            severity=AlertSeverity.MEDIUM,
            condition="current_demand < avg_demand * 0.3",
            threshold=0.3,
            comparison="lt",
            lookback_period=timedelta(hours=4),
            cooldown_period=timedelta(hours=8)
        ))

    def add_rule(self, rule: AlertRule):
        """Add or update alert rule"""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        self.rules.pop(rule_id, None)

    def check_inventory_alerts(self, inventory_data: pd.DataFrame) -> List[Alert]:
        """Check for inventory-related alerts"""
        alerts = []
        current_time = datetime.now()

        for _, row in inventory_data.iterrows():
            product_id = row.get('product_id', row.get('item_id', 'unknown'))
            stock_level = row.get('stock_level', row.get('quantity', 0))
            safety_stock = row.get('safety_stock', stock_level * 0.2)

            # Check critical shortage
            rule = self.rules.get('inventory_critical')
            if rule and stock_level < safety_stock * rule.threshold:
                if self._check_cooldown(f"inv_critical_{product_id}"):
                    alert = Alert(
                        alert_id=f"inv_critical_{product_id}_{int(current_time.timestamp())}",
                        alert_type=AlertType.INVENTORY_SHORTAGE,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical Inventory Shortage: {product_id}",
                        message=f"Stock level ({stock_level:.0f}) is critically low (<50% of safety stock {safety_stock:.0f})",
                        entity_id=product_id,
                        entity_type="product",
                        timestamp=current_time,
                        metadata={
                            "stock_level": stock_level,
                            "safety_stock": safety_stock,
                            "shortage_ratio": stock_level / safety_stock if safety_stock > 0 else 0
                        }
                    )
                    alert.recommendations = self._generate_inventory_recommendations(row, alert.severity)
                    alerts.append(alert)
                    self._update_cooldown(f"inv_critical_{product_id}", rule.cooldown_period)

            # Check low inventory
            rule = self.rules.get('inventory_low')
            if rule and stock_level < safety_stock:
                if self._check_cooldown(f"inv_low_{product_id}"):
                    alert = Alert(
                        alert_id=f"inv_low_{product_id}_{int(current_time.timestamp())}",
                        alert_type=AlertType.INVENTORY_SHORTAGE,
                        severity=AlertSeverity.HIGH,
                        title=f"Low Inventory Warning: {product_id}",
                        message=f"Stock level ({stock_level:.0f}) below safety stock ({safety_stock:.0f})",
                        entity_id=product_id,
                        entity_type="product",
                        timestamp=current_time,
                        metadata={
                            "stock_level": stock_level,
                            "safety_stock": safety_stock
                        }
                    )
                    alert.recommendations = self._generate_inventory_recommendations(row, alert.severity)
                    alerts.append(alert)
                    self._update_cooldown(f"inv_low_{product_id}", rule.cooldown_period)

        return alerts

    def check_waste_alerts(self, inventory_data: pd.DataFrame) -> List[Alert]:
        """Check for waste risk alerts"""
        alerts = []
        current_time = datetime.now()

        for _, row in inventory_data.iterrows():
            product_id = row.get('product_id', row.get('item_id', 'unknown'))

            # Calculate days to expiry
            expiry_date = row.get('expiry_date', row.get('expiration_date'))
            if expiry_date is None:
                continue

            if isinstance(expiry_date, str):
                try:
                    expiry_date = pd.to_datetime(expiry_date)
                except:
                    continue

            days_to_expiry = (expiry_date - pd.Timestamp.now()).days
            stock_level = row.get('stock_level', row.get('quantity', 0))

            # Check critical waste risk
            rule = self.rules.get('waste_critical')
            if rule and days_to_expiry < rule.threshold:
                if self._check_cooldown(f"waste_critical_{product_id}"):
                    alert = Alert(
                        alert_id=f"waste_critical_{product_id}_{int(current_time.timestamp())}",
                        alert_type=AlertType.WASTE_RISK,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical Waste Risk: {product_id}",
                        message=f"Product expires in {days_to_expiry} days with {stock_level:.0f} units in stock",
                        entity_id=product_id,
                        entity_type="product",
                        timestamp=current_time,
                        metadata={
                            "days_to_expiry": days_to_expiry,
                            "stock_level": stock_level,
                            "expiry_date": expiry_date.isoformat()
                        }
                    )
                    alert.recommendations = self._generate_waste_recommendations(row, days_to_expiry)
                    alerts.append(alert)
                    self._update_cooldown(f"waste_critical_{product_id}", rule.cooldown_period)

            # Check waste warning
            rule = self.rules.get('waste_warning')
            if rule and days_to_expiry < rule.threshold:
                if self._check_cooldown(f"waste_warning_{product_id}"):
                    alert = Alert(
                        alert_id=f"waste_warning_{product_id}_{int(current_time.timestamp())}",
                        alert_type=AlertType.WASTE_RISK,
                        severity=AlertSeverity.MEDIUM,
                        title=f"Waste Warning: {product_id}",
                        message=f"Product expires in {days_to_expiry} days",
                        entity_id=product_id,
                        entity_type="product",
                        timestamp=current_time,
                        metadata={
                            "days_to_expiry": days_to_expiry,
                            "stock_level": stock_level
                        }
                    )
                    alert.recommendations = self._generate_waste_recommendations(row, days_to_expiry)
                    alerts.append(alert)
                    self._update_cooldown(f"waste_warning_{product_id}", rule.cooldown_period)

        return alerts

    def detect_sales_anomalies(self, sales_data: pd.DataFrame) -> List[Alert]:
        """Detect sales anomalies using ML"""
        alerts = []
        current_time = datetime.now()

        # Group by product
        for product_id in sales_data['product_id'].unique():
            product_sales = sales_data[sales_data['product_id'] == product_id].copy()

            if len(product_sales) < 7:  # Need minimum history
                continue

            # Prepare features for anomaly detection
            product_sales['hour'] = pd.to_datetime(product_sales['timestamp']).dt.hour
            product_sales['day_of_week'] = pd.to_datetime(product_sales['timestamp']).dt.dayofweek

            features = product_sales[['quantity', 'hour', 'day_of_week']].values

            # Get or create anomaly detector
            detector_key = f"sales_{product_id}"
            if detector_key not in self.anomaly_detectors:
                self.anomaly_detectors[detector_key] = AnomalyDetector(contamination=0.05)
                self.anomaly_detectors[detector_key].fit(features)

            detector = self.anomaly_detectors[detector_key]

            # Detect anomalies
            predictions, scores = detector.detect(features)

            # Check recent anomalies
            recent_mask = pd.to_datetime(product_sales['timestamp']) > (pd.Timestamp.now() - pd.Timedelta(hours=4))
            recent_anomalies = predictions[recent_mask] == -1

            if recent_anomalies.any():
                recent_data = product_sales[recent_mask]
                anomaly_rows = recent_data[predictions[recent_mask] == -1]

                avg_quantity = product_sales['quantity'].mean()
                current_quantity = anomaly_rows['quantity'].iloc[-1]

                # Determine if spike or drop
                if current_quantity > avg_quantity * 1.5:
                    alert_type = AlertType.DEMAND_SPIKE
                    severity = AlertSeverity.HIGH
                    message = f"Unusual demand spike detected: {current_quantity:.0f} units (avg: {avg_quantity:.0f})"
                else:
                    alert_type = AlertType.DEMAND_DROP
                    severity = AlertSeverity.MEDIUM
                    message = f"Unusual demand drop detected: {current_quantity:.0f} units (avg: {avg_quantity:.0f})"

                if self._check_cooldown(f"sales_anomaly_{product_id}"):
                    alert = Alert(
                        alert_id=f"sales_anomaly_{product_id}_{int(current_time.timestamp())}",
                        alert_type=alert_type,
                        severity=severity,
                        title=f"Sales Anomaly Detected: {product_id}",
                        message=message,
                        entity_id=product_id,
                        entity_type="product",
                        timestamp=current_time,
                        confidence=float(abs(scores[recent_mask][predictions[recent_mask] == -1].mean())),
                        metadata={
                            "current_quantity": float(current_quantity),
                            "avg_quantity": float(avg_quantity),
                            "anomaly_score": float(scores[recent_mask].mean())
                        }
                    )
                    alert.recommendations = self._generate_sales_recommendations(
                        product_id, current_quantity, avg_quantity
                    )
                    alerts.append(alert)
                    self._update_cooldown(f"sales_anomaly_{product_id}", timedelta(hours=6))

        return alerts

    def check_demand_patterns(self, sales_data: pd.DataFrame,
                             forecast_data: pd.DataFrame) -> List[Alert]:
        """Check demand patterns against forecasts"""
        alerts = []
        current_time = datetime.now()

        for _, forecast_row in forecast_data.iterrows():
            product_id = forecast_row['product_id']
            predicted_demand = forecast_row.get('predicted_demand', 0)

            # Get actual recent sales
            recent_sales = sales_data[
                (sales_data['product_id'] == product_id) &
                (pd.to_datetime(sales_data['timestamp']) > pd.Timestamp.now() - pd.Timedelta(hours=4))
            ]

            if recent_sales.empty:
                continue

            actual_demand = recent_sales['quantity'].sum()

            # Check for significant deviation
            if predicted_demand > 0:
                deviation_ratio = actual_demand / predicted_demand

                # Demand spike
                rule = self.rules.get('demand_spike')
                if rule and deviation_ratio > rule.threshold:
                    if self._check_cooldown(f"demand_spike_{product_id}"):
                        alert = Alert(
                            alert_id=f"demand_spike_{product_id}_{int(current_time.timestamp())}",
                            alert_type=AlertType.DEMAND_SPIKE,
                            severity=AlertSeverity.HIGH,
                            title=f"Demand Spike: {product_id}",
                            message=f"Actual demand ({actual_demand:.0f}) is {deviation_ratio:.1f}x predicted ({predicted_demand:.0f})",
                            entity_id=product_id,
                            entity_type="product",
                            timestamp=current_time,
                            metadata={
                                "actual_demand": float(actual_demand),
                                "predicted_demand": float(predicted_demand),
                                "deviation_ratio": float(deviation_ratio)
                            }
                        )
                        alert.recommendations = self._generate_demand_spike_recommendations(
                            product_id, actual_demand, predicted_demand
                        )
                        alerts.append(alert)
                        self._update_cooldown(f"demand_spike_{product_id}", rule.cooldown_period)

                # Demand drop
                rule = self.rules.get('demand_drop')
                if rule and deviation_ratio < rule.threshold:
                    if self._check_cooldown(f"demand_drop_{product_id}"):
                        alert = Alert(
                            alert_id=f"demand_drop_{product_id}_{int(current_time.timestamp())}",
                            alert_type=AlertType.DEMAND_DROP,
                            severity=AlertSeverity.MEDIUM,
                            title=f"Demand Drop: {product_id}",
                            message=f"Actual demand ({actual_demand:.0f}) is only {deviation_ratio:.1%} of predicted ({predicted_demand:.0f})",
                            entity_id=product_id,
                            entity_type="product",
                            timestamp=current_time,
                            metadata={
                                "actual_demand": float(actual_demand),
                                "predicted_demand": float(predicted_demand),
                                "deviation_ratio": float(deviation_ratio)
                            }
                        )
                        alert.recommendations = self._generate_demand_drop_recommendations(
                            product_id, actual_demand, predicted_demand
                        )
                        alerts.append(alert)
                        self._update_cooldown(f"demand_drop_{product_id}", rule.cooldown_period)

        return alerts

    def _generate_inventory_recommendations(self, row: pd.Series,
                                           severity: AlertSeverity) -> List[str]:
        """Generate actionable inventory recommendations"""
        recommendations = []

        stock_level = row.get('stock_level', row.get('quantity', 0))
        safety_stock = row.get('safety_stock', stock_level * 0.2)
        lead_time = row.get('lead_time_days', 3)

        if severity == AlertSeverity.CRITICAL:
            recommendations.extend([
                f"URGENT: Place emergency order immediately",
                f"Consider expedited shipping (lead time: {lead_time} days)",
                "Alert procurement team for priority processing",
                "Monitor stock hourly until replenishment arrives"
            ])
        else:
            reorder_qty = max(safety_stock * 2, stock_level * 3)
            recommendations.extend([
                f"Place reorder for approximately {reorder_qty:.0f} units",
                f"Standard lead time: {lead_time} days",
                "Review demand forecast for next 2 weeks",
                "Consider increasing safety stock levels"
            ])

        return recommendations

    def _generate_waste_recommendations(self, row: pd.Series,
                                       days_to_expiry: int) -> List[str]:
        """Generate waste reduction recommendations"""
        recommendations = []

        stock_level = row.get('stock_level', row.get('quantity', 0))
        product_id = row.get('product_id', row.get('item_id', 'unknown'))

        if days_to_expiry <= 2:
            discount_pct = min(50, (3 - days_to_expiry) * 25)
            recommendations.extend([
                f"URGENT: Apply {discount_pct}% discount immediately",
                "Move to front of shelf (FIFO enforcement)",
                "Promote in store announcements",
                f"Consider donation if not sold within 24 hours",
                "Bundle with complementary products"
            ])
        else:
            discount_pct = 15
            recommendations.extend([
                f"Apply {discount_pct}% promotional discount",
                "Feature in weekly promotions",
                "Increase visibility on shelves",
                "Monitor daily and increase discount if needed"
            ])

        return recommendations

    def _generate_sales_recommendations(self, product_id: str,
                                       current: float,
                                       average: float) -> List[str]:
        """Generate sales-based recommendations"""
        recommendations = []

        if current > average * 1.5:
            recommendations.extend([
                "Review inventory levels immediately",
                "Consider emergency reorder if stock low",
                "Analyze cause of spike (promotion, event, trend)",
                "Adjust forecast models with new data",
                "Monitor competitor activity"
            ])
        else:
            recommendations.extend([
                "Review pricing strategy",
                "Check for quality issues or customer feedback",
                "Analyze competitor promotions",
                "Consider promotional campaign",
                "Adjust future orders to prevent overstock"
            ])

        return recommendations

    def _generate_demand_spike_recommendations(self, product_id: str,
                                              actual: float,
                                              predicted: float) -> List[str]:
        """Generate demand spike recommendations"""
        return [
            f"Verify inventory can support sustained demand",
            f"Place additional order for {actual * 1.5:.0f} units",
            "Investigate root cause (event, promotion, viral trend)",
            "Update forecasting model with spike pattern",
            "Prepare communication for potential stockout",
            "Consider dynamic pricing strategy"
        ]

    def _generate_demand_drop_recommendations(self, product_id: str,
                                             actual: float,
                                             predicted: float) -> List[str]:
        """Generate demand drop recommendations"""
        return [
            "Review pricing competitiveness",
            "Check product quality and reviews",
            "Analyze seasonal patterns",
            "Consider promotional discount",
            "Reduce upcoming orders to prevent overstock",
            "Investigate competitor activity"
        ]

    def _check_cooldown(self, key: str) -> bool:
        """Check if cooldown period has passed"""
        if key not in self.cooldown_tracker:
            return True
        return datetime.now() > self.cooldown_tracker[key]

    def _update_cooldown(self, key: str, period: timedelta):
        """Update cooldown tracker"""
        self.cooldown_tracker[key] = datetime.now() + period

    def prioritize_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Prioritize and rank alerts
        Scoring: severity * 10 + confidence * 5 + recency_factor
        """
        current_time = datetime.now()

        for alert in alerts:
            # Calculate recency factor (more recent = higher priority)
            age_hours = (current_time - alert.timestamp).total_seconds() / 3600
            recency_factor = max(0, 10 - age_hours)  # 10 points for new, decreases over time

            # Calculate priority score
            alert.metadata['priority_score'] = (
                alert.severity.value * 10 +
                alert.confidence * 5 +
                recency_factor
            )

        # Sort by priority score
        return sorted(alerts, key=lambda a: a.metadata.get('priority_score', 0), reverse=True)

    def aggregate_alerts(self, alerts: List[Alert],
                        time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Aggregate and summarize alerts
        """
        current_time = datetime.now()

        # Filter recent alerts
        recent_alerts = [
            a for a in alerts
            if (current_time - a.timestamp) <= time_window
        ]

        # Group by type
        by_type = {}
        for alert in recent_alerts:
            type_key = alert.alert_type.value
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(alert)

        # Group by severity
        by_severity = {}
        for alert in recent_alerts:
            sev_key = alert.severity.name
            if sev_key not in by_severity:
                by_severity[sev_key] = []
            by_severity[sev_key].append(alert)

        # Generate summary
        summary = {
            "total_alerts": len(recent_alerts),
            "time_window_hours": time_window.total_seconds() / 3600,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "critical_count": len(by_severity.get("CRITICAL", [])),
            "high_count": len(by_severity.get("HIGH", [])),
            "top_alerts": [a.to_dict() for a in self.prioritize_alerts(recent_alerts)[:5]],
            "affected_entities": len(set(a.entity_id for a in recent_alerts))
        }

        return summary

    def check_escalation(self, alert: Alert,
                        max_age: timedelta = timedelta(hours=24)) -> bool:
        """
        Check if alert should be escalated
        Returns True if escalation needed
        """
        current_time = datetime.now()
        age = current_time - alert.timestamp

        # Escalation criteria
        should_escalate = False

        # Critical alerts not acknowledged within 1 hour
        if (alert.severity == AlertSeverity.CRITICAL and
            alert.status == AlertStatus.NEW and
            age > timedelta(hours=1)):
            should_escalate = True

        # High alerts not acknowledged within 4 hours
        if (alert.severity == AlertSeverity.HIGH and
            alert.status == AlertStatus.NEW and
            age > timedelta(hours=4)):
            should_escalate = True

        # Any alert not resolved within max_age
        if (alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED] and
            age > max_age):
            should_escalate = True

        return should_escalate

    def schedule_notifications(self, alerts: List[Alert],
                              time_windows: Dict[AlertSeverity, List[int]] = None) -> Dict[str, List[Alert]]:
        """
        Schedule alerts for notification based on severity and time windows
        time_windows: Dict mapping severity to list of hours (24-hour format)
        """
        if time_windows is None:
            time_windows = {
                AlertSeverity.CRITICAL: list(range(24)),  # Anytime
                AlertSeverity.HIGH: list(range(6, 23)),    # 6 AM - 11 PM
                AlertSeverity.MEDIUM: list(range(8, 20)),  # 8 AM - 8 PM
                AlertSeverity.LOW: [9, 14, 17],            # 9 AM, 2 PM, 5 PM
                AlertSeverity.INFO: [9, 17]                # 9 AM, 5 PM
            }

        current_hour = datetime.now().hour

        scheduled = {
            "immediate": [],
            "scheduled": [],
            "deferred": []
        }

        for alert in alerts:
            allowed_hours = time_windows.get(alert.severity, list(range(24)))

            if current_hour in allowed_hours:
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                    scheduled["immediate"].append(alert)
                else:
                    scheduled["scheduled"].append(alert)
            else:
                scheduled["deferred"].append(alert)

        return scheduled

    def save_alert(self, alert: Alert):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.alert_type.value,
            alert.severity.value,
            alert.title,
            alert.message,
            alert.entity_id,
            alert.entity_type,
            alert.timestamp.isoformat(),
            alert.status.value,
            alert.confidence,
            json.dumps(alert.metadata),
            json.dumps(alert.recommendations),
            alert.escalation_count,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            alert.resolved_at.isoformat() if alert.resolved_at else None
        ))

        conn.commit()
        conn.close()

        self.alert_history.append(alert)

    def get_alert_history(self,
                         entity_id: Optional[str] = None,
                         alert_type: Optional[AlertType] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve alert history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)

        if alert_type:
            query += " AND alert_type = ?"
            params.append(alert_type.value)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        results = []

        for row in cursor.fetchall():
            alert_dict = dict(zip(columns, row))
            alert_dict['metadata'] = json.loads(alert_dict['metadata']) if alert_dict['metadata'] else {}
            alert_dict['recommendations'] = json.loads(alert_dict['recommendations']) if alert_dict['recommendations'] else []
            results.append(alert_dict)

        conn.close()
        return results

    def update_alert_status(self, alert_id: str, status: AlertStatus):
        """Update alert status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        current_time = datetime.now()

        update_fields = ["status = ?"]
        params = [status.value]

        if status == AlertStatus.ACKNOWLEDGED:
            update_fields.append("acknowledged_at = ?")
            params.append(current_time.isoformat())
        elif status == AlertStatus.RESOLVED:
            update_fields.append("resolved_at = ?")
            params.append(current_time.isoformat())
        elif status == AlertStatus.ESCALATED:
            update_fields.append("escalation_count = escalation_count + 1")

        params.append(alert_id)

        query = f"UPDATE alerts SET {', '.join(update_fields)} WHERE alert_id = ?"
        cursor.execute(query, params)

        conn.commit()
        conn.close()

    def run_full_check(self,
                      inventory_data: pd.DataFrame,
                      sales_data: pd.DataFrame,
                      forecast_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive alert check across all systems
        """
        all_alerts = []

        # Check inventory alerts
        all_alerts.extend(self.check_inventory_alerts(inventory_data))

        # Check waste alerts
        all_alerts.extend(self.check_waste_alerts(inventory_data))

        # Check sales anomalies
        all_alerts.extend(self.detect_sales_anomalies(sales_data))

        # Check demand patterns if forecast available
        if forecast_data is not None and not forecast_data.empty:
            all_alerts.extend(self.check_demand_patterns(sales_data, forecast_data))

        # Prioritize all alerts
        prioritized_alerts = self.prioritize_alerts(all_alerts)

        # Save to database
        for alert in prioritized_alerts:
            self.save_alert(alert)

        # Check for escalations
        escalation_needed = []
        for alert in prioritized_alerts:
            if self.check_escalation(alert):
                alert.status = AlertStatus.ESCALATED
                alert.escalation_count += 1
                self.update_alert_status(alert.alert_id, AlertStatus.ESCALATED)
                escalation_needed.append(alert)

        # Schedule notifications
        scheduled = self.schedule_notifications(prioritized_alerts)

        # Generate summary
        summary = self.aggregate_alerts(prioritized_alerts)

        return {
            "total_alerts": len(all_alerts),
            "prioritized_alerts": [a.to_dict() for a in prioritized_alerts],
            "escalations": [a.to_dict() for a in escalation_needed],
            "scheduled_notifications": {
                k: [a.to_dict() for a in v]
                for k, v in scheduled.items()
            },
            "summary": summary
        }


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = AlertEngine()

    # Sample inventory data
    inventory_data = pd.DataFrame([
        {
            "product_id": "PROD001",
            "stock_level": 15,
            "safety_stock": 50,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=1),
            "lead_time_days": 3
        },
        {
            "product_id": "PROD002",
            "stock_level": 45,
            "safety_stock": 50,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=10),
            "lead_time_days": 5
        }
    ])

    # Sample sales data
    sales_data = pd.DataFrame([
        {
            "product_id": "PROD001",
            "quantity": 120,
            "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=2)
        },
        {
            "product_id": "PROD001",
            "quantity": 10,
            "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=1)
        }
    ])

    # Run full check
    results = engine.run_full_check(inventory_data, sales_data)

    print(f"Generated {results['total_alerts']} alerts")
    print(f"Critical alerts: {results['summary']['critical_count']}")
    print(f"Escalations needed: {len(results['escalations'])}")
