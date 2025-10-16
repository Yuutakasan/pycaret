"""
Alert Engine Usage Examples

Demonstrates various use cases and integration patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine.alert_engine import (
    AlertEngine, Alert, AlertType, AlertSeverity, AlertStatus,
    AlertRule, AnomalyDetector
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def example_1_basic_usage():
    """Example 1: Basic alert checking"""
    print("=" * 60)
    print("Example 1: Basic Alert Checking")
    print("=" * 60)

    # Initialize engine
    engine = AlertEngine(db_path="examples_alerts.db")

    # Sample inventory data with issues
    inventory_data = pd.DataFrame([
        {
            "product_id": "MILK_001",
            "stock_level": 12,  # Low stock
            "safety_stock": 50,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=1),  # Expiring soon
            "lead_time_days": 2
        },
        {
            "product_id": "BREAD_001",
            "stock_level": 45,
            "safety_stock": 50,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=5),
            "lead_time_days": 1
        }
    ])

    # Check inventory alerts
    inventory_alerts = engine.check_inventory_alerts(inventory_data)
    waste_alerts = engine.check_waste_alerts(inventory_data)

    print(f"\nInventory Alerts: {len(inventory_alerts)}")
    for alert in inventory_alerts:
        print(f"  - {alert.severity.name}: {alert.title}")
        print(f"    Recommendations: {alert.recommendations[0]}")

    print(f"\nWaste Alerts: {len(waste_alerts)}")
    for alert in waste_alerts:
        print(f"  - {alert.severity.name}: {alert.title}")
        print(f"    Days to expiry: {alert.metadata['days_to_expiry']}")


def example_2_ml_anomaly_detection():
    """Example 2: ML-based anomaly detection"""
    print("\n" + "=" * 60)
    print("Example 2: ML-Based Anomaly Detection")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Generate realistic sales history with anomaly
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')

    # Normal sales pattern with some noise
    quantities = np.random.normal(50, 10, 95).tolist()

    # Add anomalies
    quantities.extend([150, 160, 155, 150, 145])  # Sudden spike

    sales_data = pd.DataFrame([
        {
            "product_id": "COFFEE_001",
            "quantity": qty,
            "timestamp": date
        }
        for qty, date in zip(quantities, dates)
    ])

    # Detect anomalies
    anomaly_alerts = engine.detect_sales_anomalies(sales_data)

    print(f"\nDetected {len(anomaly_alerts)} sales anomalies")
    for alert in anomaly_alerts:
        print(f"  - {alert.title}")
        print(f"    Current: {alert.metadata['current_quantity']:.0f}, "
              f"Avg: {alert.metadata['avg_quantity']:.0f}")
        print(f"    Confidence: {alert.confidence:.2f}")
        print(f"    Top recommendation: {alert.recommendations[0]}")


def example_3_demand_forecasting_integration():
    """Example 3: Integration with demand forecasting"""
    print("\n" + "=" * 60)
    print("Example 3: Demand Forecasting Integration")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Recent sales data
    sales_data = pd.DataFrame([
        {
            "product_id": "CHIPS_001",
            "quantity": 180,  # Much higher than predicted
            "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=2)
        },
        {
            "product_id": "SODA_001",
            "quantity": 15,  # Much lower than predicted
            "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=1)
        }
    ])

    # Forecast predictions
    forecast_data = pd.DataFrame([
        {"product_id": "CHIPS_001", "predicted_demand": 80},
        {"product_id": "SODA_001", "predicted_demand": 100}
    ])

    # Check demand patterns
    demand_alerts = engine.check_demand_patterns(sales_data, forecast_data)

    print(f"\nDemand Pattern Alerts: {len(demand_alerts)}")
    for alert in demand_alerts:
        print(f"  - {alert.alert_type.value}: {alert.entity_id}")
        print(f"    {alert.message}")
        print(f"    Recommendations:")
        for rec in alert.recommendations[:3]:
            print(f"      • {rec}")


def example_4_alert_prioritization():
    """Example 4: Alert prioritization and aggregation"""
    print("\n" + "=" * 60)
    print("Example 4: Alert Prioritization and Aggregation")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Create mix of alerts
    alerts = [
        Alert(
            alert_id=f"alert_{i}",
            alert_type=np.random.choice([AlertType.INVENTORY_SHORTAGE, AlertType.WASTE_RISK]),
            severity=np.random.choice([AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM]),
            title=f"Alert {i}",
            message=f"Test message {i}",
            entity_id=f"PROD_{i:03d}",
            entity_type="product",
            timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 120)),
            confidence=np.random.uniform(0.7, 1.0)
        )
        for i in range(20)
    ]

    # Prioritize
    prioritized = engine.prioritize_alerts(alerts)

    print("\nTop 5 Prioritized Alerts:")
    for alert in prioritized[:5]:
        score = alert.metadata.get('priority_score', 0)
        print(f"  {alert.entity_id}: {alert.severity.name} "
              f"(Score: {score:.1f}, Confidence: {alert.confidence:.2f})")

    # Aggregate
    summary = engine.aggregate_alerts(alerts, timedelta(hours=2))

    print(f"\nAlert Summary (Last 2 hours):")
    print(f"  Total: {summary['total_alerts']}")
    print(f"  By Type: {summary['by_type']}")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  Affected Entities: {summary['affected_entities']}")


def example_5_custom_rules():
    """Example 5: Creating custom alert rules"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Alert Rules")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Create custom rules for high-value products
    high_value_rule = AlertRule(
        rule_id="high_value_critical",
        alert_type=AlertType.INVENTORY_SHORTAGE,
        severity=AlertSeverity.CRITICAL,
        condition="stock_level < safety_stock * 0.8",
        threshold=0.8,
        comparison="lt",
        lookback_period=timedelta(hours=1),
        cooldown_period=timedelta(hours=3)
    )

    engine.add_rule(high_value_rule)

    # Test with high-value product
    inventory_data = pd.DataFrame([
        {
            "product_id": "PREMIUM_WINE_001",
            "stock_level": 35,
            "safety_stock": 50,
            "product_value": 150.00,
            "lead_time_days": 7
        }
    ])

    alerts = engine.check_inventory_alerts(inventory_data)

    print(f"\nCustom Rule Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert.title}")
        print(f"    Triggered by rule: high_value_critical")
        print(f"    {alert.message}")


def example_6_escalation_workflow():
    """Example 6: Alert escalation workflow"""
    print("\n" + "=" * 60)
    print("Example 6: Alert Escalation Workflow")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Create old unacknowledged critical alert
    old_alert = Alert(
        alert_id="critical_001",
        alert_type=AlertType.INVENTORY_SHORTAGE,
        severity=AlertSeverity.CRITICAL,
        title="Critical Stock Shortage - Emergency Item",
        message="Emergency medical supplies critically low",
        entity_id="MED_SUPPLIES_001",
        entity_type="product",
        timestamp=datetime.now() - timedelta(hours=3),
        status=AlertStatus.NEW
    )

    # Save alert
    engine.save_alert(old_alert)

    # Check if escalation needed
    needs_escalation = engine.check_escalation(old_alert)

    print(f"\nAlert: {old_alert.title}")
    print(f"Status: {old_alert.status.value}")
    print(f"Age: {(datetime.now() - old_alert.timestamp).total_seconds() / 3600:.1f} hours")
    print(f"Needs Escalation: {needs_escalation}")

    if needs_escalation:
        print("\nEscalating alert...")
        engine.update_alert_status(old_alert.alert_id, AlertStatus.ESCALATED)
        print("  - Notified senior management")
        print("  - Triggered emergency procurement")
        print("  - Updated status to ESCALATED")


def example_7_notification_scheduling():
    """Example 7: Notification scheduling by severity"""
    print("\n" + "=" * 60)
    print("Example 7: Notification Scheduling")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Create alerts of different severities
    alerts = [
        Alert(
            alert_id="n1",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message="Immediate action required",
            entity_id="P1",
            entity_type="product",
            timestamp=datetime.now()
        ),
        Alert(
            alert_id="n2",
            alert_type=AlertType.WASTE_RISK,
            severity=AlertSeverity.MEDIUM,
            title="Medium Alert",
            message="Review needed",
            entity_id="P2",
            entity_type="product",
            timestamp=datetime.now()
        ),
        Alert(
            alert_id="n3",
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.LOW,
            title="Low Alert",
            message="FYI notification",
            entity_id="P3",
            entity_type="product",
            timestamp=datetime.now()
        )
    ]

    # Schedule notifications
    scheduled = engine.schedule_notifications(alerts)

    print(f"\nCurrent hour: {datetime.now().hour}")
    print(f"Immediate notifications: {len(scheduled['immediate'])}")
    for alert in scheduled['immediate']:
        print(f"  - {alert.title} ({alert.severity.name})")

    print(f"\nScheduled notifications: {len(scheduled['scheduled'])}")
    for alert in scheduled['scheduled']:
        print(f"  - {alert.title} ({alert.severity.name})")

    print(f"\nDeferred notifications: {len(scheduled['deferred'])}")
    for alert in scheduled['deferred']:
        print(f"  - {alert.title} ({alert.severity.name})")


def example_8_full_workflow():
    """Example 8: Complete workflow with all features"""
    print("\n" + "=" * 60)
    print("Example 8: Complete Workflow")
    print("=" * 60)

    engine = AlertEngine(db_path="examples_alerts.db")

    # Comprehensive inventory data
    inventory_data = pd.DataFrame([
        {
            "product_id": "YOGURT_001",
            "stock_level": 10,
            "safety_stock": 40,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=2),
            "lead_time_days": 1
        },
        {
            "product_id": "CHEESE_001",
            "stock_level": 55,
            "safety_stock": 60,
            "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=14),
            "lead_time_days": 3
        }
    ])

    # Sales history with patterns
    sales_data = pd.DataFrame([
        {
            "product_id": "YOGURT_001",
            "quantity": np.random.normal(20, 5),
            "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=i)
        }
        for i in range(48)
    ])

    # Forecast data
    forecast_data = pd.DataFrame([
        {"product_id": "YOGURT_001", "predicted_demand": 20},
        {"product_id": "CHEESE_001", "predicted_demand": 15}
    ])

    # Run full check
    results = engine.run_full_check(inventory_data, sales_data, forecast_data)

    print(f"\n{'='*60}")
    print("ALERT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Alerts Generated: {results['total_alerts']}")
    print(f"Critical: {results['summary']['critical_count']}")
    print(f"High: {results['summary']['high_count']}")
    print(f"Affected Products: {results['summary']['affected_entities']}")

    print(f"\n{'='*60}")
    print("TOP 3 PRIORITY ALERTS")
    print(f"{'='*60}")
    for i, alert in enumerate(results['prioritized_alerts'][:3], 1):
        print(f"\n{i}. {alert['title']}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Entity: {alert['entity_id']}")
        print(f"   Message: {alert['message']}")
        print(f"   Recommendations:")
        for rec in alert['recommendations'][:2]:
            print(f"     • {rec}")

    print(f"\n{'='*60}")
    print("NOTIFICATION SCHEDULE")
    print(f"{'='*60}")
    immediate = results['scheduled_notifications']['immediate']
    scheduled = results['scheduled_notifications']['scheduled']
    print(f"Send Immediately: {len(immediate)} alerts")
    print(f"Send Later: {len(scheduled)} alerts")

    if results['escalations']:
        print(f"\n{'='*60}")
        print("ESCALATIONS REQUIRED")
        print(f"{'='*60}")
        for alert in results['escalations']:
            print(f"  - {alert['title']} (Age: {alert['metadata'].get('age_hours', 'N/A')} hours)")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_ml_anomaly_detection()
    example_3_demand_forecasting_integration()
    example_4_alert_prioritization()
    example_5_custom_rules()
    example_6_escalation_workflow()
    example_7_notification_scheduling()
    example_8_full_workflow()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

    # Cleanup
    import os
    if os.path.exists("examples_alerts.db"):
        os.remove("examples_alerts.db")
        print("\nCleaned up example database")
