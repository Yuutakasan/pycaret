"""
Unit Tests for Alert Engine

Tests cover:
- Rule-based alert generation
- ML-based anomaly detection
- Threshold monitoring and escalation
- Priority ranking
- Recommendation generation
- Alert aggregation
- Notification scheduling
- Alert history tracking
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine.alert_engine import (
    AlertEngine, Alert, AlertType, AlertSeverity, AlertStatus,
    AlertRule, AnomalyDetector
)


class TestAnomalyDetector(unittest.TestCase):
    """Test ML-based anomaly detection"""

    def setUp(self):
        self.detector = AnomalyDetector(contamination=0.1)

    def test_fit_and_detect(self):
        """Test training and detection"""
        # Generate normal data
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, (100, 3))

        # Fit detector
        self.detector.fit(normal_data)
        self.assertTrue(self.detector.is_fitted)

        # Generate test data with anomalies
        test_data = np.vstack([
            np.random.normal(100, 10, (90, 3)),
            np.random.normal(200, 10, (10, 3))  # Anomalies
        ])

        predictions, scores = self.detector.detect(test_data)

        # Should detect some anomalies
        anomaly_count = np.sum(predictions == -1)
        self.assertGreater(anomaly_count, 0)
        self.assertLess(anomaly_count, len(test_data))

    def test_statistical_detection(self):
        """Test z-score based detection"""
        values = np.array([10, 12, 11, 13, 10, 100, 11, 12])  # 100 is anomaly

        anomalies = self.detector.detect_statistical(values, z_threshold=2.0)

        # Should detect the outlier
        self.assertTrue(anomalies[5])  # Index 5 has value 100

    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        small_data = np.random.normal(100, 10, (3, 3))

        # Should not fit
        self.detector.fit(small_data)
        self.assertFalse(self.detector.is_fitted)

        # Should return no anomalies
        values = np.array([1, 2])
        anomalies = self.detector.detect_statistical(values)
        self.assertEqual(len(anomalies), 2)
        self.assertFalse(anomalies.any())


class TestAlertRule(unittest.TestCase):
    """Test alert rule evaluation"""

    def test_rule_evaluation(self):
        """Test rule condition evaluation"""
        rule = AlertRule(
            rule_id="test_rule",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            condition="value < threshold",
            threshold=50.0,
            comparison="lt",
            lookback_period=timedelta(hours=1)
        )

        # Test less than
        self.assertTrue(rule.evaluate(30.0))
        self.assertFalse(rule.evaluate(60.0))

    def test_comparison_operators(self):
        """Test all comparison operators"""
        rule = AlertRule(
            rule_id="test",
            alert_type=AlertType.WASTE_RISK,
            severity=AlertSeverity.HIGH,
            condition="test",
            threshold=50.0,
            comparison="gt",
            lookback_period=timedelta(hours=1)
        )

        # Greater than
        rule.comparison = "gt"
        self.assertTrue(rule.evaluate(60.0))
        self.assertFalse(rule.evaluate(40.0))

        # Less than or equal
        rule.comparison = "lte"
        self.assertTrue(rule.evaluate(50.0))
        self.assertTrue(rule.evaluate(40.0))
        self.assertFalse(rule.evaluate(60.0))

        # Greater than or equal
        rule.comparison = "gte"
        self.assertTrue(rule.evaluate(50.0))
        self.assertTrue(rule.evaluate(60.0))
        self.assertFalse(rule.evaluate(40.0))

    def test_disabled_rule(self):
        """Test that disabled rules don't trigger"""
        rule = AlertRule(
            rule_id="test",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            condition="test",
            threshold=50.0,
            comparison="lt",
            lookback_period=timedelta(hours=1),
            enabled=False
        )

        # Should not trigger even if condition met
        self.assertFalse(rule.evaluate(30.0))


class TestAlertEngine(unittest.TestCase):
    """Test main alert engine functionality"""

    def setUp(self):
        """Set up test environment"""
        # Use temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.engine = AlertEngine(db_path=self.temp_db.name)

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_inventory_alerts_critical(self):
        """Test critical inventory shortage alerts"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "PROD001",
                "stock_level": 10,
                "safety_stock": 50,
                "lead_time_days": 3
            }
        ])

        alerts = self.engine.check_inventory_alerts(inventory_data)

        # Should generate critical alert (stock < 50% of safety stock)
        self.assertGreater(len(alerts), 0)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)

        # Check alert details
        alert = critical_alerts[0]
        self.assertEqual(alert.alert_type, AlertType.INVENTORY_SHORTAGE)
        self.assertIn("PROD001", alert.title)
        self.assertGreater(len(alert.recommendations), 0)

    def test_inventory_alerts_low(self):
        """Test low inventory alerts"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "PROD002",
                "stock_level": 45,
                "safety_stock": 50,
                "lead_time_days": 5
            }
        ])

        alerts = self.engine.check_inventory_alerts(inventory_data)

        # Should generate high severity alert
        self.assertGreater(len(alerts), 0)
        high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        self.assertGreater(len(high_alerts), 0)

    def test_waste_alerts_critical(self):
        """Test critical waste risk alerts"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "PROD003",
                "stock_level": 100,
                "safety_stock": 50,
                "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=1),
                "lead_time_days": 3
            }
        ])

        alerts = self.engine.check_waste_alerts(inventory_data)

        # Should generate critical waste alert
        self.assertGreater(len(alerts), 0)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)

        # Check recommendations include urgent actions
        alert = critical_alerts[0]
        self.assertIn("URGENT", " ".join(alert.recommendations))

    def test_waste_alerts_warning(self):
        """Test waste warning alerts"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "PROD004",
                "stock_level": 50,
                "safety_stock": 40,
                "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=5),
                "lead_time_days": 3
            }
        ])

        alerts = self.engine.check_waste_alerts(inventory_data)

        # Should generate medium severity alert
        self.assertGreater(len(alerts), 0)
        medium_alerts = [a for a in alerts if a.severity == AlertSeverity.MEDIUM]
        self.assertGreater(len(medium_alerts), 0)

    def test_sales_anomaly_detection(self):
        """Test ML-based sales anomaly detection"""
        # Generate sales history
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='H')
        sales_data = pd.DataFrame([
            {
                "product_id": "PROD005",
                "quantity": np.random.normal(100, 10),
                "timestamp": date
            }
            for date in dates
        ])

        # Add anomaly
        sales_data.loc[len(sales_data)] = {
            "product_id": "PROD005",
            "quantity": 300,  # Spike
            "timestamp": pd.Timestamp.now()
        }

        alerts = self.engine.detect_sales_anomalies(sales_data)

        # Should detect the anomaly
        self.assertGreater(len(alerts), 0)

    def test_demand_spike_detection(self):
        """Test demand spike detection"""
        sales_data = pd.DataFrame([
            {
                "product_id": "PROD006",
                "quantity": 150,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=1)
            }
        ])

        forecast_data = pd.DataFrame([
            {
                "product_id": "PROD006",
                "predicted_demand": 50
            }
        ])

        alerts = self.engine.check_demand_patterns(sales_data, forecast_data)

        # Should detect spike
        self.assertGreater(len(alerts), 0)
        spike_alerts = [a for a in alerts if a.alert_type == AlertType.DEMAND_SPIKE]
        self.assertGreater(len(spike_alerts), 0)

    def test_demand_drop_detection(self):
        """Test demand drop detection"""
        sales_data = pd.DataFrame([
            {
                "product_id": "PROD007",
                "quantity": 10,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=1)
            }
        ])

        forecast_data = pd.DataFrame([
            {
                "product_id": "PROD007",
                "predicted_demand": 100
            }
        ])

        alerts = self.engine.check_demand_patterns(sales_data, forecast_data)

        # Should detect drop
        self.assertGreater(len(alerts), 0)
        drop_alerts = [a for a in alerts if a.alert_type == AlertType.DEMAND_DROP]
        self.assertGreater(len(drop_alerts), 0)

    def test_alert_prioritization(self):
        """Test alert priority ranking"""
        alerts = [
            Alert(
                alert_id="1",
                alert_type=AlertType.INVENTORY_SHORTAGE,
                severity=AlertSeverity.CRITICAL,
                title="Test 1",
                message="Critical",
                entity_id="P1",
                entity_type="product",
                timestamp=datetime.now(),
                confidence=0.95
            ),
            Alert(
                alert_id="2",
                alert_type=AlertType.WASTE_RISK,
                severity=AlertSeverity.MEDIUM,
                title="Test 2",
                message="Medium",
                entity_id="P2",
                entity_type="product",
                timestamp=datetime.now() - timedelta(hours=5),
                confidence=0.7
            ),
            Alert(
                alert_id="3",
                alert_type=AlertType.DEMAND_SPIKE,
                severity=AlertSeverity.HIGH,
                title="Test 3",
                message="High",
                entity_id="P3",
                entity_type="product",
                timestamp=datetime.now() - timedelta(minutes=30),
                confidence=0.85
            )
        ]

        prioritized = self.engine.prioritize_alerts(alerts)

        # Critical should be first
        self.assertEqual(prioritized[0].severity, AlertSeverity.CRITICAL)

        # All should have priority scores
        for alert in prioritized:
            self.assertIn('priority_score', alert.metadata)

    def test_alert_aggregation(self):
        """Test alert aggregation and summarization"""
        alerts = []
        for i in range(10):
            alerts.append(Alert(
                alert_id=f"alert_{i}",
                alert_type=AlertType.INVENTORY_SHORTAGE if i % 2 == 0 else AlertType.WASTE_RISK,
                severity=AlertSeverity.CRITICAL if i < 3 else AlertSeverity.MEDIUM,
                title=f"Test {i}",
                message=f"Message {i}",
                entity_id=f"P{i}",
                entity_type="product",
                timestamp=datetime.now() - timedelta(minutes=i*5)
            ))

        summary = self.engine.aggregate_alerts(alerts, timedelta(hours=1))

        # Check summary structure
        self.assertEqual(summary['total_alerts'], 10)
        self.assertIn('by_type', summary)
        self.assertIn('by_severity', summary)
        self.assertEqual(summary['critical_count'], 3)

    def test_escalation_check(self):
        """Test alert escalation logic"""
        # Critical alert not acknowledged for 2 hours
        old_critical = Alert(
            alert_id="old_critical",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            title="Old Critical",
            message="Test",
            entity_id="P1",
            entity_type="product",
            timestamp=datetime.now() - timedelta(hours=2),
            status=AlertStatus.NEW
        )

        should_escalate = self.engine.check_escalation(old_critical)
        self.assertTrue(should_escalate)

        # Recent critical alert
        new_critical = Alert(
            alert_id="new_critical",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.CRITICAL,
            title="New Critical",
            message="Test",
            entity_id="P2",
            entity_type="product",
            timestamp=datetime.now() - timedelta(minutes=30),
            status=AlertStatus.NEW
        )

        should_escalate = self.engine.check_escalation(new_critical)
        self.assertFalse(should_escalate)

    def test_notification_scheduling(self):
        """Test notification scheduling by severity"""
        alerts = [
            Alert(
                alert_id="1",
                alert_type=AlertType.INVENTORY_SHORTAGE,
                severity=AlertSeverity.CRITICAL,
                title="Critical",
                message="Test",
                entity_id="P1",
                entity_type="product",
                timestamp=datetime.now()
            ),
            Alert(
                alert_id="2",
                alert_type=AlertType.WASTE_RISK,
                severity=AlertSeverity.LOW,
                title="Low",
                message="Test",
                entity_id="P2",
                entity_type="product",
                timestamp=datetime.now()
            )
        ]

        scheduled = self.engine.schedule_notifications(alerts)

        # Critical should be immediate
        self.assertIn('immediate', scheduled)
        self.assertIn('scheduled', scheduled)
        self.assertIn('deferred', scheduled)

        # Critical alerts should be in immediate
        critical_in_immediate = any(
            a.severity == AlertSeverity.CRITICAL
            for a in scheduled['immediate']
        )
        self.assertTrue(critical_in_immediate)

    def test_alert_persistence(self):
        """Test alert saving and retrieval"""
        alert = Alert(
            alert_id="persist_test",
            alert_type=AlertType.INVENTORY_SHORTAGE,
            severity=AlertSeverity.HIGH,
            title="Persistence Test",
            message="Testing database save",
            entity_id="PROD999",
            entity_type="product",
            timestamp=datetime.now(),
            metadata={"test": "value"},
            recommendations=["Action 1", "Action 2"]
        )

        # Save alert
        self.engine.save_alert(alert)

        # Retrieve alert
        history = self.engine.get_alert_history(entity_id="PROD999")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['alert_id'], "persist_test")
        self.assertEqual(history[0]['entity_id'], "PROD999")

    def test_alert_status_update(self):
        """Test alert status updates"""
        alert = Alert(
            alert_id="status_test",
            alert_type=AlertType.WASTE_RISK,
            severity=AlertSeverity.MEDIUM,
            title="Status Test",
            message="Testing status update",
            entity_id="PROD888",
            entity_type="product",
            timestamp=datetime.now()
        )

        # Save alert
        self.engine.save_alert(alert)

        # Update to acknowledged
        self.engine.update_alert_status("status_test", AlertStatus.ACKNOWLEDGED)

        # Retrieve and check
        history = self.engine.get_alert_history(entity_id="PROD888")
        self.assertEqual(history[0]['status'], AlertStatus.ACKNOWLEDGED.value)
        self.assertIsNotNone(history[0]['acknowledged_at'])

    def test_cooldown_mechanism(self):
        """Test alert cooldown to prevent spam"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "PROD_COOLDOWN",
                "stock_level": 10,
                "safety_stock": 50,
                "lead_time_days": 3
            }
        ])

        # First check should generate alert
        alerts1 = self.engine.check_inventory_alerts(inventory_data)
        self.assertGreater(len(alerts1), 0)

        # Immediate second check should not generate (cooldown active)
        alerts2 = self.engine.check_inventory_alerts(inventory_data)
        self.assertEqual(len(alerts2), 0)

    def test_full_check_integration(self):
        """Test complete alert check workflow"""
        inventory_data = pd.DataFrame([
            {
                "product_id": "FULL_001",
                "stock_level": 15,
                "safety_stock": 50,
                "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=1),
                "lead_time_days": 3
            }
        ])

        sales_data = pd.DataFrame([
            {
                "product_id": "FULL_001",
                "quantity": 50,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=i)
            }
            for i in range(10)
        ])

        forecast_data = pd.DataFrame([
            {
                "product_id": "FULL_001",
                "predicted_demand": 50
            }
        ])

        results = self.engine.run_full_check(inventory_data, sales_data, forecast_data)

        # Check result structure
        self.assertIn('total_alerts', results)
        self.assertIn('prioritized_alerts', results)
        self.assertIn('escalations', results)
        self.assertIn('scheduled_notifications', results)
        self.assertIn('summary', results)

        # Should have generated some alerts
        self.assertGreater(results['total_alerts'], 0)


if __name__ == '__main__':
    unittest.main()
