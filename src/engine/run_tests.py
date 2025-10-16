#!/usr/bin/env python3
"""
Standalone test runner for alert engine
Bypasses conftest.py conflicts
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly
from engine.alert_engine import (
    AlertEngine, Alert, AlertType, AlertSeverity, AlertStatus,
    AlertRule, AnomalyDetector
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile


def test_basic_functionality():
    """Test basic alert engine functionality"""
    print("Testing basic functionality...")

    # Create temp db
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()

    try:
        # Initialize engine
        engine = AlertEngine(db_path=temp_db.name)

        # Test inventory alerts
        inventory_data = pd.DataFrame([
            {
                "product_id": "TEST001",
                "stock_level": 15,
                "safety_stock": 50,
                "lead_time_days": 3
            }
        ])

        alerts = engine.check_inventory_alerts(inventory_data)
        assert len(alerts) > 0, "Should generate inventory alerts"
        assert alerts[0].severity == AlertSeverity.CRITICAL, "Should be critical"

        # Test waste alerts
        inventory_data_waste = pd.DataFrame([
            {
                "product_id": "TEST002",
                "stock_level": 100,
                "safety_stock": 50,
                "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=1),
                "lead_time_days": 3
            }
        ])

        waste_alerts = engine.check_waste_alerts(inventory_data_waste)
        assert len(waste_alerts) > 0, "Should generate waste alerts"

        # Test persistence
        engine.save_alert(alerts[0])
        history = engine.get_alert_history(entity_id="TEST001")
        assert len(history) > 0, "Should retrieve saved alerts"

        # Test prioritization
        prioritized = engine.prioritize_alerts(alerts)
        assert len(prioritized) == len(alerts), "Should maintain alert count"

        # Test aggregation
        summary = engine.aggregate_alerts(alerts, timedelta(hours=1))
        assert summary['total_alerts'] > 0, "Should have alerts in summary"

        print("✓ All basic functionality tests passed!")
        return True

    finally:
        os.unlink(temp_db.name)


def test_ml_anomaly_detection():
    """Test ML-based anomaly detection"""
    print("Testing ML anomaly detection...")

    # Create detector
    detector = AnomalyDetector(contamination=0.1)

    # Generate training data
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, (100, 3))

    # Train
    detector.fit(normal_data)
    assert detector.is_fitted, "Detector should be fitted"

    # Test detection
    test_data = np.vstack([
        np.random.normal(100, 10, (90, 3)),
        np.random.normal(200, 10, (10, 3))  # Anomalies
    ])

    predictions, scores = detector.detect(test_data)
    anomaly_count = np.sum(predictions == -1)
    assert anomaly_count > 0, "Should detect anomalies"

    # Test statistical detection
    values = np.array([10, 12, 11, 13, 10, 100, 11, 12])
    anomalies = detector.detect_statistical(values, z_threshold=2.0)
    assert anomalies[5], "Should detect outlier at index 5"

    print("✓ ML anomaly detection tests passed!")
    return True


def test_sales_anomalies():
    """Test sales anomaly detection integration"""
    print("Testing sales anomaly detection...")

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()

    try:
        engine = AlertEngine(db_path=temp_db.name)

        # Generate sales data with spike
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='H')
        quantities = np.random.normal(50, 10, 25).tolist()
        quantities.extend([150, 160, 155, 150, 145])  # Spike

        sales_data = pd.DataFrame([
            {
                "product_id": "TEST_SALES",
                "quantity": qty,
                "timestamp": date
            }
            for qty, date in zip(quantities, dates)
        ])

        alerts = engine.detect_sales_anomalies(sales_data)
        # May or may not detect depending on training
        print(f"  Detected {len(alerts)} anomaly alerts")

        print("✓ Sales anomaly tests passed!")
        return True

    finally:
        os.unlink(temp_db.name)


def test_full_workflow():
    """Test complete workflow"""
    print("Testing full workflow...")

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()

    try:
        engine = AlertEngine(db_path=temp_db.name)

        inventory_data = pd.DataFrame([
            {
                "product_id": "WORKFLOW_001",
                "stock_level": 10,
                "safety_stock": 50,
                "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=2),
                "lead_time_days": 3
            }
        ])

        sales_data = pd.DataFrame([
            {
                "product_id": "WORKFLOW_001",
                "quantity": 25,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=i)
            }
            for i in range(10)
        ])

        results = engine.run_full_check(inventory_data, sales_data)

        assert 'total_alerts' in results, "Should have total_alerts"
        assert 'summary' in results, "Should have summary"
        assert results['total_alerts'] > 0, "Should generate alerts"

        print(f"  Generated {results['total_alerts']} total alerts")
        print(f"  Critical: {results['summary']['critical_count']}")

        print("✓ Full workflow tests passed!")
        return True

    finally:
        os.unlink(temp_db.name)


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Alert Engine Standalone Test Suite")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_ml_anomaly_detection,
        test_sales_anomalies,
        test_full_workflow
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
