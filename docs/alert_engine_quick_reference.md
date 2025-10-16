# Alert Engine Quick Reference

## Essential Commands

### Initialize Engine
```python
from engine.alert_engine import AlertEngine
engine = AlertEngine(db_path=".swarm/alerts.db")
```

### Run Full Check
```python
results = engine.run_full_check(
    inventory_data=df_inventory,
    sales_data=df_sales,
    forecast_data=df_forecast  # Optional
)
```

### Check Specific Alerts
```python
# Inventory only
inventory_alerts = engine.check_inventory_alerts(inventory_df)

# Waste only
waste_alerts = engine.check_waste_alerts(inventory_df)

# Sales anomalies
anomaly_alerts = engine.detect_sales_anomalies(sales_df)

# Demand patterns
demand_alerts = engine.check_demand_patterns(sales_df, forecast_df)
```

### Manage Alerts
```python
# Save alert
engine.save_alert(alert)

# Update status
engine.update_alert_status(alert_id, AlertStatus.ACKNOWLEDGED)

# Get history
history = engine.get_alert_history(entity_id="PROD001", limit=50)
```

### Prioritize & Aggregate
```python
# Prioritize alerts
prioritized = engine.prioritize_alerts(alerts)

# Aggregate summary
summary = engine.aggregate_alerts(alerts, timedelta(hours=1))

# Schedule notifications
scheduled = engine.schedule_notifications(alerts)
```

## Alert Severity Levels

| Severity | Value | Use Case | Response Time |
|----------|-------|----------|---------------|
| CRITICAL | 4 | Immediate action required | < 1 hour |
| HIGH | 3 | Action required soon | < 4 hours |
| MEDIUM | 2 | Action needed | < 24 hours |
| LOW | 1 | Monitor situation | < 48 hours |
| INFO | 0 | Informational only | As convenient |

## Alert Types

| Type | Trigger | Example |
|------|---------|---------|
| INVENTORY_SHORTAGE | Stock < Safety Stock | "Stock level 15 < Safety 50" |
| WASTE_RISK | Days to expiry < 7 | "Expires in 2 days" |
| DEMAND_SPIKE | Sales > 2x Predicted | "Sales 200 vs predicted 80" |
| DEMAND_DROP | Sales < 30% Predicted | "Sales 20 vs predicted 100" |
| SALES_ANOMALY | ML detected outlier | "Unusual pattern detected" |

## Alert Status Flow

```
NEW → ACKNOWLEDGED → IN_PROGRESS → RESOLVED
  ↓
DISMISSED / ESCALATED
```

## Default Rules

| Rule ID | Type | Severity | Threshold | Cooldown |
|---------|------|----------|-----------|----------|
| inventory_critical | Shortage | CRITICAL | <50% safety | 6h |
| inventory_low | Shortage | HIGH | <100% safety | 12h |
| waste_critical | Waste | CRITICAL | <2 days expiry | 12h |
| waste_warning | Waste | MEDIUM | <7 days expiry | 24h |
| demand_spike | Spike | HIGH | >2x predicted | 8h |
| demand_drop | Drop | MEDIUM | <30% predicted | 8h |

## Custom Rule Example

```python
from engine.alert_engine import AlertRule, AlertType, AlertSeverity
from datetime import timedelta

rule = AlertRule(
    rule_id="my_custom_rule",
    alert_type=AlertType.INVENTORY_SHORTAGE,
    severity=AlertSeverity.HIGH,
    condition="stock < safety_stock * 0.75",
    threshold=0.75,
    comparison="lt",
    lookback_period=timedelta(hours=2),
    cooldown_period=timedelta(hours=6)
)

engine.add_rule(rule)
```

## Data Format Requirements

### Inventory Data
```python
pd.DataFrame({
    "product_id": str,        # Required
    "stock_level": float,     # Required
    "safety_stock": float,    # Required
    "expiry_date": datetime,  # Optional (for waste alerts)
    "lead_time_days": int     # Optional
})
```

### Sales Data
```python
pd.DataFrame({
    "product_id": str,     # Required
    "quantity": float,     # Required
    "timestamp": datetime  # Required
})
```

### Forecast Data
```python
pd.DataFrame({
    "product_id": str,           # Required
    "predicted_demand": float    # Required
})
```

## Result Structure

```python
{
    "total_alerts": int,
    "prioritized_alerts": [alert_dict, ...],
    "escalations": [alert_dict, ...],
    "scheduled_notifications": {
        "immediate": [alert_dict, ...],
        "scheduled": [alert_dict, ...],
        "deferred": [alert_dict, ...]
    },
    "summary": {
        "total_alerts": int,
        "by_type": {type: count},
        "by_severity": {severity: count},
        "critical_count": int,
        "high_count": int,
        "top_alerts": [alert_dict, ...],
        "affected_entities": int
    }
}
```

## Alert Dictionary Structure

```python
{
    "alert_id": str,
    "alert_type": str,
    "severity": int,
    "title": str,
    "message": str,
    "entity_id": str,
    "entity_type": str,
    "timestamp": str (ISO format),
    "status": str,
    "confidence": float,
    "metadata": dict,
    "recommendations": [str, ...],
    "escalation_count": int,
    "acknowledged_at": str or None,
    "resolved_at": str or None
}
```

## Common Patterns

### Daily Monitoring
```python
def daily_alert_check():
    engine = AlertEngine()
    results = engine.run_full_check(
        inventory_data=get_current_inventory(),
        sales_data=get_last_24h_sales(),
        forecast_data=get_today_forecast()
    )

    # Send critical alerts immediately
    for alert in results['scheduled_notifications']['immediate']:
        send_notification(alert)

    # Generate daily report
    generate_report(results['summary'])
```

### Alert Response
```python
def handle_alert(alert_id):
    engine = AlertEngine()

    # Acknowledge
    engine.update_alert_status(alert_id, AlertStatus.ACKNOWLEDGED)

    # Get details
    history = engine.get_alert_history(limit=1)
    alert = history[0]

    # Execute recommendations
    for action in alert['recommendations']:
        execute_action(action)

    # Mark resolved
    engine.update_alert_status(alert_id, AlertStatus.RESOLVED)
```

### Performance Monitoring
```python
def monitor_alert_performance():
    engine = AlertEngine()
    history = engine.get_alert_history(
        start_date=datetime.now() - timedelta(days=30),
        limit=1000
    )

    # Calculate metrics
    total = len(history)
    resolved = sum(1 for h in history if h['status'] == 'resolved')
    avg_time = calculate_avg_resolution_time(history)

    print(f"Resolution rate: {resolved/total*100:.1f}%")
    print(f"Avg resolution time: {avg_time:.1f} hours")
```

## ML Anomaly Detection

### Train Detector
```python
from engine.alert_engine import AnomalyDetector

detector = AnomalyDetector(contamination=0.1)

# Prepare features: [quantity, hour, day_of_week]
features = historical_data[['quantity', 'hour', 'day_of_week']].values

detector.fit(features)
```

### Detect Anomalies
```python
predictions, scores = detector.detect(new_features)

# Find anomalies (-1 = anomaly, 1 = normal)
anomaly_indices = np.where(predictions == -1)[0]
```

### Statistical Detection
```python
values = data['quantity'].values
anomalies = detector.detect_statistical(values, z_threshold=3.0)
```

## Integration Snippets

### Flask REST API
```python
@app.route('/alerts/check', methods=['POST'])
def check():
    data = request.json
    results = engine.run_full_check(
        pd.DataFrame(data['inventory']),
        pd.DataFrame(data['sales'])
    )
    return jsonify(results)
```

### Slack Webhook
```python
def notify_slack(alert):
    requests.post(SLACK_WEBHOOK, json={
        "text": f"*{alert['severity']}*: {alert['title']}",
        "attachments": [{
            "text": alert['message'],
            "color": "danger" if alert['severity'] == 'CRITICAL' else "warning"
        }]
    })
```

### Scheduled Job (Cron)
```bash
# Run every hour
0 * * * * python -c "from scripts import run_alerts; run_alerts()"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No alerts generated | Check data completeness, rule thresholds |
| Too many alerts | Increase cooldown, adjust thresholds |
| ML not detecting | Need >7 days training data |
| Database locked | Close connections, check concurrent access |
| Missing recommendations | Verify alert type handlers |

## Files

- **Engine**: `/mnt/d/github/pycaret/src/engine/alert_engine.py`
- **Tests**: `/mnt/d/github/pycaret/tests/test_alert_engine.py`
- **Examples**: `/mnt/d/github/pycaret/examples/alert_engine_examples.py`
- **Guide**: `/mnt/d/github/pycaret/docs/alert_engine_guide.md`
- **Database**: `.swarm/alerts.db` (auto-created)
