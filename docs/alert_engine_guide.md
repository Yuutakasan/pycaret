# Alert Engine User Guide

## Overview

The Alert Engine is an intelligent monitoring and recommendation system that combines rule-based alerts with ML-based anomaly detection to proactively identify and respond to inventory, waste, and sales issues.

## Key Features

### 1. Rule-Based Alert System
- **Inventory Shortage**: Critical and low inventory warnings
- **Waste Risk**: Expiry-based alerts with urgency levels
- **Demand Patterns**: Spike and drop detection
- **Cost Anomalies**: Unusual pricing patterns
- **Supply Chain**: Lead time and delivery issues

### 2. ML-Based Anomaly Detection
- **Isolation Forest**: Multivariate anomaly detection
- **Statistical Methods**: Z-score based outlier detection
- **Time Series Analysis**: Pattern recognition in sales data
- **Adaptive Learning**: Continuous model improvement

### 3. Threshold Monitoring with Escalation
- **Severity Levels**: CRITICAL, HIGH, MEDIUM, LOW, INFO
- **Auto-Escalation**: Time-based escalation for unresolved alerts
- **Cooldown Periods**: Prevent alert spam
- **Status Tracking**: NEW, ACKNOWLEDGED, IN_PROGRESS, RESOLVED

### 4. Priority Ranking System
- **Multi-Factor Scoring**: Severity + Confidence + Recency
- **Automatic Prioritization**: Intelligent alert ordering
- **Entity Grouping**: Track alerts by product/store/category

### 5. Actionable Recommendations
- **Context-Aware**: Tailored to alert type and severity
- **Step-by-Step**: Clear action items
- **Escalation Paths**: What to do if situation worsens
- **Resource Links**: Relevant contacts and procedures

### 6. Alert Aggregation
- **Time-Window Summaries**: Hourly, daily, weekly views
- **Type Grouping**: By alert type and severity
- **Trend Analysis**: Pattern identification
- **Executive Dashboards**: High-level overviews

### 7. Notification Scheduling
- **Time Windows**: Severity-based notification hours
- **Batching**: Group related alerts
- **Escalation Routing**: Auto-route to appropriate teams
- **Multi-Channel**: Email, SMS, Slack, webhooks

### 8. Alert History Tracking
- **SQLite Database**: Persistent storage
- **Audit Trail**: Complete alert lifecycle
- **Query Interface**: Flexible retrieval
- **Performance Metrics**: Resolution times and patterns

## Quick Start

```python
from engine.alert_engine import AlertEngine
import pandas as pd

# Initialize engine
engine = AlertEngine(db_path=".swarm/alerts.db")

# Prepare inventory data
inventory_data = pd.DataFrame([
    {
        "product_id": "PROD001",
        "stock_level": 15,
        "safety_stock": 50,
        "expiry_date": pd.Timestamp.now() + pd.Timedelta(days=2),
        "lead_time_days": 3
    }
])

# Prepare sales data
sales_data = pd.DataFrame([
    {
        "product_id": "PROD001",
        "quantity": 25,
        "timestamp": pd.Timestamp.now() - pd.Timedelta(hours=i)
    }
    for i in range(24)
])

# Run comprehensive check
results = engine.run_full_check(
    inventory_data=inventory_data,
    sales_data=sales_data
)

# Access results
print(f"Total alerts: {results['total_alerts']}")
print(f"Critical alerts: {results['summary']['critical_count']}")

# Get prioritized alerts
for alert in results['prioritized_alerts'][:5]:
    print(f"{alert['severity']}: {alert['title']}")
    print(f"Recommendations: {alert['recommendations']}")
```

## Alert Types

### Inventory Shortage
Triggered when stock levels fall below safety thresholds.

**Critical**: Stock < 50% of safety stock
- Emergency reorder required
- Expedited shipping recommended
- Hourly monitoring until replenishment

**High**: Stock < safety stock
- Standard reorder recommended
- Review demand forecast
- Consider increasing safety stock

### Waste Risk
Triggered by approaching expiry dates.

**Critical**: Expires in < 2 days
- Immediate 50%+ discount
- Move to front of shelf
- Consider donation within 24h

**Medium**: Expires in < 7 days
- Apply promotional discount (15-30%)
- Feature in weekly promotions
- Increase visibility

### Sales Anomalies
ML-detected unusual patterns.

**Demand Spike**: Sales > 2x predicted
- Verify inventory sufficiency
- Emergency reorder if needed
- Investigate cause (event, viral trend)

**Demand Drop**: Sales < 30% predicted
- Review pricing competitiveness
- Check quality issues
- Analyze seasonality

## Custom Rules

```python
from engine.alert_engine import AlertRule, AlertType, AlertSeverity
from datetime import timedelta

# Create custom rule
custom_rule = AlertRule(
    rule_id="custom_high_value_shortage",
    alert_type=AlertType.INVENTORY_SHORTAGE,
    severity=AlertSeverity.CRITICAL,
    condition="stock_level < safety_stock AND product_value > 1000",
    threshold=1.0,
    comparison="lt",
    lookback_period=timedelta(hours=2),
    cooldown_period=timedelta(hours=4)
)

# Add to engine
engine.add_rule(custom_rule)
```

## ML Anomaly Detection

```python
from engine.alert_engine import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(contamination=0.1)

# Train on historical data
import numpy as np
historical_features = np.random.normal(100, 10, (1000, 3))
detector.fit(historical_features)

# Detect anomalies in new data
new_data = np.random.normal(100, 10, (100, 3))
predictions, scores = detector.detect(new_data)

# Anomalies marked as -1
anomaly_indices = np.where(predictions == -1)[0]
print(f"Detected {len(anomaly_indices)} anomalies")
```

## Alert Lifecycle

1. **Generation**: Rule evaluation or ML detection
2. **Prioritization**: Automatic ranking by severity + confidence
3. **Persistence**: Saved to SQLite database
4. **Notification**: Scheduled based on severity and time windows
5. **Acknowledgment**: Team member acknowledges receipt
6. **Investigation**: Status updated to IN_PROGRESS
7. **Resolution**: Actions taken, alert resolved
8. **Analysis**: Review for process improvement

## Status Management

```python
from engine.alert_engine import AlertStatus

# Acknowledge alert
engine.update_alert_status("alert_123", AlertStatus.ACKNOWLEDGED)

# Mark as in progress
engine.update_alert_status("alert_123", AlertStatus.IN_PROGRESS)

# Resolve alert
engine.update_alert_status("alert_123", AlertStatus.RESOLVED)
```

## Query Alert History

```python
from datetime import datetime, timedelta
from engine.alert_engine import AlertType

# Get recent critical alerts
history = engine.get_alert_history(
    alert_type=AlertType.INVENTORY_SHORTAGE,
    start_date=datetime.now() - timedelta(days=7),
    limit=50
)

# Filter by entity
product_alerts = engine.get_alert_history(
    entity_id="PROD001",
    limit=100
)

# Analyze patterns
for alert in history:
    print(f"{alert['timestamp']}: {alert['title']}")
    print(f"Status: {alert['status']}")
    print(f"Resolution time: {alert.get('resolved_at', 'Pending')}")
```

## Escalation Configuration

```python
# Check if alert needs escalation
alert = results['prioritized_alerts'][0]
needs_escalation = engine.check_escalation(
    alert,
    max_age=timedelta(hours=24)
)

if needs_escalation:
    # Escalate to management
    engine.update_alert_status(alert['alert_id'], AlertStatus.ESCALATED)
    # Send escalation notification
    send_management_notification(alert)
```

## Notification Scheduling

```python
from engine.alert_engine import AlertSeverity

# Define custom time windows
time_windows = {
    AlertSeverity.CRITICAL: list(range(24)),     # 24/7
    AlertSeverity.HIGH: list(range(6, 23)),       # 6 AM - 11 PM
    AlertSeverity.MEDIUM: list(range(8, 20)),     # 8 AM - 8 PM
    AlertSeverity.LOW: [9, 14, 17],               # Business hours only
}

# Schedule notifications
scheduled = engine.schedule_notifications(
    alerts=results['prioritized_alerts'],
    time_windows=time_windows
)

# Process immediate alerts
for alert in scheduled['immediate']:
    send_notification(alert)

# Queue scheduled alerts
for alert in scheduled['scheduled']:
    queue_notification(alert)
```

## Alert Aggregation

```python
from datetime import timedelta

# Get hourly summary
summary = engine.aggregate_alerts(
    alerts=results['prioritized_alerts'],
    time_window=timedelta(hours=1)
)

print(f"Total alerts: {summary['total_alerts']}")
print(f"By type: {summary['by_type']}")
print(f"By severity: {summary['by_severity']}")
print(f"Affected entities: {summary['affected_entities']}")

# Top 5 most critical
for alert in summary['top_alerts']:
    print(f"- {alert['title']} (Score: {alert['metadata']['priority_score']})")
```

## Performance Considerations

### Database Optimization
- Indexed on: alert_type, entity_id, timestamp, status
- Regular cleanup of old resolved alerts
- Partition by date for large datasets

### ML Model Training
- Minimum 7 days of data recommended
- Retrain weekly or when data patterns shift
- Monitor model performance metrics

### Alert Volume Management
- Use cooldown periods to prevent spam
- Aggregate similar alerts
- Set appropriate severity thresholds

## Integration Examples

### REST API Integration
```python
from flask import Flask, jsonify

app = Flask(__name__)
engine = AlertEngine()

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    data = request.get_json()
    results = engine.run_full_check(
        inventory_data=pd.DataFrame(data['inventory']),
        sales_data=pd.DataFrame(data['sales'])
    )
    return jsonify(results)

@app.route('/api/alerts/history/<entity_id>')
def get_history(entity_id):
    history = engine.get_alert_history(entity_id=entity_id)
    return jsonify(history)
```

### Webhook Notifications
```python
import requests

def send_webhook_notification(alert):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    payload = {
        "text": f"*{alert['severity']}*: {alert['title']}",
        "attachments": [{
            "text": alert['message'],
            "fields": [
                {"title": "Entity", "value": alert['entity_id'], "short": True},
                {"title": "Type", "value": alert['alert_type'], "short": True}
            ],
            "color": "danger" if alert['severity'] == 'CRITICAL' else "warning"
        }]
    }

    requests.post(webhook_url, json=payload)
```

### Email Notifications
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(alert, recipients):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[{alert['severity']}] {alert['title']}"
    msg['From'] = "alerts@example.com"
    msg['To'] = ", ".join(recipients)

    # Create HTML body
    html = f"""
    <html>
      <body>
        <h2 style="color: {'red' if alert['severity'] == 'CRITICAL' else 'orange'}">
          {alert['title']}
        </h2>
        <p>{alert['message']}</p>
        <h3>Recommendations:</h3>
        <ul>
          {''.join(f'<li>{rec}</li>' for rec in alert['recommendations'])}
        </ul>
        <p><small>Alert ID: {alert['alert_id']}</small></p>
      </body>
    </html>
    """

    msg.attach(MIMEText(html, 'html'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login("your-email@gmail.com", "your-password")
        server.send_message(msg)
```

## Best Practices

1. **Regular Monitoring**: Run full checks every 15-60 minutes
2. **Tune Thresholds**: Adjust based on business metrics
3. **Review History**: Weekly analysis of alert patterns
4. **Update Rules**: Refine rules based on false positives
5. **Train Models**: Regular ML model updates
6. **Test Notifications**: Verify delivery channels
7. **Document Actions**: Track resolution procedures
8. **Measure Impact**: Monitor waste reduction, stockout prevention

## Troubleshooting

### No Alerts Generated
- Verify data quality and completeness
- Check rule thresholds
- Ensure rules are enabled
- Review cooldown settings

### Too Many Alerts
- Increase cooldown periods
- Adjust severity thresholds
- Enable aggregation
- Review rule conditions

### ML Detection Issues
- Ensure sufficient training data (>7 days)
- Check feature engineering
- Adjust contamination parameter
- Retrain models regularly

### Database Performance
- Add indexes for custom queries
- Archive old alerts
- Optimize query patterns
- Consider PostgreSQL for large scale

## API Reference

See inline documentation in `/mnt/d/github/pycaret/src/engine/alert_engine.py` for detailed API reference.

## Support

For issues and questions:
- Check test cases: `/mnt/d/github/pycaret/tests/test_alert_engine.py`
- Review source code: `/mnt/d/github/pycaret/src/engine/alert_engine.py`
- Contact: backend-team@example.com
