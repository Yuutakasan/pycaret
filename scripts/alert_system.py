#!/usr/bin/env python3
"""
PyCaret Alert Notification System
Real-time monitoring and alerting for critical events
"""

import os
import sys
import json
import time
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import threading
from queue import Queue

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    DATA_QUALITY = "data_quality"
    SECURITY = "security"
    RESOURCE = "resource"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    type: AlertType
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['level'] = self.level.value
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class AlertStorage:
    """SQLite storage for alerts"""

    def __init__(self, db_path: Path):
        """Initialize alert storage"""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON alerts(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_resolved
                ON alerts(resolved)
            """)

    def save_alert(self, alert: Alert):
        """Save alert to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts
                (id, level, type, title, message, timestamp, source, metadata, resolved, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.level.value,
                alert.type.value,
                alert.title,
                alert.message,
                alert.timestamp.isoformat(),
                alert.source,
                json.dumps(alert.metadata) if alert.metadata else None,
                1 if alert.resolved else 0,
                alert.resolved_at.isoformat() if alert.resolved_at else None
            ))

    def get_recent_alerts(self, hours: int = 24, resolved: Optional[bool] = None) -> List[Alert]:
        """Get recent alerts"""
        since = datetime.now() - timedelta(hours=hours)

        query = "SELECT * FROM alerts WHERE timestamp >= ?"
        params = [since.isoformat()]

        if resolved is not None:
            query += " AND resolved = ?"
            params.append(1 if resolved else 0)

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            alerts = []
            for row in cursor:
                alerts.append(Alert(
                    id=row['id'],
                    level=AlertLevel(row['level']),
                    type=AlertType(row['type']),
                    title=row['title'],
                    message=row['message'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    source=row['source'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None,
                    resolved=bool(row['resolved']),
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None
                ))

            return alerts


class NotificationChannel:
    """Base class for notification channels"""

    def send(self, alert: Alert) -> bool:
        """Send alert notification"""
        raise NotImplementedError


class EmailChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, config: Dict):
        """Initialize email channel"""
        self.config = config
        self.enabled = config.get("enabled", False)
        self.recipients = config.get("recipients", [])
        self.smtp_server = os.getenv("SMTP_SERVER", config.get("smtp_server", "localhost"))
        self.smtp_port = int(os.getenv("SMTP_PORT", config.get("smtp_port", 25)))
        self.from_address = os.getenv("EMAIL_FROM", config.get("from_address", "noreply@pycaret.org"))

    def send(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.enabled or not self.recipients:
            return False

        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            msg['From'] = self.from_address
            msg['To'] = ", ".join(self.recipients)

            body = f"""
            Alert Level: {alert.level.value.upper()}
            Alert Type: {alert.type.value}
            Source: {alert.source}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            {alert.message}

            {'Metadata:' if alert.metadata else ''}
            {json.dumps(alert.metadata, indent=2) if alert.metadata else ''}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if os.getenv("SMTP_TLS", "false").lower() == "true":
                    server.starttls()
                if os.getenv("SMTP_USER"):
                    server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
                server.send_message(msg)

            logger.info(f"Email sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Error sending email for alert {alert.id}: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, config: Dict):
        """Initialize webhook channel"""
        self.config = config
        self.enabled = config.get("enabled", False)
        self.url = os.getenv("WEBHOOK_URL", config.get("url", ""))
        self.timeout = config.get("timeout", 10)

    def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        if not self.enabled or not self.url:
            return False

        try:
            payload = alert.to_dict()
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            logger.info(f"Webhook sent for alert {alert.id}: {response.status_code}")
            return True

        except Exception as e:
            logger.error(f"Error sending webhook for alert {alert.id}: {e}")
            return False


class SlackChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, config: Dict):
        """Initialize Slack channel"""
        self.config = config
        self.enabled = config.get("enabled", False)
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", config.get("webhook_url", ""))

    def send(self, alert: Alert) -> bool:
        """Send Slack notification"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            # Color based on alert level
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9900",
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#990000"
            }

            payload = {
                "attachments": [{
                    "color": color_map.get(alert.level, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Level", "value": alert.level.value.upper(), "short": True},
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "PyCaret Alert System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Slack notification sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification for alert {alert.id}: {e}")
            return False


class AlertSystem:
    """Main alert system coordinator"""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize alert system"""
        self.config_path = config_path or PROJECT_ROOT / "config" / "alert_config.json"
        self.config = self._load_config()

        # Initialize storage
        db_path = PROJECT_ROOT / ".swarm" / "alerts.db"
        self.storage = AlertStorage(db_path)

        # Initialize notification channels
        self.channels: List[NotificationChannel] = []
        self._init_channels()

        # Alert queue for async processing
        self.alert_queue = Queue()
        self.worker_thread = None
        self.running = False

    def _load_config(self) -> Dict:
        """Load configuration"""
        default_config = {
            "channels": {
                "email": {
                    "enabled": False,
                    "recipients": []
                },
                "webhook": {
                    "enabled": True,
                    "url": ""
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "rules": {
                "rate_limit": {
                    "enabled": True,
                    "max_alerts_per_hour": 10,
                    "per_type": True
                },
                "deduplication": {
                    "enabled": True,
                    "window_minutes": 5
                }
            },
            "monitors": {
                "performance": {
                    "enabled": True,
                    "cpu_threshold": 90,
                    "memory_threshold": 90,
                    "response_time_threshold": 5000
                },
                "availability": {
                    "enabled": True,
                    "check_interval": 60
                },
                "data_quality": {
                    "enabled": True,
                    "freshness_threshold_hours": 6
                }
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
                    logger.info(f"Loaded alert configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading alert config: {e}")

        return default_config

    def _init_channels(self):
        """Initialize notification channels"""
        channels_config = self.config.get("channels", {})

        if channels_config.get("email", {}).get("enabled"):
            self.channels.append(EmailChannel(channels_config["email"]))

        if channels_config.get("webhook", {}).get("enabled"):
            self.channels.append(WebhookChannel(channels_config["webhook"]))

        if channels_config.get("slack", {}).get("enabled"):
            self.channels.append(SlackChannel(channels_config["slack"]))

        logger.info(f"Initialized {len(self.channels)} notification channels")

    def create_alert(
        self,
        level: AlertLevel,
        type: AlertType,
        title: str,
        message: str,
        source: str = "system",
        metadata: Optional[Dict] = None
    ) -> Alert:
        """Create new alert"""
        alert = Alert(
            id=f"{type.value}_{int(datetime.now().timestamp() * 1000)}",
            level=level,
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )

        # Save to storage
        self.storage.save_alert(alert)

        # Queue for notification
        self.alert_queue.put(alert)

        logger.info(f"Created alert {alert.id}: {title}")

        return alert

    def _process_alerts(self):
        """Process alert queue"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)

                # Send to all channels
                for channel in self.channels:
                    try:
                        channel.send(alert)
                    except Exception as e:
                        logger.error(f"Error in notification channel: {e}")

                self.alert_queue.task_done()

            except Exception as e:
                if self.running:  # Ignore timeout when stopping
                    logger.error(f"Error processing alert: {e}")

    def start(self):
        """Start alert processing"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.worker_thread.start()
            logger.info("Alert system started")

    def stop(self):
        """Stop alert processing"""
        if self.running:
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            logger.info("Alert system stopped")

    def get_stats(self) -> Dict:
        """Get alert statistics"""
        recent_alerts = self.storage.get_recent_alerts(hours=24)

        stats = {
            "total_24h": len(recent_alerts),
            "unresolved": len([a for a in recent_alerts if not a.resolved]),
            "by_level": {},
            "by_type": {}
        }

        for alert in recent_alerts:
            # By level
            level_key = alert.level.value
            stats["by_level"][level_key] = stats["by_level"].get(level_key, 0) + 1

            # By type
            type_key = alert.type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

        return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="PyCaret Alert System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Send test alert")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    alert_system = AlertSystem(config_path=Path(args.config) if args.config else None)

    if args.test:
        alert_system.start()
        alert_system.create_alert(
            level=AlertLevel.INFO,
            type=AlertType.CUSTOM,
            title="Test Alert",
            message="This is a test alert from PyCaret Alert System",
            source="cli"
        )
        time.sleep(2)
        alert_system.stop()
        print("Test alert sent")
        return

    if args.stats:
        stats = alert_system.get_stats()
        print(json.dumps(stats, indent=2))
        return

    # Run as service
    alert_system.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        alert_system.stop()


if __name__ == "__main__":
    main()
