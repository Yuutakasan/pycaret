#!/usr/bin/env python3
"""
PyCaret Performance Monitoring System
Real-time performance metrics collection and analysis
"""

import os
import sys
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
import threading

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"performance_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    response_time_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsStorage:
    """SQLite storage for performance metrics"""

    def __init__(self, db_path: Path):
        """Initialize metrics storage"""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp TEXT PRIMARY KEY,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_mb REAL,
                    disk_usage_percent REAL,
                    disk_io_read_mb REAL,
                    disk_io_write_mb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    process_count INTEGER,
                    response_time_ms REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON metrics(timestamp)
            """)

    def save_metric(self, metric: PerformanceMetric):
        """Save performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metrics
                (timestamp, cpu_percent, memory_percent, memory_mb, disk_usage_percent,
                 disk_io_read_mb, disk_io_write_mb, network_sent_mb, network_recv_mb,
                 process_count, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp.isoformat(),
                metric.cpu_percent,
                metric.memory_percent,
                metric.memory_mb,
                metric.disk_usage_percent,
                metric.disk_io_read_mb,
                metric.disk_io_write_mb,
                metric.network_sent_mb,
                metric.network_recv_mb,
                metric.process_count,
                metric.response_time_ms
            ))

    def get_metrics(self, hours: int = 24) -> List[PerformanceMetric]:
        """Get metrics from last N hours"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM metrics
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (since.isoformat(),))

            metrics = []
            for row in cursor:
                metrics.append(PerformanceMetric(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    cpu_percent=row['cpu_percent'],
                    memory_percent=row['memory_percent'],
                    memory_mb=row['memory_mb'],
                    disk_usage_percent=row['disk_usage_percent'],
                    disk_io_read_mb=row['disk_io_read_mb'],
                    disk_io_write_mb=row['disk_io_write_mb'],
                    network_sent_mb=row['network_sent_mb'],
                    network_recv_mb=row['network_recv_mb'],
                    process_count=row['process_count'],
                    response_time_ms=row['response_time_ms']
                ))

            return metrics

    def cleanup_old_metrics(self, days: int = 30):
        """Remove metrics older than N days"""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM metrics WHERE timestamp < ?
            """, (cutoff.isoformat(),))

            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} old metrics")


class PerformanceMonitor:
    """Performance monitoring system"""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize performance monitor"""
        self.config_path = config_path or PROJECT_ROOT / "config" / "monitor_config.json"
        self.config = self._load_config()

        # Initialize storage
        db_path = PROJECT_ROOT / ".swarm" / "metrics.db"
        self.storage = MetricsStorage(db_path)

        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.last_disk_io = None
        self.last_network_io = None

    def _load_config(self) -> Dict:
        """Load configuration"""
        default_config = {
            "collection_interval": 60,  # seconds
            "retention_days": 30,
            "thresholds": {
                "cpu_warning": 70,
                "cpu_critical": 90,
                "memory_warning": 75,
                "memory_critical": 90,
                "disk_warning": 80,
                "disk_critical": 95,
                "response_time_warning": 3000,  # ms
                "response_time_critical": 5000  # ms
            },
            "alerts_enabled": True,
            "cleanup_enabled": True,
            "cleanup_interval": 86400  # daily
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
                    logger.info(f"Loaded monitor configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading monitor config: {e}")

        return default_config

    def collect_metrics(self) -> PerformanceMetric:
        """Collect current performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = 0
        disk_io_write_mb = 0

        if self.last_disk_io:
            disk_io_read_mb = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024)
            disk_io_write_mb = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024 * 1024)

        self.last_disk_io = disk_io

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0
        network_recv_mb = 0

        if self.last_network_io:
            network_sent_mb = (network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024 * 1024)
            network_recv_mb = (network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024 * 1024)

        self.last_network_io = network_io

        # Process count
        process_count = len(psutil.pids())

        # Create metric
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_usage_percent=disk_usage_percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count
        )

        return metric

    def check_thresholds(self, metric: PerformanceMetric):
        """Check if metrics exceed thresholds and trigger alerts"""
        if not self.config.get("alerts_enabled", True):
            return

        thresholds = self.config.get("thresholds", {})

        # Import alert system if available
        try:
            from alert_system import AlertSystem, AlertLevel, AlertType

            alert_system = AlertSystem()

            # CPU checks
            if metric.cpu_percent >= thresholds.get("cpu_critical", 90):
                alert_system.create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.PERFORMANCE,
                    title="Critical CPU Usage",
                    message=f"CPU usage at {metric.cpu_percent:.1f}%",
                    source="performance_monitor",
                    metadata={"cpu_percent": metric.cpu_percent}
                )
            elif metric.cpu_percent >= thresholds.get("cpu_warning", 70):
                alert_system.create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.PERFORMANCE,
                    title="High CPU Usage",
                    message=f"CPU usage at {metric.cpu_percent:.1f}%",
                    source="performance_monitor",
                    metadata={"cpu_percent": metric.cpu_percent}
                )

            # Memory checks
            if metric.memory_percent >= thresholds.get("memory_critical", 90):
                alert_system.create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.RESOURCE,
                    title="Critical Memory Usage",
                    message=f"Memory usage at {metric.memory_percent:.1f}% ({metric.memory_mb:.0f} MB)",
                    source="performance_monitor",
                    metadata={"memory_percent": metric.memory_percent, "memory_mb": metric.memory_mb}
                )
            elif metric.memory_percent >= thresholds.get("memory_warning", 75):
                alert_system.create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.RESOURCE,
                    title="High Memory Usage",
                    message=f"Memory usage at {metric.memory_percent:.1f}% ({metric.memory_mb:.0f} MB)",
                    source="performance_monitor",
                    metadata={"memory_percent": metric.memory_percent, "memory_mb": metric.memory_mb}
                )

            # Disk checks
            if metric.disk_usage_percent >= thresholds.get("disk_critical", 95):
                alert_system.create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.RESOURCE,
                    title="Critical Disk Usage",
                    message=f"Disk usage at {metric.disk_usage_percent:.1f}%",
                    source="performance_monitor",
                    metadata={"disk_usage_percent": metric.disk_usage_percent}
                )
            elif metric.disk_usage_percent >= thresholds.get("disk_warning", 80):
                alert_system.create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.RESOURCE,
                    title="High Disk Usage",
                    message=f"Disk usage at {metric.disk_usage_percent:.1f}%",
                    source="performance_monitor",
                    metadata={"disk_usage_percent": metric.disk_usage_percent}
                )

        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        interval = self.config.get("collection_interval", 60)
        cleanup_interval = self.config.get("cleanup_interval", 86400)
        last_cleanup = time.time()

        while self.running:
            try:
                # Collect and save metrics
                metric = self.collect_metrics()
                self.storage.save_metric(metric)

                # Check thresholds
                self.check_thresholds(metric)

                # Periodic cleanup
                if self.config.get("cleanup_enabled", True):
                    if time.time() - last_cleanup > cleanup_interval:
                        retention_days = self.config.get("retention_days", 30)
                        self.storage.cleanup_old_metrics(retention_days)
                        last_cleanup = time.time()

                # Wait for next collection
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def start(self):
        """Start monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")

    def stop(self):
        """Stop monitoring"""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Performance monitoring stopped")

    def get_current_status(self) -> Dict:
        """Get current system status"""
        metric = self.collect_metrics()
        return metric.to_dict()

    def get_statistics(self, hours: int = 24) -> Dict:
        """Get performance statistics"""
        metrics = self.storage.get_metrics(hours=hours)

        if not metrics:
            return {}

        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]

        stats = {
            "period_hours": hours,
            "total_samples": len(metrics),
            "cpu": {
                "current": metrics[0].cpu_percent,
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current": metrics[0].memory_percent,
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "disk": {
                "current": metrics[0].disk_usage_percent
            },
            "latest_timestamp": metrics[0].timestamp.isoformat()
        }

        return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="PyCaret Performance Monitor")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--hours", type=int, default=24, help="Hours for statistics")

    args = parser.parse_args()

    monitor = PerformanceMonitor(config_path=Path(args.config) if args.config else None)

    if args.status:
        status = monitor.get_current_status()
        print(json.dumps(status, indent=2))
        return

    if args.stats:
        stats = monitor.get_statistics(hours=args.hours)
        print(json.dumps(stats, indent=2))
        return

    # Run as service
    monitor.start()
    try:
        logger.info("Performance monitor running. Press Ctrl+C to stop.")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
