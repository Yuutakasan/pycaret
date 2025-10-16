#!/usr/bin/env python3
"""
PyCaret Data Refresh Scheduler
Automated data refresh with configurable intervals
"""

import os
import sys
import time
import logging
import schedule
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"data_refresh_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataRefreshScheduler:
    """Manages scheduled data refresh operations"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize scheduler with configuration"""
        self.config_path = config_path or PROJECT_ROOT / "config" / "refresh_config.json"
        self.config = self.load_config()
        self.refresh_history: List[Dict] = []
        self.max_history = 100

    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "schedules": {
                "hourly": {
                    "enabled": True,
                    "interval": "1h",
                    "tasks": ["update_metrics", "check_health"]
                },
                "daily": {
                    "enabled": True,
                    "interval": "1d",
                    "time": "02:00",
                    "tasks": ["full_refresh", "generate_reports"]
                },
                "weekly": {
                    "enabled": True,
                    "interval": "1w",
                    "day": "monday",
                    "time": "03:00",
                    "tasks": ["deep_analysis", "model_retrain"]
                }
            },
            "data_sources": [
                {
                    "name": "github_stats",
                    "type": "github_api",
                    "refresh_interval": "1h"
                },
                {
                    "name": "pypi_downloads",
                    "type": "pypi_api",
                    "refresh_interval": "6h"
                },
                {
                    "name": "model_metrics",
                    "type": "local_db",
                    "refresh_interval": "1h"
                }
            ],
            "notifications": {
                "enabled": True,
                "on_failure": True,
                "on_success": False,
                "email": {
                    "enabled": False,
                    "recipients": []
                },
                "webhook": {
                    "enabled": True,
                    "url": os.getenv("WEBHOOK_URL", "")
                }
            },
            "retry": {
                "max_attempts": 3,
                "backoff_factor": 2,
                "initial_delay": 60
            }
        }

        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}, using defaults")
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")

        return default_config

    def save_config(self):
        """Save current configuration to file"""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def refresh_data_source(self, source: Dict) -> bool:
        """Refresh a single data source"""
        source_name = source.get("name", "unknown")
        source_type = source.get("type", "unknown")

        logger.info(f"Refreshing data source: {source_name} (type: {source_type})")

        try:
            # Call appropriate refresh script based on source type
            script_map = {
                "github_api": "scripts/fetch_github_stats.py",
                "pypi_api": "scripts/fetch_pypi_stats.py",
                "local_db": "scripts/update_metrics.py"
            }

            script_path = PROJECT_ROOT / script_map.get(source_type, "")

            if script_path.exists():
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    logger.info(f"Successfully refreshed {source_name}")
                    return True
                else:
                    logger.error(f"Failed to refresh {source_name}: {result.stderr}")
                    return False
            else:
                logger.warning(f"Script not found for {source_name}: {script_path}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout refreshing {source_name}")
            return False
        except Exception as e:
            logger.error(f"Error refreshing {source_name}: {e}")
            return False

    def refresh_all_sources(self) -> Dict:
        """Refresh all configured data sources"""
        logger.info("Starting full data refresh")

        start_time = datetime.now()
        results = {
            "total": len(self.config["data_sources"]),
            "success": 0,
            "failed": 0,
            "sources": {}
        }

        for source in self.config["data_sources"]:
            source_name = source.get("name", "unknown")
            success = self.refresh_data_source(source)

            results["sources"][source_name] = {
                "success": success,
                "timestamp": datetime.now().isoformat()
            }

            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

        duration = (datetime.now() - start_time).total_seconds()
        results["duration_seconds"] = duration
        results["timestamp"] = start_time.isoformat()

        # Record in history
        self.refresh_history.append(results)
        if len(self.refresh_history) > self.max_history:
            self.refresh_history.pop(0)

        logger.info(f"Refresh completed: {results['success']}/{results['total']} successful in {duration:.2f}s")

        # Send notifications
        if results["failed"] > 0 and self.config["notifications"]["on_failure"]:
            self.send_notification("failure", results)
        elif results["failed"] == 0 and self.config["notifications"]["on_success"]:
            self.send_notification("success", results)

        return results

    def send_notification(self, status: str, results: Dict):
        """Send notification about refresh status"""
        message = f"Data refresh {status}: {results['success']}/{results['total']} sources updated"

        # Webhook notification
        if self.config["notifications"]["webhook"]["enabled"]:
            webhook_url = self.config["notifications"]["webhook"]["url"]
            if webhook_url:
                try:
                    import requests
                    payload = {
                        "status": status,
                        "message": message,
                        "details": results,
                        "timestamp": datetime.now().isoformat()
                    }
                    response = requests.post(webhook_url, json=payload, timeout=10)
                    logger.info(f"Webhook notification sent: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error sending webhook notification: {e}")

        # Email notification
        if self.config["notifications"]["email"]["enabled"]:
            self.send_email(status, message, results)

    def send_email(self, status: str, message: str, results: Dict):
        """Send email notification"""
        try:
            recipients = self.config["notifications"]["email"]["recipients"]
            if not recipients:
                return

            msg = MIMEMultipart()
            msg['Subject'] = f"PyCaret Data Refresh - {status.upper()}"
            msg['From'] = os.getenv("EMAIL_FROM", "noreply@pycaret.org")
            msg['To'] = ", ".join(recipients)

            body = f"""
            {message}

            Details:
            - Total sources: {results['total']}
            - Successful: {results['success']}
            - Failed: {results['failed']}
            - Duration: {results['duration_seconds']:.2f} seconds
            - Timestamp: {results['timestamp']}

            Source Details:
            {json.dumps(results['sources'], indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            smtp_server = os.getenv("SMTP_SERVER", "localhost")
            smtp_port = int(os.getenv("SMTP_PORT", "25"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if os.getenv("SMTP_TLS", "false").lower() == "true":
                    server.starttls()
                if os.getenv("SMTP_USER"):
                    server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
                server.send_message(msg)

            logger.info(f"Email notification sent to {len(recipients)} recipients")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")

    def scheduled_hourly_refresh(self):
        """Hourly refresh task"""
        logger.info("Running hourly refresh")
        # Refresh only hourly sources
        for source in self.config["data_sources"]:
            if source.get("refresh_interval") == "1h":
                self.refresh_data_source(source)

    def scheduled_daily_refresh(self):
        """Daily refresh task"""
        logger.info("Running daily refresh")
        self.refresh_all_sources()

    def scheduled_weekly_refresh(self):
        """Weekly refresh task"""
        logger.info("Running weekly refresh")
        self.refresh_all_sources()
        # Run deep analysis
        self.run_deep_analysis()

    def run_deep_analysis(self):
        """Run comprehensive analysis"""
        logger.info("Running deep analysis")
        try:
            script_path = PROJECT_ROOT / "scripts" / "analyze_trends.py"
            if script_path.exists():
                subprocess.run([sys.executable, str(script_path)], check=True, timeout=600)
        except Exception as e:
            logger.error(f"Error running deep analysis: {e}")

    def setup_schedules(self):
        """Setup all scheduled tasks"""
        schedules_config = self.config.get("schedules", {})

        # Hourly schedule
        if schedules_config.get("hourly", {}).get("enabled", False):
            schedule.every().hour.do(self.scheduled_hourly_refresh)
            logger.info("Hourly refresh scheduled")

        # Daily schedule
        daily_config = schedules_config.get("daily", {})
        if daily_config.get("enabled", False):
            daily_time = daily_config.get("time", "02:00")
            schedule.every().day.at(daily_time).do(self.scheduled_daily_refresh)
            logger.info(f"Daily refresh scheduled at {daily_time}")

        # Weekly schedule
        weekly_config = schedules_config.get("weekly", {})
        if weekly_config.get("enabled", False):
            weekly_day = weekly_config.get("day", "monday")
            weekly_time = weekly_config.get("time", "03:00")
            getattr(schedule.every(), weekly_day).at(weekly_time).do(self.scheduled_weekly_refresh)
            logger.info(f"Weekly refresh scheduled on {weekly_day} at {weekly_time}")

    def get_next_run_time(self) -> Optional[datetime]:
        """Get next scheduled run time"""
        jobs = schedule.get_jobs()
        if not jobs:
            return None

        next_run = min(job.next_run for job in jobs)
        return next_run

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "running": True,
            "next_run": self.get_next_run_time().isoformat() if self.get_next_run_time() else None,
            "scheduled_jobs": len(schedule.get_jobs()),
            "last_refresh": self.refresh_history[-1] if self.refresh_history else None,
            "total_refreshes": len(self.refresh_history)
        }

    def run(self):
        """Run the scheduler"""
        logger.info("Starting data refresh scheduler")
        logger.info(f"Next run: {self.get_next_run_time()}")

        # Run initial refresh if configured
        if self.config.get("run_on_startup", True):
            logger.info("Running initial refresh")
            self.refresh_all_sources()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PyCaret Data Refresh Scheduler")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration")

    args = parser.parse_args()

    scheduler = DataRefreshScheduler(config_path=args.config)

    if args.create_config:
        scheduler.save_config()
        print(f"Configuration created at {scheduler.config_path}")
        return

    if args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.once:
        results = scheduler.refresh_all_sources()
        print(json.dumps(results, indent=2))
        return

    # Setup and run scheduler
    scheduler.setup_schedules()
    scheduler.run()


if __name__ == "__main__":
    main()
