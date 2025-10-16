# PyCaret Deployment Scripts

This directory contains all deployment and automation scripts for the PyCaret Dashboard infrastructure.

## Scripts Overview

### 1. Dashboard Deployment (`deploy_dashboard.sh`)

Automated dashboard generation and deployment script.

**Location**: `/mnt/d/github/pycaret/scripts/deploy_dashboard.sh`

**Usage**:
```bash
# Basic deployment
./scripts/deploy_dashboard.sh

# Production deployment with Docker
ENVIRONMENT=production DEPLOY_METHOD=docker ./scripts/deploy_dashboard.sh

# Skip tests
SKIP_TESTS=true ./scripts/deploy_dashboard.sh
```

**Features**:
- Pre-deployment checks
- Automated testing
- Dashboard generation
- Local or Docker deployment
- Health checks
- Rollback on failure
- Notification support

**Environment Variables**:
- `ENVIRONMENT`: production, staging, development (default: production)
- `DEPLOY_METHOD`: local, docker (default: local)
- `DASHBOARD_PORT`: Dashboard port (default: 8050)
- `WORKERS`: Number of worker processes (default: 4)
- `SKIP_TESTS`: Skip pre-deployment tests (default: false)
- `WEBHOOK_URL`: Webhook for deployment notifications

### 2. Data Refresh Scheduler (`data_refresh.py`)

Automated data refresh with configurable intervals.

**Location**: `/mnt/d/github/pycaret/scripts/data_refresh.py`

**Usage**:
```bash
# Run scheduler
python scripts/data_refresh.py

# Run once and exit
python scripts/data_refresh.py --once

# Show status
python scripts/data_refresh.py --status

# Create default configuration
python scripts/data_refresh.py --create-config
```

**Features**:
- Configurable refresh schedules (hourly, daily, weekly)
- Multiple data source support
- Retry mechanism with backoff
- Email and webhook notifications
- Comprehensive logging
- Status monitoring

**Configuration**: `/mnt/d/github/pycaret/config/refresh_config.json`

### 3. Alert Notification System (`alert_system.py`)

Real-time monitoring and alerting for critical events.

**Location**: `/mnt/d/github/pycaret/scripts/alert_system.py`

**Usage**:
```bash
# Run alert system
python scripts/alert_system.py

# Send test alert
python scripts/alert_system.py --test

# Show statistics
python scripts/alert_system.py --stats
```

**Features**:
- Multi-channel notifications (Email, Webhook, Slack)
- Alert severity levels (INFO, WARNING, ERROR, CRITICAL)
- Alert types (Performance, Availability, Data Quality, Security)
- SQLite storage for alert history
- Rate limiting and deduplication
- Async alert processing

**Configuration**: `/mnt/d/github/pycaret/config/alert_config.json`

### 4. Performance Monitor (`performance_monitor.py`)

Real-time performance metrics collection and analysis.

**Location**: `/mnt/d/github/pycaret/scripts/performance_monitor.py`

**Usage**:
```bash
# Run monitor
python scripts/performance_monitor.py

# Show current status
python scripts/performance_monitor.py --status

# Show statistics
python scripts/performance_monitor.py --stats --hours 24
```

**Features**:
- CPU, memory, disk, and network monitoring
- Configurable thresholds
- Automatic alerting
- SQLite metrics storage
- Automatic cleanup of old metrics
- Real-time performance tracking

**Configuration**: `/mnt/d/github/pycaret/config/monitor_config.json`

### 5. Backup & Restore (`backup_restore.sh`)

Automated backup and disaster recovery system.

**Location**: `/mnt/d/github/pycaret/scripts/backup_restore.sh`

**Usage**:
```bash
# Create backup
./scripts/backup_restore.sh backup

# List backups
./scripts/backup_restore.sh list

# Restore from backup
./scripts/backup_restore.sh restore backups/pycaret_backup_20251008_120000.tar.gz

# Verify backup
./scripts/backup_restore.sh verify backups/pycaret_backup_20251008_120000.tar.gz

# Cleanup old backups
./scripts/backup_restore.sh cleanup
```

**Features**:
- Full system backups (databases, config, data, code)
- Compression with tar/gzip
- Optional encryption with OpenSSL
- S3 upload support
- Automated cleanup with retention policies
- Backup verification
- Complete restore functionality

**Environment Variables**:
- `BACKUP_DIR`: Backup directory (default: ./backups)
- `BACKUP_ENCRYPTION_KEY`: Encryption key
- `AWS_S3_BUCKET`: S3 bucket for remote backups
- `BACKUP_UPLOAD_URL`: Custom upload endpoint

## Installation

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# Additional dependencies for monitoring
pip install psutil schedule requests

# Make scripts executable
chmod +x scripts/*.sh
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y curl jq sqlite3 openssl

# For performance monitoring
sudo apt-get install -y sysstat
```

## Configuration

Create configuration directory and files:

```bash
mkdir -p config

# Create configuration files
python scripts/data_refresh.py --create-config
cp config/refresh_config.json config/alert_config.json
cp config/refresh_config.json config/monitor_config.json
```

Edit configurations as needed:
- `/mnt/d/github/pycaret/config/refresh_config.json`
- `/mnt/d/github/pycaret/config/alert_config.json`
- `/mnt/d/github/pycaret/config/monitor_config.json`

## Automated Scheduling

### Using Systemd (Recommended for Production)

Create systemd service files:

```bash
# Data refresh service
sudo tee /etc/systemd/system/pycaret-refresh.service <<EOF
[Unit]
Description=PyCaret Data Refresh Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/pycaret
ExecStart=/usr/bin/python3 /path/to/pycaret/scripts/data_refresh.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Performance monitor service
sudo tee /etc/systemd/system/pycaret-monitor.service <<EOF
[Unit]
Description=PyCaret Performance Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/pycaret
ExecStart=/usr/bin/python3 /path/to/pycaret/scripts/performance_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Alert system service
sudo tee /etc/systemd/system/pycaret-alerts.service <<EOF
[Unit]
Description=PyCaret Alert System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/pycaret
ExecStart=/usr/bin/python3 /path/to/pycaret/scripts/alert_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable pycaret-refresh pycaret-monitor pycaret-alerts
sudo systemctl start pycaret-refresh pycaret-monitor pycaret-alerts

# Check status
sudo systemctl status pycaret-refresh
```

### Using Cron

```bash
# Edit crontab
crontab -e

# Add entries
# Data refresh (runs continuously, start on reboot)
@reboot cd /path/to/pycaret && python scripts/data_refresh.py

# Performance monitor (runs continuously, start on reboot)
@reboot cd /path/to/pycaret && python scripts/performance_monitor.py

# Daily backup at 2 AM
0 2 * * * cd /path/to/pycaret && ./scripts/backup_restore.sh backup

# Weekly cleanup on Sunday at 3 AM
0 3 * * 0 cd /path/to/pycaret && ./scripts/backup_restore.sh cleanup
```

## Monitoring

### Check Service Status

```bash
# Systemd services
sudo systemctl status pycaret-refresh
sudo systemctl status pycaret-monitor
sudo systemctl status pycaret-alerts

# View logs
sudo journalctl -u pycaret-refresh -f
sudo journalctl -u pycaret-monitor -f
sudo journalctl -u pycaret-alerts -f

# Check file logs
tail -f logs/data_refresh_*.log
tail -f logs/performance_*.log
tail -f logs/alerts_*.log
```

### Performance Monitoring

```bash
# Current system status
python scripts/performance_monitor.py --status

# Statistics for last 24 hours
python scripts/performance_monitor.py --stats --hours 24

# Real-time monitoring
watch -n 5 'python scripts/performance_monitor.py --status'
```

### Alert Monitoring

```bash
# Alert statistics
python scripts/alert_system.py --stats

# Recent alerts
sqlite3 .swarm/alerts.db "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10;"
```

## Troubleshooting

### Common Issues

1. **Script Permission Denied**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Python Module Not Found**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Locked**
   ```bash
   # Close other connections or restart services
   sudo systemctl restart pycaret-refresh
   ```

4. **High Resource Usage**
   ```bash
   # Reduce collection interval
   # Edit config/monitor_config.json
   # Increase "collection_interval" to 300 (5 minutes)
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run script
python scripts/data_refresh.py
```

## Integration with Docker

All scripts are integrated into the Docker deployment:

```bash
# See docker-compose.yml for service definitions
cd docker
docker-compose up -d

# Check logs
docker-compose logs -f data-refresh
docker-compose logs -f performance-monitor
docker-compose logs -f alert-system
```

## Security Considerations

1. **Protect Configuration Files**
   ```bash
   chmod 600 config/*.json
   ```

2. **Secure Credentials**
   - Never commit credentials to Git
   - Use environment variables
   - Use secrets management (AWS Secrets Manager, Vault)

3. **Encrypt Backups**
   ```bash
   export BACKUP_ENCRYPTION_KEY="your-secure-key"
   ./scripts/backup_restore.sh backup
   ```

4. **Regular Updates**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Support

For issues or questions:
- GitHub Issues: https://github.com/pycaret/pycaret/issues
- Documentation: See `/mnt/d/github/pycaret/docs/deployment_guide.md`
