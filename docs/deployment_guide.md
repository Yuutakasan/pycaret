# PyCaret Dashboard Deployment Guide

Complete guide for deploying and managing the PyCaret Dashboard infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Deployment Methods](#deployment-methods)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Backup & Recovery](#backup--recovery)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Troubleshooting](#troubleshooting)
10. [Security](#security)

---

## Overview

The PyCaret Dashboard deployment system includes:

- **Automated Dashboard Generation**: `/mnt/d/github/pycaret/scripts/deploy_dashboard.sh`
- **Data Refresh Scheduler**: `/mnt/d/github/pycaret/scripts/data_refresh.py`
- **Alert Notification System**: `/mnt/d/github/pycaret/scripts/alert_system.py`
- **Performance Monitoring**: `/mnt/d/github/pycaret/scripts/performance_monitor.py`
- **Backup & Recovery**: `/mnt/d/github/pycaret/scripts/backup_restore.sh`
- **Docker Containerization**: `/mnt/d/github/pycaret/docker/`
- **CI/CD Pipeline**: `/mnt/d/github/pycaret/.github/workflows/deploy.yml`

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌───▼──────┐
│  Dashboard   │ │  Dashboard│ │ Dashboard│
│  Instance 1  │ │ Instance 2│ │Instance 3│
└──────┬───────┘ └────┬──────┘ └────┬─────┘
       │              │             │
       └──────────────┼─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌───▼──────┐
│Data Refresh  │ │Performance│ │  Alert   │
│  Service     │ │  Monitor  │ │  System  │
└──────────────┘ └───────────┘ └──────────┘
```

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- **CPU**: 2+ cores (4+ recommended for production)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 20GB minimum (SSD recommended)
- **Network**: Stable internet connection

### Software Dependencies

```bash
# Python 3.8+
python3 --version

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version

# Node.js (optional, for asset building)
node --version
npm --version
```

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system packages
sudo apt-get update
sudo apt-get install -y build-essential curl git

# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

---

## Quick Start

### 1. Local Development Deployment

```bash
# Clone repository
git clone https://github.com/pycaret/pycaret.git
cd pycaret

# Deploy locally
chmod +x scripts/deploy_dashboard.sh
./scripts/deploy_dashboard.sh

# Dashboard will be available at http://localhost:8050
```

### 2. Docker Deployment

```bash
# Using Docker Compose
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f dashboard
```

### 3. Production Deployment

```bash
# Set environment variables
export ENVIRONMENT=production
export DEPLOY_METHOD=docker
export DASHBOARD_PORT=8050
export WORKERS=4

# Run deployment script
./scripts/deploy_dashboard.sh

# Verify deployment
curl http://localhost:8050/health
```

---

## Deployment Methods

### Method 1: Local Deployment

**Pros**: Simple, fast setup
**Cons**: Single point of failure, manual scaling

```bash
# Deploy
./scripts/deploy_dashboard.sh

# Stop
pkill -f "dashboard.py"

# Restart
./scripts/deploy_dashboard.sh
```

### Method 2: Docker Deployment

**Pros**: Isolated environment, easy scaling
**Cons**: Requires Docker knowledge

```bash
# Start all services
cd docker
docker-compose up -d

# Scale dashboard instances
docker-compose up -d --scale dashboard=3

# Stop all services
docker-compose down

# View service logs
docker-compose logs -f [service-name]
```

### Method 3: Kubernetes Deployment

**Pros**: High availability, auto-scaling
**Cons**: Complex setup

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n pycaret
kubectl get svc -n pycaret
```

### Method 4: Cloud Deployment

#### AWS Deployment

```bash
# Using ECS
aws ecs create-cluster --cluster-name pycaret-cluster
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json
aws ecs create-service --cluster pycaret-cluster --service-name dashboard --task-definition pycaret-dashboard

# Using Elastic Beanstalk
eb init -p docker pycaret-dashboard
eb create pycaret-env
eb deploy
```

#### Google Cloud Platform

```bash
# Using Cloud Run
gcloud run deploy pycaret-dashboard \
  --image gcr.io/PROJECT_ID/pycaret-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Deployment

```bash
# Using Container Instances
az container create \
  --resource-group pycaret-rg \
  --name pycaret-dashboard \
  --image pycaret/dashboard:latest \
  --dns-name-label pycaret-dashboard \
  --ports 8050
```

---

## Configuration

### Environment Variables

Create `/mnt/d/github/pycaret/.env` file:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Dashboard
DASHBOARD_PORT=8050
WORKERS=4
TIMEOUT=300

# Deployment
DEPLOY_METHOD=docker
SKIP_TESTS=false

# Database
DATABASE_URL=sqlite:////.swarm/metrics.db

# Monitoring
WEBHOOK_URL=https://your-webhook-url.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_TLS=true
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@pycaret.org

# Backup
AWS_S3_BUCKET=pycaret-backups
BACKUP_ENCRYPTION_KEY=your-encryption-key
BACKUP_UPLOAD_URL=https://your-backup-server.com

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

### Data Refresh Configuration

Create `/mnt/d/github/pycaret/config/refresh_config.json`:

```json
{
  "schedules": {
    "hourly": {
      "enabled": true,
      "interval": "1h",
      "tasks": ["update_metrics", "check_health"]
    },
    "daily": {
      "enabled": true,
      "interval": "1d",
      "time": "02:00",
      "tasks": ["full_refresh", "generate_reports"]
    },
    "weekly": {
      "enabled": true,
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
    }
  ],
  "notifications": {
    "enabled": true,
    "on_failure": true,
    "on_success": false
  }
}
```

### Alert Configuration

Create `/mnt/d/github/pycaret/config/alert_config.json`:

```json
{
  "channels": {
    "email": {
      "enabled": false,
      "recipients": ["admin@pycaret.org"]
    },
    "webhook": {
      "enabled": true,
      "url": "https://your-webhook.com"
    },
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    }
  },
  "monitors": {
    "performance": {
      "enabled": true,
      "cpu_threshold": 90,
      "memory_threshold": 90
    },
    "availability": {
      "enabled": true,
      "check_interval": 60
    }
  }
}
```

### Performance Monitor Configuration

Create `/mnt/d/github/pycaret/config/monitor_config.json`:

```json
{
  "collection_interval": 60,
  "retention_days": 30,
  "thresholds": {
    "cpu_warning": 70,
    "cpu_critical": 90,
    "memory_warning": 75,
    "memory_critical": 90,
    "disk_warning": 80,
    "disk_critical": 95
  },
  "alerts_enabled": true,
  "cleanup_enabled": true
}
```

---

## Monitoring

### Dashboard Health Check

```bash
# Check if dashboard is running
curl http://localhost:8050/health

# Expected response
{"status": "healthy", "timestamp": "2025-10-08T10:00:00Z"}
```

### Performance Metrics

```bash
# Get current performance status
python scripts/performance_monitor.py --status

# Get statistics for last 24 hours
python scripts/performance_monitor.py --stats --hours 24
```

### Alert Statistics

```bash
# Get alert statistics
python scripts/alert_system.py --stats

# Send test alert
python scripts/alert_system.py --test
```

### Data Refresh Status

```bash
# Check scheduler status
python scripts/data_refresh.py --status

# Run manual refresh
python scripts/data_refresh.py --once
```

### Docker Container Monitoring

```bash
# View container status
docker-compose ps

# View resource usage
docker stats

# View logs
docker-compose logs -f dashboard
docker-compose logs -f performance-monitor

# Inspect container
docker inspect pycaret-dashboard
```

### Log Monitoring

```bash
# View deployment logs
tail -f logs/deploy_*.log

# View dashboard logs
tail -f logs/dashboard_*.log

# View performance logs
tail -f logs/performance_*.log

# View alert logs
tail -f logs/alerts_*.log
```

---

## Backup & Recovery

### Creating Backups

```bash
# Manual backup
./scripts/backup_restore.sh backup

# Automated backup (add to crontab)
0 2 * * * /path/to/scripts/backup_restore.sh backup

# With encryption
BACKUP_ENCRYPTION_KEY=your-key ./scripts/backup_restore.sh backup

# With S3 upload
AWS_S3_BUCKET=your-bucket ./scripts/backup_restore.sh backup
```

### Listing Backups

```bash
# List available backups
./scripts/backup_restore.sh list

# Output example:
#   pycaret_backup_20251008_020000.tar.gz - Size: 125M - Date: 2025-10-08
#   pycaret_backup_20251007_020000.tar.gz - Size: 123M - Date: 2025-10-07
```

### Restoring from Backup

```bash
# Restore from specific backup
./scripts/backup_restore.sh restore backups/pycaret_backup_20251008_020000.tar.gz

# With decryption
BACKUP_ENCRYPTION_KEY=your-key ./scripts/backup_restore.sh restore backups/pycaret_backup_20251008_020000.tar.gz.enc

# Verify backup before restore
./scripts/backup_restore.sh verify backups/pycaret_backup_20251008_020000.tar.gz
```

### Backup Schedule Recommendations

```bash
# Add to crontab: crontab -e

# Daily backups at 2 AM
0 2 * * * BACKUP_DIR=/backups /path/to/scripts/backup_restore.sh backup

# Weekly cleanup at 3 AM on Sundays
0 3 * * 0 /path/to/scripts/backup_restore.sh cleanup

# Monthly full backup with S3 upload
0 1 1 * * AWS_S3_BUCKET=pycaret-backups /path/to/scripts/backup_restore.sh backup
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline is configured in `/mnt/d/github/pycaret/.github/workflows/deploy.yml`

#### Workflow Stages

1. **Test**: Run tests on multiple Python versions
2. **Security**: Security scanning with Bandit and Safety
3. **Build**: Build and push Docker images
4. **Deploy Staging**: Auto-deploy to staging on `develop` branch
5. **Deploy Production**: Auto-deploy to production on `main`/`master` branch

#### Required Secrets

Configure in GitHub Settings → Secrets and variables → Actions:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
BACKUP_S3_BUCKET
BACKUP_ENCRYPTION_KEY
WEBHOOK_URL
SLACK_WEBHOOK
SMTP_USER
SMTP_PASSWORD
```

#### Manual Deployment

```bash
# Trigger deployment via GitHub CLI
gh workflow run deploy.yml --ref main -f environment=production

# Trigger rollback
gh workflow run deploy.yml --ref main -f environment=production
```

#### Deployment Verification

```bash
# Check workflow status
gh run list --workflow=deploy.yml

# View workflow logs
gh run view [RUN_ID] --log

# Check deployment
curl https://dashboard.pycaret.org/health
```

---

## Troubleshooting

### Common Issues

#### 1. Dashboard Not Starting

```bash
# Check logs
tail -f logs/dashboard_*.log

# Check if port is in use
sudo lsof -i :8050

# Kill existing process
pkill -f "dashboard.py"

# Restart
./scripts/deploy_dashboard.sh
```

#### 2. High Memory Usage

```bash
# Check memory
python scripts/performance_monitor.py --status

# Restart with fewer workers
WORKERS=2 ./scripts/deploy_dashboard.sh

# Or scale down Docker
docker-compose up -d --scale dashboard=1
```

#### 3. Data Refresh Failing

```bash
# Check scheduler status
python scripts/data_refresh.py --status

# Run manual refresh
python scripts/data_refresh.py --once

# Check logs
tail -f logs/data_refresh_*.log
```

#### 4. Docker Container Issues

```bash
# Restart container
docker-compose restart dashboard

# Rebuild and restart
docker-compose up -d --build dashboard

# Check container logs
docker-compose logs --tail=100 dashboard

# Remove and recreate
docker-compose down
docker-compose up -d
```

#### 5. Alerts Not Sending

```bash
# Test alert system
python scripts/alert_system.py --test

# Check configuration
cat config/alert_config.json

# Verify webhook
curl -X POST $WEBHOOK_URL -d '{"test": "message"}'
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
./scripts/deploy_dashboard.sh

# Or in Docker
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d
```

### Performance Optimization

```bash
# Increase workers
export WORKERS=8
./scripts/deploy_dashboard.sh

# Enable caching
export ENABLE_CACHE=true

# Optimize database
sqlite3 .swarm/metrics.db "VACUUM;"
sqlite3 .swarm/metrics.db "ANALYZE;"
```

---

## Security

### Best Practices

1. **Use HTTPS**: Always use SSL/TLS in production
2. **Secure Secrets**: Never commit secrets to Git
3. **Regular Updates**: Keep dependencies updated
4. **Access Control**: Implement authentication
5. **Network Security**: Use firewalls and VPCs
6. **Backup Encryption**: Always encrypt backups
7. **Audit Logs**: Enable comprehensive logging

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/dashboard.key \
  -out docker/ssl/dashboard.crt

# Use Let's Encrypt (production)
certbot certonly --standalone -d dashboard.pycaret.org
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8050/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8050 -j ACCEPT
```

### Security Scanning

```bash
# Scan dependencies
safety check

# Scan code
bandit -r .

# Scan Docker image
docker scan pycaret-dashboard:latest
```

---

## Additional Resources

### Scripts Reference

- **Deploy Dashboard**: `/mnt/d/github/pycaret/scripts/deploy_dashboard.sh`
- **Data Refresh**: `/mnt/d/github/pycaret/scripts/data_refresh.py`
- **Alert System**: `/mnt/d/github/pycaret/scripts/alert_system.py`
- **Performance Monitor**: `/mnt/d/github/pycaret/scripts/performance_monitor.py`
- **Backup & Restore**: `/mnt/d/github/pycaret/scripts/backup_restore.sh`

### Configuration Files

- **Refresh Config**: `/mnt/d/github/pycaret/config/refresh_config.json`
- **Alert Config**: `/mnt/d/github/pycaret/config/alert_config.json`
- **Monitor Config**: `/mnt/d/github/pycaret/config/monitor_config.json`

### Docker Files

- **Dockerfile**: `/mnt/d/github/pycaret/docker/Dockerfile.dashboard`
- **Docker Compose**: `/mnt/d/github/pycaret/docker/docker-compose.yml`
- **Dockerignore**: `/mnt/d/github/pycaret/docker/.dockerignore`

### CI/CD

- **GitHub Actions**: `/mnt/d/github/pycaret/.github/workflows/deploy.yml`

### Support

- **Documentation**: https://pycaret.org/docs
- **GitHub Issues**: https://github.com/pycaret/pycaret/issues
- **Community**: https://github.com/pycaret/pycaret/discussions

---

## Changelog

### Version 1.0.0 (2025-10-08)
- Initial deployment infrastructure
- Automated dashboard deployment
- Data refresh scheduler
- Alert notification system
- Performance monitoring
- Backup and recovery system
- Docker containerization
- CI/CD pipeline with GitHub Actions

---

## License

This deployment guide is part of the PyCaret project and follows the same license.
