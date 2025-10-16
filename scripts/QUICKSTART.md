# PyCaret Deployment - Quick Start Guide

Get your PyCaret Dashboard up and running in 5 minutes!

## 🚀 Quick Deployment Options

### Option 1: One-Command Local Deployment (Fastest)

```bash
cd /mnt/d/github/pycaret
./scripts/deploy_dashboard.sh
```

Dashboard will be available at: **http://localhost:8050**

### Option 2: Docker Deployment (Recommended)

```bash
cd /mnt/d/github/pycaret/docker
docker-compose up -d
```

Services:
- Dashboard: **http://localhost:8050**
- All automation services running in background

### Option 3: Production Deployment

```bash
# Set environment
export ENVIRONMENT=production
export DEPLOY_METHOD=docker
export WORKERS=4

# Deploy
./scripts/deploy_dashboard.sh

# Verify
curl http://localhost:8050/health
```

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Docker & Docker Compose (for Docker deployment)
- [ ] 4GB+ RAM available
- [ ] 10GB+ disk space

## 🔧 Essential Scripts

All scripts are located in `/mnt/d/github/pycaret/scripts/`

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy_dashboard.sh` | Deploy dashboard | `./deploy_dashboard.sh` |
| `data_refresh.py` | Auto-refresh data | `python data_refresh.py` |
| `performance_monitor.py` | Monitor system | `python performance_monitor.py --status` |
| `alert_system.py` | Send alerts | `python alert_system.py --test` |
| `backup_restore.sh` | Backup/restore | `./backup_restore.sh backup` |

## 🐳 Docker Commands Cheat Sheet

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Restart service
docker-compose restart dashboard

# Stop all services
docker-compose down

# Scale dashboard instances
docker-compose up -d --scale dashboard=3

# View resource usage
docker stats
```

## 📊 Verify Deployment

```bash
# 1. Check dashboard health
curl http://localhost:8050/health

# 2. View performance metrics
python scripts/performance_monitor.py --status

# 3. Check data refresh status
python scripts/data_refresh.py --status

# 4. View recent alerts
python scripts/alert_system.py --stats
```

## 🔍 Monitoring Commands

```bash
# View all logs
tail -f logs/*.log

# Check deployment log
tail -f logs/deploy_*.log

# Monitor performance in real-time
watch -n 5 'python scripts/performance_monitor.py --status'

# View Docker container logs
docker-compose logs -f
```

## 🛠️ Common Tasks

### Create Manual Backup
```bash
./scripts/backup_restore.sh backup
```

### Restore from Backup
```bash
./scripts/backup_restore.sh list
./scripts/backup_restore.sh restore backups/pycaret_backup_YYYYMMDD_HHMMSS.tar.gz
```

### Refresh Data Manually
```bash
python scripts/data_refresh.py --once
```

### Send Test Alert
```bash
python scripts/alert_system.py --test
```

### Restart Dashboard
```bash
# Local deployment
pkill -f "dashboard.py"
./scripts/deploy_dashboard.sh

# Docker deployment
docker-compose restart dashboard
```

## ⚙️ Configuration

### Environment Variables (.env)

Create `.env` file in project root:

```bash
# Application
ENVIRONMENT=production
DASHBOARD_PORT=8050
WORKERS=4

# Monitoring
WEBHOOK_URL=https://your-webhook-url.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Backup
AWS_S3_BUCKET=your-backup-bucket
BACKUP_ENCRYPTION_KEY=your-encryption-key
```

### Quick Configuration Setup

```bash
# Create config directory
mkdir -p config

# Generate default configurations
python scripts/data_refresh.py --create-config

# Edit configurations as needed
nano config/refresh_config.json
nano config/alert_config.json
nano config/monitor_config.json
```

## 🔄 Automated Services

### Enable Automation (Systemd)

```bash
# Create service file
sudo tee /etc/systemd/system/pycaret-dashboard.service <<EOF
[Unit]
Description=PyCaret Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/mnt/d/github/pycaret
ExecStart=/mnt/d/github/pycaret/scripts/deploy_dashboard.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable pycaret-dashboard
sudo systemctl start pycaret-dashboard
sudo systemctl status pycaret-dashboard
```

### Schedule Backups (Cron)

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /mnt/d/github/pycaret && ./scripts/backup_restore.sh backup
```

## 🚨 Troubleshooting

### Dashboard Not Starting
```bash
# Check if port is in use
sudo lsof -i :8050

# Kill existing process
pkill -f "dashboard.py"

# Check logs
tail -f logs/dashboard_*.log

# Restart
./scripts/deploy_dashboard.sh
```

### Docker Container Issues
```bash
# Restart container
docker-compose restart dashboard

# Rebuild container
docker-compose up -d --build dashboard

# View detailed logs
docker-compose logs --tail=100 dashboard
```

### High Memory Usage
```bash
# Check current usage
python scripts/performance_monitor.py --status

# Reduce workers
WORKERS=2 ./scripts/deploy_dashboard.sh
```

## 📚 Documentation

- **Full Deployment Guide**: `/mnt/d/github/pycaret/docs/deployment_guide.md`
- **Scripts Documentation**: `/mnt/d/github/pycaret/scripts/README.md`
- **Docker Configuration**: `/mnt/d/github/pycaret/docker/`

## 🔐 Security Best Practices

1. **Never commit secrets to Git**
   ```bash
   # Use .env file (already in .gitignore)
   cp .env.example .env
   nano .env
   ```

2. **Encrypt backups**
   ```bash
   export BACKUP_ENCRYPTION_KEY="your-secure-key"
   ./scripts/backup_restore.sh backup
   ```

3. **Use HTTPS in production**
   ```bash
   # Configure SSL in nginx or use reverse proxy
   ```

4. **Regular security updates**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 🎯 Next Steps

1. ✅ Deploy dashboard (you are here!)
2. 📊 Configure monitoring and alerts
3. 🔄 Set up automated data refresh
4. 💾 Schedule regular backups
5. 🚀 Deploy to production environment
6. 📈 Monitor performance and optimize

## 💡 Tips

- **Development**: Use local deployment for faster iteration
- **Staging**: Use Docker deployment for testing
- **Production**: Use Docker with CI/CD pipeline
- **Monitoring**: Enable all monitoring services
- **Backups**: Schedule daily backups with S3 upload

## 📞 Support

- **Issues**: https://github.com/pycaret/pycaret/issues
- **Documentation**: https://pycaret.org/docs
- **Community**: https://github.com/pycaret/pycaret/discussions

---

**Ready to deploy? Run:** `./scripts/deploy_dashboard.sh`
