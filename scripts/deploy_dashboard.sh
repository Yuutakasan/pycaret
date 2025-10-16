#!/bin/bash
#################################################
# PyCaret Dashboard Deployment Script
# Automated dashboard generation and deployment
#################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="${PROJECT_ROOT}/dashboard"
OUTPUT_DIR="${DASHBOARD_DIR}/output"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/deploy_${TIMESTAMP}.log"

# Environment variables with defaults
ENVIRONMENT="${ENVIRONMENT:-production}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8050}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#################################################
# Utility Functions
#################################################

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

check_dependencies() {
    log "Checking dependencies..."

    local deps=("python3" "pip" "git" "docker")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep is not installed"
            return 1
        fi
    done

    log "All dependencies satisfied"
    return 0
}

setup_environment() {
    log "Setting up environment..."

    # Create necessary directories
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

    # Activate virtual environment if it exists
    if [ -d "${PROJECT_ROOT}/venv" ]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
        log "Virtual environment activated"
    fi

    # Install/update dependencies
    if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        pip install -q --upgrade -r "${PROJECT_ROOT}/requirements.txt"
        log "Dependencies updated"
    fi
}

run_tests() {
    log "Running pre-deployment tests..."

    cd "$PROJECT_ROOT"

    # Run unit tests
    if python -m pytest tests/ --tb=short -q; then
        log "Tests passed successfully"
        return 0
    else
        error "Tests failed"
        return 1
    fi
}

generate_dashboard() {
    log "Generating dashboard components..."

    cd "$DASHBOARD_DIR"

    # Run dashboard generation script
    if [ -f "generate_dashboard.py" ]; then
        python generate_dashboard.py --output "$OUTPUT_DIR" --env "$ENVIRONMENT"
        log "Dashboard generated successfully"
    else
        warn "Dashboard generation script not found"
    fi
}

build_assets() {
    log "Building static assets..."

    # Minify CSS/JS if needed
    if command -v npm &> /dev/null; then
        cd "$DASHBOARD_DIR"
        if [ -f "package.json" ]; then
            npm run build 2>/dev/null || true
            log "Assets built successfully"
        fi
    fi
}

deploy_local() {
    log "Deploying dashboard locally..."

    # Kill existing dashboard process
    pkill -f "dashboard.py" || true

    # Start dashboard
    cd "$DASHBOARD_DIR"
    nohup python dashboard.py \
        --port "$DASHBOARD_PORT" \
        --host "0.0.0.0" \
        > "${LOG_DIR}/dashboard_${TIMESTAMP}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "${LOG_DIR}/dashboard.pid"

    log "Dashboard started with PID: $pid"
    log "Dashboard available at http://localhost:${DASHBOARD_PORT}"
}

deploy_docker() {
    log "Deploying dashboard with Docker..."

    cd "$PROJECT_ROOT"

    # Build Docker image
    docker build -f docker/Dockerfile.dashboard \
        -t pycaret-dashboard:latest \
        -t pycaret-dashboard:${TIMESTAMP} \
        .

    # Stop existing container
    docker stop pycaret-dashboard || true
    docker rm pycaret-dashboard || true

    # Run new container
    docker run -d \
        --name pycaret-dashboard \
        -p ${DASHBOARD_PORT}:8050 \
        -v "${OUTPUT_DIR}:/app/output" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        --restart unless-stopped \
        pycaret-dashboard:latest

    log "Dashboard container started"
    log "Dashboard available at http://localhost:${DASHBOARD_PORT}"
}

health_check() {
    log "Running health check..."

    local max_attempts=10
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:${DASHBOARD_PORT}/health" > /dev/null 2>&1; then
            log "Health check passed"
            return 0
        fi

        warn "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done

    error "Health check failed after $max_attempts attempts"
    return 1
}

send_notification() {
    local status=$1
    local message=$2

    # Send notification via webhook if configured
    if [ -n "${WEBHOOK_URL:-}" ]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"status\": \"$status\", \"message\": \"$message\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
            > /dev/null 2>&1 || true
    fi
}

cleanup() {
    log "Cleaning up old logs and artifacts..."

    # Remove logs older than 30 days
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete

    # Remove old Docker images
    if command -v docker &> /dev/null; then
        docker image prune -f --filter "until=720h" || true
    fi

    log "Cleanup completed"
}

rollback() {
    error "Deployment failed, initiating rollback..."

    # Stop current deployment
    if [ -f "${LOG_DIR}/dashboard.pid" ]; then
        kill $(cat "${LOG_DIR}/dashboard.pid") 2>/dev/null || true
    fi

    # Restore from backup if available
    if [ -d "${OUTPUT_DIR}.backup" ]; then
        rm -rf "$OUTPUT_DIR"
        mv "${OUTPUT_DIR}.backup" "$OUTPUT_DIR"
        log "Restored from backup"
    fi

    send_notification "failure" "Deployment failed and rolled back"
}

#################################################
# Main Deployment Flow
#################################################

main() {
    log "Starting PyCaret dashboard deployment"
    log "Environment: $ENVIRONMENT"

    # Set trap for cleanup on exit
    trap 'rollback' ERR

    # Pre-deployment checks
    check_dependencies || exit 1
    setup_environment

    # Backup current deployment
    if [ -d "$OUTPUT_DIR" ]; then
        cp -r "$OUTPUT_DIR" "${OUTPUT_DIR}.backup"
        log "Created backup of current deployment"
    fi

    # Run tests
    if [ "${SKIP_TESTS:-false}" != "true" ]; then
        run_tests || exit 1
    fi

    # Generate and build
    generate_dashboard
    build_assets

    # Deploy based on method
    case "${DEPLOY_METHOD:-local}" in
        docker)
            deploy_docker
            ;;
        local)
            deploy_local
            ;;
        *)
            error "Unknown deployment method: ${DEPLOY_METHOD}"
            exit 1
            ;;
    esac

    # Post-deployment checks
    sleep 5
    health_check || exit 1

    # Cleanup
    cleanup

    # Remove backup on success
    rm -rf "${OUTPUT_DIR}.backup"

    # Send success notification
    send_notification "success" "Dashboard deployed successfully"

    log "Deployment completed successfully"
    log "Log file: $LOG_FILE"
}

# Run main function
main "$@"
