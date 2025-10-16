#!/bin/bash
#################################################
# PyCaret Backup and Recovery System
# Automated backup and disaster recovery
#################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_ROOT}/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pycaret_backup_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/backup_${TIMESTAMP}.log"

# Retention settings
DAILY_RETENTION=7
WEEKLY_RETENTION=4
MONTHLY_RETENTION=12

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

#################################################
# Backup Functions
#################################################

create_backup_dir() {
    log "Creating backup directory..."
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$LOG_DIR"
}

backup_database() {
    log "Backing up databases..."

    local db_backup_dir="$BACKUP_DIR/$BACKUP_NAME/databases"
    mkdir -p "$db_backup_dir"

    # Backup SQLite databases
    if [ -d "${PROJECT_ROOT}/.swarm" ]; then
        find "${PROJECT_ROOT}/.swarm" -name "*.db" -type f | while read -r db; do
            local db_name=$(basename "$db")
            cp "$db" "$db_backup_dir/$db_name"
            log "Backed up database: $db_name"
        done
    fi

    # Create database manifest
    cat > "$db_backup_dir/manifest.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "databases": $(find "$db_backup_dir" -name "*.db" -type f -printf '"%f"\n' | jq -Rs 'split("\n") | map(select(. != ""))')
}
EOF
}

backup_configuration() {
    log "Backing up configuration files..."

    local config_backup_dir="$BACKUP_DIR/$BACKUP_NAME/config"
    mkdir -p "$config_backup_dir"

    # Backup config directory
    if [ -d "${PROJECT_ROOT}/config" ]; then
        cp -r "${PROJECT_ROOT}/config" "$config_backup_dir/"
    fi

    # Backup environment files (excluding secrets)
    if [ -f "${PROJECT_ROOT}/.env.example" ]; then
        cp "${PROJECT_ROOT}/.env.example" "$config_backup_dir/"
    fi

    # Backup Docker configs
    if [ -d "${PROJECT_ROOT}/docker" ]; then
        cp -r "${PROJECT_ROOT}/docker" "$config_backup_dir/"
    fi

    log "Configuration backed up"
}

backup_data() {
    log "Backing up data files..."

    local data_backup_dir="$BACKUP_DIR/$BACKUP_NAME/data"
    mkdir -p "$data_backup_dir"

    # Backup dashboard output
    if [ -d "${PROJECT_ROOT}/dashboard/output" ]; then
        cp -r "${PROJECT_ROOT}/dashboard/output" "$data_backup_dir/"
    fi

    # Backup logs (last 7 days)
    if [ -d "$LOG_DIR" ]; then
        find "$LOG_DIR" -name "*.log" -mtime -7 -type f -exec cp {} "$data_backup_dir/" \;
    fi

    # Backup any data directories
    if [ -d "${PROJECT_ROOT}/data" ]; then
        cp -r "${PROJECT_ROOT}/data" "$data_backup_dir/"
    fi

    log "Data files backed up"
}

backup_code() {
    log "Backing up source code..."

    local code_backup_dir="$BACKUP_DIR/$BACKUP_NAME/code"
    mkdir -p "$code_backup_dir"

    # Create git bundle if in a git repo
    if [ -d "${PROJECT_ROOT}/.git" ]; then
        cd "$PROJECT_ROOT"
        git bundle create "$code_backup_dir/repo.bundle" --all
        git log -1 --format="%H" > "$code_backup_dir/commit.txt"
        log "Git repository bundled"
    fi

    # Backup critical scripts
    if [ -d "${PROJECT_ROOT}/scripts" ]; then
        cp -r "${PROJECT_ROOT}/scripts" "$code_backup_dir/"
    fi

    log "Source code backed up"
}

create_backup_manifest() {
    log "Creating backup manifest..."

    local manifest_file="$BACKUP_DIR/$BACKUP_NAME/manifest.json"

    # Calculate backup size
    local backup_size=$(du -sb "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)

    cat > "$manifest_file" <<EOF
{
    "backup_name": "$BACKUP_NAME",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "backup_type": "full",
    "size_bytes": $backup_size,
    "components": {
        "databases": $([ -d "$BACKUP_DIR/$BACKUP_NAME/databases" ] && echo "true" || echo "false"),
        "configuration": $([ -d "$BACKUP_DIR/$BACKUP_NAME/config" ] && echo "true" || echo "false"),
        "data": $([ -d "$BACKUP_DIR/$BACKUP_NAME/data" ] && echo "true" || echo "false"),
        "code": $([ -d "$BACKUP_DIR/$BACKUP_NAME/code" ] && echo "true" || echo "false")
    }
}
EOF

    log "Backup manifest created"
}

compress_backup() {
    log "Compressing backup..."

    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

    local compressed_size=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
    log "Backup compressed: ${compressed_size}"

    # Remove uncompressed backup
    rm -rf "$BACKUP_NAME"
}

encrypt_backup() {
    if [ -n "${BACKUP_ENCRYPTION_KEY:-}" ]; then
        log "Encrypting backup..."

        cd "$BACKUP_DIR"
        openssl enc -aes-256-cbc -salt \
            -in "${BACKUP_NAME}.tar.gz" \
            -out "${BACKUP_NAME}.tar.gz.enc" \
            -pass pass:"$BACKUP_ENCRYPTION_KEY"

        # Remove unencrypted backup
        rm "${BACKUP_NAME}.tar.gz"

        log "Backup encrypted"
    fi
}

upload_backup() {
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"

    if [ -f "${backup_file}.enc" ]; then
        backup_file="${backup_file}.enc"
    fi

    # Upload to S3 if configured
    if [ -n "${AWS_S3_BUCKET:-}" ]; then
        log "Uploading backup to S3..."

        if command -v aws &> /dev/null; then
            aws s3 cp "$backup_file" "s3://${AWS_S3_BUCKET}/backups/$(basename $backup_file)"
            log "Backup uploaded to S3"
        else
            warn "AWS CLI not found, skipping S3 upload"
        fi
    fi

    # Upload to custom endpoint if configured
    if [ -n "${BACKUP_UPLOAD_URL:-}" ]; then
        log "Uploading backup to remote server..."
        curl -X POST "$BACKUP_UPLOAD_URL" \
            -F "file=@$backup_file" \
            -F "timestamp=$TIMESTAMP" \
            > /dev/null 2>&1 || warn "Remote upload failed"
    fi
}

cleanup_old_backups() {
    log "Cleaning up old backups..."

    cd "$BACKUP_DIR"

    # Keep daily backups for DAILY_RETENTION days
    find . -maxdepth 1 -name "pycaret_backup_*.tar.gz*" -mtime +$DAILY_RETENTION -delete

    # Keep weekly backups (Sundays)
    # Keep monthly backups (1st of month)
    # This is a simplified version - enhance as needed

    local cleaned=$(find . -maxdepth 1 -name "pycaret_backup_*.tar.gz*" | wc -l)
    log "Cleanup complete. $cleaned backups retained."
}

#################################################
# Restore Functions
#################################################

list_backups() {
    log "Available backups:"

    if [ ! -d "$BACKUP_DIR" ]; then
        warn "No backup directory found"
        return 1
    fi

    cd "$BACKUP_DIR"
    local backups=$(find . -maxdepth 1 -name "pycaret_backup_*.tar.gz*" -type f | sort -r)

    if [ -z "$backups" ]; then
        warn "No backups found"
        return 1
    fi

    echo "$backups" | while read -r backup; do
        local size=$(du -h "$backup" | cut -f1)
        local date=$(stat -c %y "$backup" | cut -d' ' -f1)
        echo "  $(basename $backup) - Size: $size - Date: $date"
    done
}

restore_backup() {
    local backup_file="$1"

    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        return 1
    fi

    log "Restoring from backup: $(basename $backup_file)"

    # Create restore directory
    local restore_dir="${PROJECT_ROOT}/restore_${TIMESTAMP}"
    mkdir -p "$restore_dir"

    # Decrypt if needed
    if [[ "$backup_file" == *.enc ]]; then
        if [ -z "${BACKUP_ENCRYPTION_KEY:-}" ]; then
            error "Encryption key required for encrypted backup"
            return 1
        fi

        log "Decrypting backup..."
        openssl enc -aes-256-cbc -d \
            -in "$backup_file" \
            -out "${backup_file%.enc}" \
            -pass pass:"$BACKUP_ENCRYPTION_KEY"

        backup_file="${backup_file%.enc}"
    fi

    # Extract backup
    log "Extracting backup..."
    tar -xzf "$backup_file" -C "$restore_dir"

    # Find extracted directory
    local extracted_dir=$(find "$restore_dir" -maxdepth 1 -type d -name "pycaret_backup_*" | head -1)

    if [ -z "$extracted_dir" ]; then
        error "Failed to find extracted backup"
        return 1
    fi

    # Restore components
    restore_databases "$extracted_dir"
    restore_configuration "$extracted_dir"
    restore_data "$extracted_dir"

    log "Restore completed. Review changes in: $restore_dir"
    log "Manual verification required before applying to production"
}

restore_databases() {
    local backup_dir="$1/databases"

    if [ ! -d "$backup_dir" ]; then
        warn "No database backup found"
        return
    fi

    log "Restoring databases..."

    # Create backup of current databases
    if [ -d "${PROJECT_ROOT}/.swarm" ]; then
        cp -r "${PROJECT_ROOT}/.swarm" "${PROJECT_ROOT}/.swarm.backup_${TIMESTAMP}"
    fi

    # Restore databases
    cp -r "$backup_dir"/*.db "${PROJECT_ROOT}/.swarm/" 2>/dev/null || true

    log "Databases restored"
}

restore_configuration() {
    local backup_dir="$1/config"

    if [ ! -d "$backup_dir" ]; then
        warn "No configuration backup found"
        return
    fi

    log "Restoring configuration..."

    # Backup current config
    if [ -d "${PROJECT_ROOT}/config" ]; then
        cp -r "${PROJECT_ROOT}/config" "${PROJECT_ROOT}/config.backup_${TIMESTAMP}"
    fi

    # Restore config
    cp -r "$backup_dir/config" "${PROJECT_ROOT}/" 2>/dev/null || true

    log "Configuration restored"
}

restore_data() {
    local backup_dir="$1/data"

    if [ ! -d "$backup_dir" ]; then
        warn "No data backup found"
        return
    fi

    log "Restoring data files..."

    # Restore dashboard output
    if [ -d "$backup_dir/output" ]; then
        mkdir -p "${PROJECT_ROOT}/dashboard/output"
        cp -r "$backup_dir/output"/* "${PROJECT_ROOT}/dashboard/output/"
    fi

    log "Data files restored"
}

verify_backup() {
    local backup_file="$1"

    log "Verifying backup integrity..."

    # Check if file exists
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found"
        return 1
    fi

    # Test archive integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" > /dev/null 2>&1; then
            log "Backup integrity verified"
            return 0
        else
            error "Backup is corrupted"
            return 1
        fi
    fi

    log "Verification complete"
}

#################################################
# Main Functions
#################################################

perform_backup() {
    log "Starting backup process..."

    create_backup_dir
    backup_database
    backup_configuration
    backup_data
    backup_code
    create_backup_manifest
    compress_backup
    encrypt_backup
    upload_backup
    cleanup_old_backups

    log "Backup completed successfully: ${BACKUP_NAME}.tar.gz"
}

show_usage() {
    cat <<EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    backup              Create a new backup
    restore [FILE]      Restore from backup file
    list                List available backups
    verify [FILE]       Verify backup integrity
    cleanup             Clean up old backups

Environment Variables:
    BACKUP_DIR              Backup directory (default: ./backups)
    BACKUP_ENCRYPTION_KEY   Encryption key for backups
    AWS_S3_BUCKET          S3 bucket for remote backups
    BACKUP_UPLOAD_URL      Custom upload endpoint

Examples:
    $0 backup
    $0 restore backups/pycaret_backup_20251008_120000.tar.gz
    $0 list
    $0 verify backups/pycaret_backup_20251008_120000.tar.gz

EOF
}

#################################################
# Main Entry Point
#################################################

main() {
    local command="${1:-}"

    case "$command" in
        backup)
            perform_backup
            ;;
        restore)
            if [ -z "${2:-}" ]; then
                error "Backup file required for restore"
                show_usage
                exit 1
            fi
            restore_backup "$2"
            ;;
        list)
            list_backups
            ;;
        verify)
            if [ -z "${2:-}" ]; then
                error "Backup file required for verification"
                show_usage
                exit 1
            fi
            verify_backup "$2"
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
