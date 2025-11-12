#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"
LOG_DIR="${SCRIPT_DIR}/logs"
BACKUP_DIR="${SCRIPT_DIR}/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "$LOG_DIR" "$BACKUP_DIR" "$CONFIG_DIR"

validate_environment() {
    log_info "Validating production environment..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_info "Python version: $PYTHON_VERSION"
    
    if ! python3 -c "import grpc, numpy, cryptography, fastapi" 2>/dev/null; then
        log_error "Missing required Python packages"
        log_info "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
        if [ "$AVAILABLE_MEM" -lt 1024 ]; then
            log_warn "Low available memory: ${AVAILABLE_MEM}MB (recommended: 1024MB+)"
        fi
    fi
    
    log_info "Environment validation passed"
}

backup_config() {
    log_info "Creating configuration backup..."
    tar -czf "$BACKUP_DIR/config_backup_${TIMESTAMP}.tar.gz" -C "$CONFIG_DIR" . 2>/dev/null || true
}

start_system() {
    log_info "Starting Aegis Constellation Production System..."
    
    export AEGIS_ENV="production"
    export AEGIS_CONFIG_PATH="$CONFIG_DIR"
    export PYTHONPATH="$SCRIPT_DIR"
    
    MAX_RESTARTS=3
    RESTART_COUNT=0
    
    while [ $RESTART_COUNT -le $MAX_RESTARTS ]; do
        log_info "Starting application (attempt $((RESTART_COUNT + 1))/$((MAX_RESTARTS + 1)))..."
        
        if python3 main.py 2>&1 | tee -a "$LOG_DIR/aegis_production_${TIMESTAMP}.log"; then
            log_info "Application shutdown completed successfully"
            break
        else
            RESTART_COUNT=$((RESTART_COUNT + 1))
            
            log_warn "Application exited (attempt $RESTART_COUNT/$MAX_RESTARTS)"
            
            if [ $RESTART_COUNT -le $MAX_RESTARTS ]; then
                log_info "Restarting in 10 seconds..."
                sleep 10
            else
                log_error "Maximum restart attempts reached. System halted."
                exit 1
            fi
        fi
    done
}

main() {
    log_info "AEGIS CONSTELLATION v5.0 - PRODUCTION DEPLOYMENT"
    log_info "Starting at: $(date)"
    
    validate_environment
    backup_config
    start_system
    
    log_info "Aegis Constellation production session completed"
}

main "$@"
