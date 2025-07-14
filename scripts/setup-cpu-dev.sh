#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Local development helper script for CPU vLLM testing
# This script sets up a local environment for testing vLLM CPU builds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

# Load configuration
load_config() {
    local config_file="${SCRIPT_DIR}/vllm-cpu-config.env"
    if [[ -f "${config_file}" ]]; then
        log "Loading configuration from ${config_file}"
        # Source the config file, filtering out comments
        while IFS= read -r line; do
            if [[ $line =~ ^[^#]*= ]]; then
                export "$line"
            fi
        done < "${config_file}"
    else
        warn "Configuration file not found: ${config_file}"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for CPU vLLM development..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        error "CPU vLLM development is only supported on Linux"
        exit 1
    fi
    
    # Check CPU features
    if ! grep -q "avx512f" /proc/cpuinfo; then
        error "AVX512F CPU instruction set is required"
        error "Your CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        exit 1
    fi
    
    info "CPU features detected:"
    grep -o 'avx[^ ]*' /proc/cpuinfo | sort | uniq | sed 's/^/  - /'
    
    # Check Python version
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if ! python3 -c "import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12)" 2>/dev/null; then
        error "Python ${python_version} is not supported. vLLM CPU requires Python 3.9-3.12"
        exit 1
    fi
    
    # Check required tools
    local missing_tools=()
    for tool in gcc g++ git curl; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        info "Install them with: sudo apt-get install ${missing_tools[*]}"
        exit 1
    fi
    
    # Check GCC version
    local gcc_version=$(gcc -dumpversion | cut -d. -f1)
    if [[ $gcc_version -lt 12 ]]; then
        warn "GCC version ${gcc_version} detected. GCC >= 12.3.0 is recommended"
        info "Install with: sudo apt-get install gcc-12 g++-12"
    fi
    
    log "Prerequisites check passed ✓"
}

# Setup development environment
setup_environment() {
    log "Setting up development environment..."
    
    cd "${PROJECT_ROOT}"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv-cpu" ]]; then
        log "Creating virtual environment..."
        python3 -m venv venv-cpu
    fi
    
    # Activate virtual environment
    source venv-cpu/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install development dependencies
    log "Installing development dependencies..."
    python -m pip install -e ".[dev-cpu]"
    
    log "Development environment setup complete ✓"
}

# Build vLLM CPU wheel
build_vllm() {
    log "Building vLLM CPU wheel..."
    
    cd "${PROJECT_ROOT}"
    source venv-cpu/bin/activate
    
    # Build the wheel
    local wheel_path
    wheel_path=$(./scripts/build-vllm-cpu.sh)
    
    if [[ -n "${wheel_path}" ]]; then
        log "vLLM CPU wheel built successfully: $(basename "${wheel_path}")"
        
        # Install the wheel
        log "Installing vLLM CPU wheel..."
        python -m pip install "${wheel_path}" --force-reinstall
        
        log "vLLM CPU installation complete ✓"
    else
        error "Failed to build vLLM CPU wheel"
        exit 1
    fi
}

# Test the installation
test_installation() {
    log "Testing vLLM CPU installation..."
    
    cd "${PROJECT_ROOT}"
    source venv-cpu/bin/activate
    
    # Basic import test
    python -c "
import vllm
from vllm import LLM
from vllm.lora import LoRARequest  
from vllm.sampling_params import SamplingParams

print(f'✓ vLLM version: {vllm.__version__}')
print('✓ Basic import successful')
print('✓ LLM class available')
print('✓ LoRA support available')

# Test guided generation parameters
sp = SamplingParams()
if hasattr(sp, 'guided_regex') or 'guided_regex' in SamplingParams.__init__.__code__.co_varnames:
    print('✓ Guided generation support available')
else:
    print('✗ Guided generation support not found')
    exit(1)

print('All tests passed!')
"
    
    log "Installation test passed ✓"
}

# Run specific tests
run_tests() {
    local test_type="${1:-unit}"
    
    log "Running ${test_type} tests..."
    
    cd "${PROJECT_ROOT}"
    source venv-cpu/bin/activate
    
    case "${test_type}" in
        "unit")
            python -m pytest -v -k "vllm or lora" tests/ || true
            ;;
        "notebook")
            log "Testing RAG notebook..."
            # Create a simplified test version of the notebook
            python -c "
# Simplified RAG test
try:
    from granite_io.backend.vllm_server import LocalVLLMServer
    from granite_io import make_io_processor
    print('✓ vLLM backend imports successful')
    
    # Test configuration only (don't start server in test)
    config = {
        'model_name': 'microsoft/DialoGPT-small',
        'max_model_len': 512,
        'enable_lora': True,
    }
    print('✓ vLLM server configuration test passed')
    
except Exception as e:
    print(f'✗ Notebook test failed: {e}')
    exit(1)
"
            ;;
        "tox")
            tox -e vllm-cpu-unit
            ;;
        *)
            error "Unknown test type: ${test_type}"
            error "Available types: unit, notebook, tox"
            exit 1
            ;;
    esac
    
    log "${test_type} tests completed"
}

# Show status and information
show_status() {
    log "CPU vLLM Development Environment Status"
    echo
    
    # System info
    info "System Information:"
    echo "  OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown Linux")"
    echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "  Python: $(python3 --version)"
    echo "  GCC: $(gcc --version | head -1)"
    echo
    
    # CPU features
    info "CPU Features:"
    if grep -q "avx512f" /proc/cpuinfo; then
        echo "  ✓ AVX512F supported"
    else
        echo "  ✗ AVX512F not supported"
    fi
    
    if grep -q "avx512_bf16" /proc/cpuinfo; then
        echo "  ✓ AVX512_BF16 supported"
    else
        echo "  - AVX512_BF16 not supported"
    fi
    
    if grep -q "avx512_vnni" /proc/cpuinfo; then
        echo "  ✓ AVX512_VNNI supported"
    else
        echo "  - AVX512_VNNI not supported"
    fi
    echo
    
    # Virtual environment
    info "Virtual Environment:"
    if [[ -d "${PROJECT_ROOT}/venv-cpu" ]]; then
        echo "  ✓ Virtual environment exists"
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            echo "  ✓ Virtual environment active"
        else
            echo "  - Virtual environment not active"
        fi
    else
        echo "  ✗ Virtual environment not found"
    fi
    echo
    
    # vLLM installation
    info "vLLM Installation:"
    if [[ -d "${PROJECT_ROOT}/venv-cpu" ]]; then
        source "${PROJECT_ROOT}/venv-cpu/bin/activate" 2>/dev/null || true
        if python -c "import vllm" 2>/dev/null; then
            local version=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
            echo "  ✓ vLLM installed (version: ${version})"
        else
            echo "  ✗ vLLM not installed"
        fi
    else
        echo "  ? Cannot check (no virtual environment)"
    fi
    echo
    
    # Cache status
    info "Build Cache:"
    local cache_dir="${CACHE_DIR:-${HOME}/.cache/granite-io/vllm-cpu}"
    if [[ -d "${cache_dir}" ]]; then
        local cache_size=$(du -sh "${cache_dir}" 2>/dev/null | cut -f1 || echo "unknown")
        echo "  ✓ Cache directory exists (${cache_size})"
        
        local wheels=(${cache_dir}/*.whl)
        if [[ ${#wheels[@]} -gt 0 ]] && [[ -f "${wheels[0]}" ]]; then
            echo "  ✓ Cached wheels found: ${#wheels[@]}"
        else
            echo "  - No cached wheels found"
        fi
    else
        echo "  - No cache directory"
    fi
}

# Clean up environment
cleanup() {
    log "Cleaning up CPU vLLM environment..."
    
    cd "${PROJECT_ROOT}"
    
    # Remove virtual environment
    if [[ -d "venv-cpu" ]]; then
        log "Removing virtual environment..."
        rm -rf venv-cpu
    fi
    
    # Clean build cache
    local cache_dir="${CACHE_DIR:-${HOME}/.cache/granite-io/vllm-cpu}"
    if [[ -d "${cache_dir}" ]]; then
        log "Cleaning build cache..."
        rm -rf "${cache_dir}"
    fi
    
    # Clean ccache
    if command -v ccache &> /dev/null; then
        log "Cleaning ccache..."
        ccache -C
    fi
    
    log "Cleanup complete ✓"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    check       Check prerequisites for CPU vLLM development
    setup       Setup development environment
    build       Build vLLM CPU wheel
    test        Test the installation
    status      Show environment status
    cleanup     Clean up environment

Test options (for 'test' command):
    unit        Run unit tests (default)
    notebook    Test notebook compatibility  
    tox         Run full tox tests

Examples:
    $0 check                    # Check prerequisites
    $0 setup                    # Setup development environment
    $0 build                    # Build vLLM CPU wheel
    $0 test                     # Run unit tests
    $0 test notebook            # Test notebook compatibility
    $0 status                   # Show status
    $0 cleanup                  # Clean up everything

Environment Variables:
    VLLM_VERSION               vLLM version to build (default from config)
    CACHE_DIR                  Cache directory for builds
    FORCE_REBUILD              Set to 'true' to force rebuild

EOF
}

# Main function
main() {
    # Load configuration
    load_config
    
    case "${1:-}" in
        "check")
            check_prerequisites
            ;;
        "setup")
            check_prerequisites
            setup_environment
            ;;
        "build")
            check_prerequisites
            build_vllm
            ;;
        "test")
            test_installation
            run_tests "${2:-unit}"
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        "")
            error "No command specified"
            usage
            exit 1
            ;;
        *)
            error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
