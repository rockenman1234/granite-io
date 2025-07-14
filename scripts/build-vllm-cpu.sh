#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Script to build vLLM CPU wheel from source
# This script handles the compilation and caching of vLLM CPU build

set -euo pipefail

# Load configuration from config file if it exists
CONFIG_FILE="${SCRIPT_DIR:-$(dirname $0)}/vllm-cpu-config.env"
if [[ -f "${CONFIG_FILE}" ]]; then
    # Source configuration, filtering out comments and empty lines
    while IFS= read -r line; do
        if [[ $line =~ ^[A-Z_].*= ]] && [[ ! $line =~ ^# ]]; then
            # Only export if not already set
            var_name=$(echo "$line" | cut -d= -f1)
            if [[ -z "${!var_name:-}" ]]; then
                export "$line"
            fi
        fi
    done < "${CONFIG_FILE}"
fi

# Default values (can be overridden by config file or environment)
VLLM_VERSION="${VLLM_VERSION:-v0.6.4.post1}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/granite-io/vllm-cpu}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
BUILD_DIR="${BUILD_DIR:-/tmp/vllm-cpu-build}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        error "vLLM CPU build is only supported on Linux"
        exit 1
    fi
    
    # Check CPU features
    if ! grep -q "avx512f" /proc/cpuinfo; then
        error "AVX512F CPU instruction set is required for vLLM CPU build"
        exit 1
    fi
    
    # Check compiler
    if ! command -v gcc &> /dev/null; then
        error "gcc compiler not found. Please install gcc>=12.3.0"
        exit 1
    fi
    
    local gcc_version=$(gcc -dumpversion | cut -d. -f1)
    if [[ $gcc_version -lt 12 ]]; then
        warn "gcc version is ${gcc_version}. gcc>=12.3.0 is recommended"
    fi
    
    # Check Python version
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if ! python3 -c "import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12)" 2>/dev/null; then
        error "Python version ${python_version} is not supported. vLLM CPU requires Python 3.9-3.12"
        exit 1
    fi
    
    log "System requirements check passed"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    # Check if running in container or CI
    if [[ "${CI:-false}" == "true" ]] || [[ -f /.dockerenv ]]; then
        log "Running in CI/container, installing system dependencies with apt"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y
        apt-get install -y --no-install-recommends \
            ccache \
            git \
            curl \
            wget \
            ca-certificates \
            gcc-12 \
            g++-12 \
            libtcmalloc-minimal4 \
            libnuma-dev \
            ffmpeg \
            libsm6 \
            libxext6 \
            libgl1 \
            jq \
            lsof
        
        # Set gcc-12 as default
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
    else
        warn "Not running in CI/container. Please ensure the following dependencies are installed:"
        warn "  - gcc-12, g++-12"
        warn "  - libtcmalloc-minimal4, libnuma-dev"
        warn "  - ccache, git, curl, wget"
    fi
}

# Setup ccache for faster compilation
setup_ccache() {
    if command -v ccache &> /dev/null; then
        log "Setting up ccache for faster compilation"
        export CC="ccache gcc"
        export CXX="ccache g++"
        export CCACHE_DIR="${CACHE_DIR}/ccache"
        mkdir -p "${CCACHE_DIR}"
        ccache --set-config=max_size=5G
        ccache --set-config=compression=true
    fi
}

# Check if cached wheel exists and is valid
check_cached_wheel() {
    local wheel_path="${CACHE_DIR}/vllm-${VLLM_VERSION}-cp*-cp*-linux_x86_64.whl"
    
    if [[ "${FORCE_REBUILD}" == "true" ]]; then
        log "Force rebuild requested, ignoring cached wheel"
        return 1
    fi
    
    # Check if wheel exists
    local wheels=(${wheel_path})
    if [[ ${#wheels[@]} -eq 0 ]] || [[ ! -f "${wheels[0]}" ]]; then
        log "No cached wheel found for vLLM ${VLLM_VERSION}"
        return 1
    fi
    
    local wheel_file="${wheels[0]}"
    log "Found cached wheel: $(basename ${wheel_file})"
    
    # Test the wheel by importing vllm
    if python3 -c "
import tempfile
import subprocess
import sys
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # Install wheel in temporary directory
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        '${wheel_file}', '--target', tmpdir, '--quiet'
    ], check=True)
    
    # Add to path and test import
    sys.path.insert(0, tmpdir)
    try:
        import vllm
        # Test CPU backend availability
        from vllm import LLM
        print('Cached wheel is valid')
    except Exception as e:
        print(f'Cached wheel test failed: {e}')
        exit(1)
" 2>/dev/null; then
        log "Cached wheel is valid and functional"
        echo "${wheel_file}"
        return 0
    else
        warn "Cached wheel exists but failed validation, will rebuild"
        return 1
    fi
}

# Clone and checkout specific vLLM version
prepare_source() {
    log "Preparing vLLM source code (version: ${VLLM_VERSION})"
    
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    
    cd "${BUILD_DIR}"
    
    # Clone with specific tag to avoid downloading entire history
    git clone --depth 1 --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git
    cd vllm
    
    log "vLLM source prepared at ${BUILD_DIR}/vllm"
}

# Build vLLM CPU wheel
build_wheel() {
    log "Building vLLM CPU wheel..."
    
    cd "${BUILD_DIR}/vllm"
    
    # Install Python build dependencies
    log "Installing Python build dependencies..."
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install --upgrade build packaging ninja
    
    # Install CPU-specific build requirements
    log "Installing CPU build requirements..."
    if [[ -f "requirements-cpu.txt" ]]; then
        python3 -m pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    elif [[ -f "requirements/cpu-build.txt" ]]; then
        python3 -m pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    else
        # Fallback to manual installation
        log "Installing fallback dependencies..."
        python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install numpy cmake ninja packaging setuptools-scm
    fi
    
    # Set environment variables for CPU build
    export VLLM_TARGET_DEVICE=cpu
    export CMAKE_BUILD_TYPE=Release
    
    # Detect CPU features
    if grep -q "avx512_bf16" /proc/cpuinfo; then
        log "AVX512_BF16 detected, enabling optimizations"
        export VLLM_CPU_AVX512BF16=1
    else
        log "AVX512_BF16 not detected, using standard AVX512"
        export VLLM_CPU_AVX512BF16=0
    fi
    
    if grep -q "avx512_vnni" /proc/cpuinfo; then
        log "AVX512_VNNI detected, enabling optimizations"
        export VLLM_CPU_AVX512VNNI=1
    else
        log "AVX512_VNNI not detected"
        export VLLM_CPU_AVX512VNNI=0
    fi
    
    # Build wheel
    log "Starting wheel build (this may take 20-60 minutes)..."
    local start_time=$(date +%s)
    
    python3 -m build --wheel --no-isolation -x
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "Wheel build completed in $((duration / 60))m $((duration % 60))s"
    
    # Find the built wheel
    local wheel_file=$(find dist -name "*.whl" | head -1)
    if [[ -z "${wheel_file}" ]]; then
        error "No wheel file found in dist directory"
        exit 1
    fi
    
    log "Built wheel: $(basename ${wheel_file})"
    echo "${BUILD_DIR}/vllm/${wheel_file}"
}

# Cache the built wheel
cache_wheel() {
    local wheel_file="$1"
    
    mkdir -p "${CACHE_DIR}"
    local cached_wheel="${CACHE_DIR}/$(basename ${wheel_file})"
    
    log "Caching wheel to ${cached_wheel}"
    cp "${wheel_file}" "${cached_wheel}"
    
    # Create a simple metadata file
    cat > "${CACHE_DIR}/build_info.json" << EOF
{
    "version": "${VLLM_VERSION}",
    "build_date": "$(date -Iseconds)",
    "wheel_file": "$(basename ${wheel_file})",
    "gcc_version": "$(gcc -dumpversion)",
    "python_version": "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')",
    "cpu_features": {
        "avx512f": $(grep -q "avx512f" /proc/cpuinfo && echo "true" || echo "false"),
        "avx512_bf16": $(grep -q "avx512_bf16" /proc/cpuinfo && echo "true" || echo "false"),
        "avx512_vnni": $(grep -q "avx512_vnni" /proc/cpuinfo && echo "true" || echo "false")
    }
}
EOF
    
    echo "${cached_wheel}"
}

# Test the built wheel
test_wheel() {
    local wheel_file="$1"
    
    log "Testing built wheel..."
    
    # Create temporary test environment
    local test_dir=$(mktemp -d)
    cd "${test_dir}"
    
    # Install the wheel
    python3 -m pip install "${wheel_file}" --quiet
    
    # Test basic import
    python3 -c "
import vllm
from vllm import LLM
print(f'vLLM version: {vllm.__version__}')
print('Basic import test passed')
"
    
    # Test CPU backend detection
    python3 -c "
import vllm
from vllm.platforms import current_platform
print(f'Platform: {current_platform}')
print('CPU backend test passed')
"
    
    # Test LoRA support (basic check)
    python3 -c "
try:
    from vllm.lora import LoRARequest
    print('LoRA support detected')
except ImportError as e:
    print(f'LoRA support check failed: {e}')
    exit(1)
"
    
    # Test guided generation support (basic check)
    python3 -c "
try:
    from vllm.sampling_params import SamplingParams
    # Check if guided_regex parameter exists
    sp = SamplingParams()
    if hasattr(sp, 'guided_regex') or 'guided_regex' in SamplingParams.__init__.__code__.co_varnames:
        print('Guided generation support detected')
    else:
        print('Guided generation support not found')
        exit(1)
except Exception as e:
    print(f'Guided generation check failed: {e}')
    exit(1)
"
    
    # Cleanup
    cd /
    rm -rf "${test_dir}"
    
    log "Wheel testing completed successfully"
}

# Clean up build directory
cleanup() {
    if [[ -n "${BUILD_DIR:-}" ]] && [[ -d "${BUILD_DIR}" ]]; then
        log "Cleaning up build directory"
        rm -rf "${BUILD_DIR}"
    fi
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build vLLM CPU wheel from source with caching support.

OPTIONS:
    -v, --version VERSION     vLLM version to build (default: ${VLLM_VERSION})
    -c, --cache-dir DIR       Cache directory (default: ${CACHE_DIR})
    -f, --force-rebuild       Force rebuild even if cached wheel exists
    -h, --help               Show this help message

ENVIRONMENT VARIABLES:
    VLLM_VERSION             vLLM version to build
    CACHE_DIR                Cache directory for wheels
    FORCE_REBUILD            Set to 'true' to force rebuild
    BUILD_DIR                Temporary build directory

EXAMPLES:
    # Build with default settings
    $0
    
    # Build specific version
    $0 --version v0.6.3.post1
    
    # Force rebuild of cached version
    $0 --force-rebuild
    
    # Use custom cache directory
    $0 --cache-dir /tmp/vllm-cache

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VLLM_VERSION="$2"
            shift 2
            ;;
        -c|--cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -f|--force-rebuild)
            FORCE_REBUILD="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log "Starting vLLM CPU build process"
    log "Version: ${VLLM_VERSION}"
    log "Cache directory: ${CACHE_DIR}"
    
    # Trap for cleanup
    trap cleanup EXIT
    
    # Check system requirements
    check_requirements
    
    # Check for cached wheel first
    if cached_wheel=$(check_cached_wheel); then
        log "Using cached wheel: $(basename ${cached_wheel})"
        echo "${cached_wheel}"
        return 0
    fi
    
    # Install system dependencies
    install_system_deps
    
    # Setup ccache
    setup_ccache
    
    # Prepare source code
    prepare_source
    
    # Build wheel
    wheel_file=$(build_wheel)
    
    # Test the wheel
    test_wheel "${wheel_file}"
    
    # Cache the wheel
    cached_wheel=$(cache_wheel "${wheel_file}")
    
    log "vLLM CPU build completed successfully"
    log "Wheel available at: ${cached_wheel}"
    
    echo "${cached_wheel}"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
