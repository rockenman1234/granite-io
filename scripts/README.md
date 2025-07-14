# CPU vLLM Support for Granite IO

This directory contains scripts and configurations for building and testing vLLM with CPU-only inference support. This enables running the full Granite IO pipeline including LoRA adapters and constrained decoding on systems without GPUs.

## Overview

The CPU vLLM implementation supports:

- ✅ **CPU inference** - Run models entirely on CPU using optimized kernels
- ✅ **LoRA adapters** - Load and use LoRA fine-tuned models  
- ✅ **Constrained decoding** - JSON schema and regex-guided generation
- ✅ **Multi-threading** - Efficient CPU utilization with OpenMP
- ✅ **Caching** - Build cache to avoid recompilation

## Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: x86_64 with AVX512F instruction set
- **Python**: 3.9 - 3.12
- **Compiler**: GCC 12.3.0+ (recommended)
- **Memory**: 8GB+ RAM (for building and running models)

Check if your system is compatible:

```bash
# Check CPU features
lscpu | grep -i avx512

# Quick compatibility check
./scripts/setup-cpu-dev.sh check
```

### Local Development Setup

1. **Setup development environment**:

   ```bash
   ./scripts/setup-cpu-dev.sh setup
   ```

2. **Build vLLM CPU wheel**:

   ```bash
   ./scripts/setup-cpu-dev.sh build
   ```

3. **Test the installation**:

   ```bash
   ./scripts/setup-cpu-dev.sh test
   ```

4. **Check status**:

   ```bash
   ./scripts/setup-cpu-dev.sh status
   ```

### Tox Testing

Run CPU-specific tests using tox:

```bash
# Unit tests with CPU vLLM
tox -e vllm-cpu-unit

# Notebook tests (RAG example)
tox -e vllm-cpu-notebooks

# All CPU tests
tox -e vllm-cpu-unit,vllm-cpu-notebooks
```

## Files and Scripts

### Core Scripts

- **`build-vllm-cpu.sh`** - Main build script for vLLM CPU wheels
  - Handles source compilation with optimizations
  - Implements build caching for faster rebuilds
  - Tests wheel functionality after build
  
- **`setup-cpu-dev.sh`** - Development environment helper
  - Checks system prerequisites
  - Sets up Python virtual environment
  - Provides testing and status commands

- **`vllm-cpu-config.env`** - Configuration file
  - Pins vLLM version for compatibility
  - Sets CPU optimization flags
  - Configures runtime parameters

### Configuration

The configuration in `vllm-cpu-config.env` specifies:

```bash
# vLLM version with confirmed compatibility
VLLM_VERSION=v0.6.4.post1

# CPU optimization flags (auto-detected)
VLLM_CPU_AVX512BF16=auto
VLLM_CPU_AVX512VNNI=auto

# Runtime parameters
VLLM_CPU_KVCACHE_SPACE=8
VLLM_CPU_OMP_THREADS_BIND=auto
```

### Tox Environments

Two new tox environments are provided:

- **`vllm-cpu-unit`** - Runs unit tests with CPU vLLM
- **`vllm-cpu-notebooks`** - Tests notebooks (especially RAG demo)

## Architecture

### Build Process

1. **System Check** - Verify CPU features and dependencies
2. **Source Download** - Clone specific vLLM version
3. **Dependency Installation** - Install CPU-specific build requirements  
4. **Compilation** - Build with CPU optimizations enabled
5. **Testing** - Verify LoRA and constrained decoding work
6. **Caching** - Store wheel for future use

### Runtime Configuration

CPU vLLM uses environment variables for optimization:

- `VLLM_CPU_KVCACHE_SPACE` - KV cache memory allocation
- `VLLM_CPU_OMP_THREADS_BIND` - OpenMP thread binding strategy
- `VLLM_CPU_NUM_OF_RESERVED_CPU` - Cores reserved for system

## Compatibility Testing

The RAG notebook (`notebooks/rag.ipynb`) serves as the primary compatibility test as it exercises all three required features:

1. **CPU Inference** - Runs models on CPU backend
2. **LoRA Adapters** - Uses multiple LoRA models for different tasks
3. **Constrained Decoding** - Uses `guided_regex` and `guided_json`

## CI Integration

### GitHub Actions

The `.github/workflows/test_cpu_vllm.yml` workflow:

- Runs on Ubuntu with AVX512 support
- Caches compiled wheels between runs
- Tests both unit tests and notebook functionality
- Provides detailed error reporting

### Triggering

The workflow runs on:
- Manual dispatch with version selection
- Changes to CPU vLLM scripts
- Weekly scheduled runs to catch regressions

## Performance Considerations

### Build Time

- Initial build: 20-60 minutes (depending on CPU)
- Cached builds: < 5 minutes
- Use ccache to speed up rebuilds

### Runtime Performance

CPU inference is slower than GPU but suitable for:
- Development and testing
- CI environments
- Small to medium models
- Systems without GPU access

### Memory Usage

- Build process: ~8GB peak memory
- Runtime: Depends on model size and KV cache setting
- Recommended: 16GB+ for comfortable operation

## Troubleshooting

### Common Issues

1. **AVX512 Not Supported**
   ```
   Error: AVX512F CPU instruction set is required
   ```
   Solution: Use a CPU with AVX512F support or cloud instance

2. **GCC Version Too Old**
   ```
   Warning: gcc version 11. gcc>=12.3.0 is recommended
   ```
   Solution: Install GCC 12+
   ```bash
   sudo apt-get install gcc-12 g++-12
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10
   ```

3. **Build Memory Issues**
   ```
   Error: g++: internal compiler error: Killed
   ```
   Solution: Increase system memory or swap, reduce parallel build jobs

4. **Import Errors**
   ```
   ImportError: cannot import name 'guided_regex'
   ```
   Solution: Check vLLM version compatibility, rebuild with correct version

### Debug Information

Get detailed status:
```bash
./scripts/setup-cpu-dev.sh status
```

Check build logs:
```bash
# Build with verbose output
VERBOSE=1 ./scripts/build-vllm-cpu.sh

# Check cached build info
cat ~/.cache/granite-io/vllm-cpu/build_info.json
```

### Getting Help

For CPU vLLM specific issues:
1. Check system compatibility with `setup-cpu-dev.sh check`
2. Review build logs for compilation errors
3. Test with minimal configuration first
4. Check vLLM upstream documentation for CPU backend

## Version Compatibility

### Tested Versions

- **v0.6.4.post1** ✅ - Recommended, all features working
- **v0.6.3.post1** ✅ - Good compatibility
- **v0.6.2** ⚠️ - Basic compatibility, some limitations

### Version Selection

The default version is pinned in `vllm-cpu-config.env` based on:
- CPU inference stability
- LoRA adapter support
- Constrained decoding availability
- Build reliability

To test a different version:
```bash
VLLM_VERSION=v0.6.3.post1 ./scripts/build-vllm-cpu.sh
```

## Contributing

When modifying CPU vLLM support:

1. Test with the RAG notebook
2. Verify all three features work (CPU, LoRA, constrained)
3. Update version pins if needed
4. Run full test suite: `tox -e vllm-cpu-unit,vllm-cpu-notebooks`
5. Test on clean system/container

## Future Improvements

Potential enhancements:
- ARM64 support
- Docker-based builds
- Pre-built wheel distribution
- Performance benchmarking
- Integration with more notebooks
