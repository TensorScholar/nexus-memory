#!/usr/bin/env bash
#
# NEXUS Environment Validation Script
# 
# This script checks the system environment for running NEXUS benchmarks
# and paper reproduction experiments. It validates:
# - CPU topology and NUMA configuration
# - RAPL energy monitoring access
# - Rust toolchain version
# - Required dependencies
#
# Usage: ./scripts/check_env.sh
#
# For paper reproduction, all checks should pass.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
WARNED=0
FAILED=0

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${BLUE}$1${NC}"
    echo "============================================================"
}

print_ok() {
    echo -e "[${GREEN}OK${NC}] $1"
    ((++PASSED))
}

print_warn() {
    echo -e "[${YELLOW}WARN${NC}] $1"
    ((++WARNED))
}

print_fail() {
    echo -e "[${RED}FAIL${NC}] $1"
    ((++FAILED))
}

print_info() {
    echo -e "[${BLUE}INFO${NC}] $1"
}

# =============================================================================
# System Information
# =============================================================================

print_header "System Information"

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Kernel: $(uname -sr)"

if [[ "$(uname)" == "Linux" ]]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "OS: $NAME $VERSION"
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "OS: macOS $(sw_vers -productVersion)"
fi

# =============================================================================
# CPU Topology
# =============================================================================

print_header "CPU Topology"

if [[ "$(uname)" == "Linux" ]]; then
    if command -v lscpu &> /dev/null; then
        echo ""
        echo "--- lscpu output ---"
        lscpu | grep -E "(Architecture|CPU\(s\)|Thread|Core|Socket|NUMA|Model name|CPU MHz|Cache)" || true
        echo "--------------------"
        print_ok "CPU information available via lscpu"
    else
        print_warn "lscpu not available"
    fi
    
    if command -v numactl &> /dev/null; then
        echo ""
        echo "--- NUMA topology ---"
        numactl -H 2>/dev/null || echo "NUMA info not available"
        echo "---------------------"
        print_ok "NUMA tools available"
    else
        print_warn "numactl not installed (install with: apt install numactl)"
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    echo ""
    echo "--- sysctl CPU info ---"
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU"
    echo "Physical cores: $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'unknown')"
    echo "Logical cores: $(sysctl -n hw.logicalcpu 2>/dev/null || echo 'unknown')"
    echo "-----------------------"
    print_ok "CPU information available"
    print_info "NUMA not applicable on macOS"
fi

# =============================================================================
# RAPL Energy Monitoring (Linux only)
# =============================================================================

print_header "RAPL Energy Monitoring"

if [[ "$(uname)" == "Linux" ]]; then
    RAPL_PATH="/sys/class/powercap/intel-rapl"
    
    echo ""
    echo "--- RAPL Access Check ---"
    
    if [ -d "$RAPL_PATH" ]; then
        print_ok "powercap/intel-rapl subsystem found"
        
        # Check for package energy
        ENERGY_FILE="$RAPL_PATH/intel-rapl:0/energy_uj"
        if [ -f "$ENERGY_FILE" ]; then
            if [ -r "$ENERGY_FILE" ]; then
                ENERGY=$(cat "$ENERGY_FILE" 2>/dev/null || echo "unreadable")
                print_ok "RAPL energy file readable: $ENERGY µJ"
                
                # Show permissions
                echo ""
                ls -la "$ENERGY_FILE"
            else
                print_fail "RAPL energy file exists but not readable"
                echo "Fix with: sudo chmod +r $ENERGY_FILE"
                echo "Or run benchmarks with: sudo -E cargo bench --features rapl"
            fi
        else
            print_fail "RAPL energy file not found at $ENERGY_FILE"
        fi
        
        # List all RAPL domains
        echo ""
        echo "Available RAPL domains:"
        ls -la "$RAPL_PATH/" 2>/dev/null || true
        
    else
        print_fail "powercap/intel-rapl not found"
        echo "Possible causes:"
        echo "  - Non-Intel CPU"
        echo "  - Kernel missing CONFIG_POWERCAP and CONFIG_INTEL_RAPL"
        echo "  - Running in a VM without RAPL passthrough"
    fi
    
    echo "-------------------------"
    
elif [[ "$(uname)" == "Darwin" ]]; then
    print_info "RAPL not available on macOS"
    print_info "Energy benchmarks will use simulated monitoring"
    print_warn "For accurate energy measurements, run on Linux with Intel CPU"
fi

# =============================================================================
# Rust Toolchain
# =============================================================================

print_header "Rust Toolchain"

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    print_ok "Rust compiler: $RUST_VERSION"
    
    # Check minimum version (1.75 required)
    RUST_MINOR=$(rustc --version | sed 's/rustc 1\.\([0-9]*\).*/\1/')
    if [ "$RUST_MINOR" -ge 75 ]; then
        print_ok "Rust version meets minimum requirement (1.75+)"
    else
        print_fail "Rust version too old, need 1.75+ (have 1.$RUST_MINOR)"
        echo "Update with: rustup update stable"
    fi
else
    print_fail "Rust not installed"
    echo "Install from: https://rustup.rs/"
fi

if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version)
    print_ok "Cargo: $CARGO_VERSION"
else
    print_fail "Cargo not installed"
fi

# Check for nightly (optional, for some features)
if rustup show 2>/dev/null | grep -q nightly; then
    print_ok "Rust nightly toolchain available"
else
    print_info "Rust nightly not installed (optional)"
fi

# =============================================================================
# Required Tools
# =============================================================================

print_header "Required Tools"

# Git
if command -v git &> /dev/null; then
    print_ok "git: $(git --version)"
else
    print_fail "git not installed"
fi

# Python (for data generation)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_ok "Python: $PYTHON_VERSION"
    
    # Check for required Python packages
    if python3 -c "import pandas" 2>/dev/null; then
        print_ok "pandas available"
    else
        print_warn "pandas not installed (needed for data generation)"
        echo "Install with: pip install pandas"
    fi
    
    if python3 -c "import numpy" 2>/dev/null; then
        print_ok "numpy available"
    else
        print_warn "numpy not installed (needed for data generation)"
        echo "Install with: pip install numpy"
    fi
else
    print_warn "Python 3 not installed (needed for data generation)"
fi

# =============================================================================
# Memory and System Resources
# =============================================================================

print_header "System Resources"

if [[ "$(uname)" == "Linux" ]]; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
    echo "Total Memory: $TOTAL_MEM"
    echo "Available Memory: $AVAIL_MEM"
    
    # Check for huge pages (optional optimization)
    if [ -f /proc/sys/vm/nr_hugepages ]; then
        HUGE_PAGES=$(cat /proc/sys/vm/nr_hugepages)
        if [ "$HUGE_PAGES" -gt 0 ]; then
            print_ok "Huge pages enabled: $HUGE_PAGES"
        else
            print_info "Huge pages not configured (optional optimization)"
        fi
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')
    echo "Total Memory: $TOTAL_MEM"
fi

# =============================================================================
# Summary
# =============================================================================

print_header "Environment Check Summary"

echo ""
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Warnings: ${YELLOW}$WARNED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    if [ "$WARNED" -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Environment is ready for paper reproduction.${NC}"
    else
        echo -e "${YELLOW}⚠ Environment ready with warnings. Some optional features may be unavailable.${NC}"
    fi
    exit 0
else
    echo -e "${RED}✗ Some required checks failed. Please fix the issues above.${NC}"
    exit 1
fi
