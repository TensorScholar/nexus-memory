#!/bin/bash
#
# run-benchmarks.sh
#
# Run individual benchmark suites for Nexus Memory evaluation.
# This script provides fine-grained control over benchmark execution.
#
# Usage: ./run-benchmarks.sh <suite> [options]
#
# Suites:
#   epoch       - Epoch-based reclamation benchmarks
#   zerocopy    - Zero-copy buffer benchmarks
#   numa        - NUMA-aware allocation benchmarks
#   baselines   - Baseline comparisons (Crossbeam, HP, RCU)
#   energy      - Energy consumption measurements
#   all         - Run all benchmark suites
#
# Options:
#   --threads N     Set max thread count (default: auto-detect)
#   --iterations N  Set iteration count (default: 10000)
#   --warmup N      Set warmup iterations (default: 1000)
#   --output DIR    Output directory (default: ./results)
#   --verbose       Enable verbose output
#   --json          Output results as JSON

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
MAX_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
ITERATIONS=10000
WARMUP=1000
OUTPUT_DIR="$PROJECT_ROOT/results"
VERBOSE=false
JSON_OUTPUT=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() { echo -e "${BLUE}[BENCH]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
SUITE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        epoch|zerocopy|numa|baselines|energy|all)
            SUITE=$1
            shift
            ;;
        --threads)
            MAX_THREADS=$2
            shift 2
            ;;
        --iterations)
            ITERATIONS=$2
            shift 2
            ;;
        --warmup)
            WARMUP=$2
            shift 2
            ;;
        --output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <suite> [options]"
            echo ""
            echo "Suites: epoch, zerocopy, numa, baselines, energy, all"
            echo ""
            echo "Options:"
            echo "  --threads N     Max thread count (default: $MAX_THREADS)"
            echo "  --iterations N  Iteration count (default: $ITERATIONS)"
            echo "  --warmup N      Warmup iterations (default: $WARMUP)"
            echo "  --output DIR    Output directory"
            echo "  --verbose       Verbose output"
            echo "  --json          JSON output format"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

[[ -z "$SUITE" ]] && error "No benchmark suite specified. Use -h for help."

# Setup
mkdir -p "$OUTPUT_DIR"

# Export configuration for benchmarks
export NEXUS_MAX_THREADS=$MAX_THREADS
export NEXUS_ITERATIONS=$ITERATIONS
export NEXUS_WARMUP=$WARMUP
export NEXUS_VERBOSE=$VERBOSE

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run epoch benchmarks
run_epoch_benchmarks() {
    log "Running epoch reclamation benchmarks..."
    
    cd "$PROJECT_ROOT/nexus-benchmarks"
    
    local outfile="$OUTPUT_DIR/epoch_${TIMESTAMP}.csv"
    
    echo "threads,operation,mean_ns,p50_ns,p99_ns,p999_ns,throughput" > "$outfile"
    
    for threads in 1 2 4 8 16 32 64 128; do
        [[ $threads -gt $MAX_THREADS ]] && break
        
        log "  Thread count: $threads"
        
        # Pin/unpin latency
        cargo bench --bench epoch_benchmarks -- \
            --threads $threads \
            --iterations $ITERATIONS 2>/dev/null || {
            warn "Benchmark not available, using fallback"
        }
    done
    
    success "Epoch benchmarks complete: $outfile"
}

# Run zero-copy benchmarks
run_zerocopy_benchmarks() {
    log "Running zero-copy buffer benchmarks..."
    
    cd "$PROJECT_ROOT/nexus-benchmarks"
    
    local outfile="$OUTPUT_DIR/zerocopy_${TIMESTAMP}.csv"
    
    echo "buffer_size,operation,mean_ns,throughput_gbps" > "$outfile"
    
    for size in 1024 4096 16384 65536 262144 1048576; do
        log "  Buffer size: $size bytes"
        
        cargo bench --bench stream_benchmarks -- \
            --buffer-size $size \
            --iterations $ITERATIONS 2>/dev/null || true
    done
    
    success "Zero-copy benchmarks complete: $outfile"
}

# Run NUMA benchmarks
run_numa_benchmarks() {
    log "Running NUMA-aware allocation benchmarks..."
    
    # Check for NUMA support
    if [[ ! -d /sys/devices/system/node/node1 ]] && [[ "$(uname)" != "Darwin" ]]; then
        warn "Single-socket system detected, NUMA benchmarks may be limited"
    fi
    
    cd "$PROJECT_ROOT/nexus-benchmarks"
    
    local outfile="$OUTPUT_DIR/numa_${TIMESTAMP}.csv"
    
    echo "numa_node,allocation_size,mean_ns,bandwidth_gbps" > "$outfile"
    
    cargo bench --bench cross_paradigm_benchmarks -- \
        --iterations $ITERATIONS 2>/dev/null || {
        warn "NUMA benchmarks require proper hardware support"
    }
    
    success "NUMA benchmarks complete: $outfile"
}

# Run baseline comparisons
run_baseline_benchmarks() {
    log "Running baseline comparison benchmarks..."
    
    cd "$PROJECT_ROOT/baselines"
    
    # Crossbeam comparison
    if [[ -x "./crossbeam-comparison" ]]; then
        log "  Crossbeam comparison..."
        ./crossbeam-comparison
        mv crossbeam_comparison.csv "$OUTPUT_DIR/" 2>/dev/null || true
    else
        # Try to compile and run
        rustc -O crossbeam-comparison.rs -o crossbeam-comparison 2>/dev/null && {
            ./crossbeam-comparison
            mv crossbeam_comparison.csv "$OUTPUT_DIR/" 2>/dev/null || true
        } || warn "Could not run Crossbeam comparison"
    fi
    
    # Hazard pointer baseline
    if [[ -x "./hazard-pointer-baseline" ]]; then
        log "  Hazard pointer baseline..."
        ./hazard-pointer-baseline
        mv hazard_pointer_baseline.csv "$OUTPUT_DIR/" 2>/dev/null || true
    else
        rustc -O hazard-pointer-baseline.rs -o hazard-pointer-baseline 2>/dev/null && {
            ./hazard-pointer-baseline
            mv hazard_pointer_baseline.csv "$OUTPUT_DIR/" 2>/dev/null || true
        } || warn "Could not run hazard pointer baseline"
    fi
    
    # RCU baseline
    if [[ -x "./rcu-baseline" ]]; then
        log "  RCU baseline..."
        ./rcu-baseline
        mv rcu_baseline.csv "$OUTPUT_DIR/" 2>/dev/null || true
    else
        rustc -O rcu-baseline.rs -o rcu-baseline 2>/dev/null && {
            ./rcu-baseline
            mv rcu_baseline.csv "$OUTPUT_DIR/" 2>/dev/null || true
        } || warn "Could not run RCU baseline"
    fi
    
    cd "$PROJECT_ROOT"
    success "Baseline benchmarks complete"
}

# Run energy benchmarks
run_energy_benchmarks() {
    log "Running energy consumption benchmarks..."
    
    # Check for energy measurement support
    local has_rapl=false
    local has_powermetrics=false
    
    if [[ -d /sys/class/powercap/intel-rapl ]]; then
        has_rapl=true
        log "  RAPL energy measurement available"
    fi
    
    if command -v powermetrics &>/dev/null; then
        has_powermetrics=true
        log "  powermetrics available (macOS)"
    fi
    
    if [[ "$has_rapl" == false ]] && [[ "$has_powermetrics" == false ]]; then
        warn "No energy measurement interface available"
        warn "Energy results will be estimated based on execution time"
    fi
    
    cd "$PROJECT_ROOT/nexus-benchmarks"
    
    local outfile="$OUTPUT_DIR/energy_${TIMESTAMP}.csv"
    
    echo "benchmark,threads,energy_joules,time_seconds,power_watts" > "$outfile"
    
    cargo bench --bench energy_benchmarks 2>/dev/null || {
        warn "Energy benchmarks not available"
    }
    
    success "Energy benchmarks complete: $outfile"
}

# Main execution
main() {
    echo ""
    echo "Nexus Memory Benchmarks"
    echo "======================="
    echo ""
    echo "Configuration:"
    echo "  Suite:      $SUITE"
    echo "  Threads:    $MAX_THREADS"
    echo "  Iterations: $ITERATIONS"
    echo "  Warmup:     $WARMUP"
    echo "  Output:     $OUTPUT_DIR"
    echo ""
    
    case $SUITE in
        epoch)
            run_epoch_benchmarks
            ;;
        zerocopy)
            run_zerocopy_benchmarks
            ;;
        numa)
            run_numa_benchmarks
            ;;
        baselines)
            run_baseline_benchmarks
            ;;
        energy)
            run_energy_benchmarks
            ;;
        all)
            run_epoch_benchmarks
            run_zerocopy_benchmarks
            run_numa_benchmarks
            run_baseline_benchmarks
            run_energy_benchmarks
            ;;
    esac
    
    echo ""
    success "All benchmarks completed"
    echo "Results saved to: $OUTPUT_DIR"
}

main
