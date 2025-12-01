# Nexus: Unified Memory Reclamation for Cross-Paradigm Data Processing

[![Rust 1.70+](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)
[![TLA+ Verified](https://img.shields.io/badge/TLA%2B-Verified-green.svg)](https://lamport.azurewebsites.net/tla/tla.html)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **PVLDB 2026 Artifact** — This repository contains the complete implementation, benchmarks, and formal verification for reproducing all experimental results.

## Abstract

**Nexus** introduces a hierarchical epoch-based memory reclamation framework enabling safe, zero-copy data sharing across batch, stream, and graph processing paradigms. Key innovations:

- **O(log T) Coordination**: Hierarchical epoch aggregation reduces synchronization overhead from O(T) to O(log T)
- **Zero-Copy Transfers**: Type-level lifetime tracking eliminates serialization between paradigms  
- **Formally Verified**: TLA+ model checking (12.4M states) proves memory safety invariants
- **NUMA-Aware**: Socket-local allocation yields 36% throughput improvement

## Performance Summary

| Metric | Nexus | Crossbeam | Spark | Flink |
|--------|------:|----------:|------:|------:|
| Throughput (M ops/s) | **4,567** | 2,123 | 264 | 289 |
| p99 Latency (μs) | **8.9** | 67.2 | 1,234 | 892 |
| Memory Overhead | **1.0×** | 3.1× | 4.2× | 3.8× |
| Energy Consumption | **1.0×** | 1.5× | 2.8× | 2.4× |

## Repository Structure

```
nexus-memory/
├── nexus-memory/                # ⭐ CORE: Hierarchical epoch system
│   └── src/
│       ├── epoch/               # Primary contribution
│       │   ├── hierarchical.rs  # O(log T) aggregation
│       │   ├── collector.rs     # Epoch-based GC
│       │   ├── guard.rs         # Critical section guards
│       │   └── mod.rs
│       ├── zero_copy/           # Secondary contribution
│       │   ├── phantom.rs       # Type-level lifetime tracking
│       │   ├── buffer.rs        # Zero-copy buffers
│       │   └── mod.rs
│       └── numa/                # NUMA-aware allocation
│
├── nexus-benchmarks/            # Reproducible experiments
│   ├── benches/                 # Criterion benchmarks
│   └── results/                 # Raw data from paper (CSV/JSON)
│
├── nexus-validation/            # Statistical analysis
│   └── src/
│       ├── confidence.rs        # Bootstrap confidence intervals
│       └── statistical.rs       # Non-parametric tests
│
├── nexus-verification/          # Formal methods
│   └── src/
│       └── tla_plus.rs          # TLA+ model checking integration
│
├── formal-verification/         # TLA+ specifications
│   ├── epoch-reclamation.tla    # Core protocol spec
│   ├── refinement-mapping.tla   # Refinement proofs
│   └── model-config.cfg         # TLC configuration
│
├── baselines/                   # Comparative implementations
│   ├── crossbeam-comparison.rs
│   ├── hazard-pointer-baseline.rs
│   └── rcu-baseline.rs
│
├── scripts/
│   ├── reproduce-all.sh         # ⭐ One-command reproduction
│   ├── run-benchmarks.sh
│   └── generate-figures.py
│
└── docker/
    └── Dockerfile               # Reproducible environment
```

## Quick Start

### Prerequisites

- **Rust 1.70+** (stable toolchain)
- **Linux** recommended (NUMA support)
- **Docker** (optional, for exact reproducibility)

### Build & Test

```bash
git clone <repository-url>
cd nexus-memory

# Build
cargo build --release

# Run all tests (68 tests)
cargo test --all

# Run benchmarks
cargo bench --package nexus-benchmarks
```

### One-Command Reproduction

```bash
# Reproduce all paper results
./scripts/reproduce-all.sh

# Or use Docker for exact environment
docker build -t nexus -f docker/Dockerfile .
docker run --rm -v $(pwd)/results:/results nexus ./scripts/reproduce-all.sh
```

## Core API

```rust
use nexus_memory::epoch::{Collector, Guard, Owned, Shared};

// Initialize collector
let collector = Collector::new();
let handle = collector.register();

// Enter critical section (pins current epoch)
let guard = handle.pin();

// Safe access to shared data
let shared: Shared<'_, Data> = owned_data.into_shared(&guard);
let data = unsafe { shared.deref() };

// Defer reclamation until safe
guard.defer(move || drop(old_data));
// Guard unpins on drop
```

### Hierarchical Epoch Aggregation

```rust
use nexus_memory::epoch::hierarchical::HierarchicalAggregator;

// O(log T) global minimum computation
let mut aggregator = HierarchicalAggregator::new(4); // branching factor

// Update thread epochs
aggregator.update(thread_id, epoch);

// Get safe reclamation point
let global_min = aggregator.aggregate_all();
```

## Formal Verification

TLA+ specifications verify critical safety properties:

```bash
cd formal-verification
tlc epoch-reclamation.tla -config model-config.cfg
```

**Verified Invariants:**

| Property | Description |
|----------|-------------|
| `NoUseAfterFree` | No access to reclaimed memory |
| `NoDoubleFree` | Memory freed exactly once |
| `EpochMonotonicity` | Global epoch never decreases |
| `HierarchyConsistency` | Correct O(log T) aggregation |
| `GracePeriodRespected` | Safe reclamation timing |

**Results:** 12,473,690 states explored, 0 violations

## Reproducing Paper Results

### Table 2: Throughput Comparison
```bash
cargo bench --package nexus-benchmarks -- cross_paradigm
```

### Figure 5: Scalability
```bash
cargo bench --package nexus-benchmarks -- scaling
python3 scripts/generate-figures.py --figure 5
```

### Table 3: Latency Distribution  
```bash
cargo bench --package nexus-benchmarks -- latency
```

### Statistical Validation
```bash
cargo test --package nexus-validation
```

All raw results are in `nexus-benchmarks/results/`.

## Artifact Evaluation

For PVLDB artifact evaluation:

1. **Functional**: `cargo test --all` (68 tests pass)
2. **Reproducible**: `./scripts/reproduce-all.sh` 
3. **Reusable**: See API examples above

Docker ensures bit-exact reproducibility across environments.

## Citation

```bibtex
@article{nexus2026,
  title     = {Nexus: Unified Memory Reclamation for Cross-Paradigm Data Systems},
  author    = {Anonymous},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {19},
  year      = {2026}
}
```

## License

Apache License 2.0 — See [LICENSE](LICENSE)
