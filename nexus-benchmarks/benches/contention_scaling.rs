//! Contention Scaling Benchmark - O(T) vs O(log T) Global Query Cost
//!
//! This benchmark demonstrates the key scalability difference:
//! - Baseline (Flat): Computing global minimum requires O(T) scan of all participants
//! - Nexus (Hierarchical): Global minimum is pre-aggregated, requires O(log T) tree traversal
//!
//! # Methodology
//!
//! The critical insight: The O(log T) vs O(T) difference appears when QUERYING global state,
//! not when updating local state. This benchmark:
//!
//! 1. Simulates T active participants (threads)
//! 2. Measures the cost of computing "safe reclamation epoch" (global minimum)
//! 3. Baseline must atomically read ALL T participant epochs and find minimum
//! 4. Nexus reads pre-aggregated tree nodes (only log(T) depth)
//!
//! # Expected Results
//!
//! - Baseline latency grows linearly with T (O(T) scan)
//! - Nexus latency grows logarithmically with T (O(log T) tree depth)
//! - Speedup = Baseline/Nexus should be > 1.0 at high thread counts

use nexus_memory::epoch::HierarchicalEpoch;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Number of global minimum queries per measurement
const QUERIES_PER_MEASUREMENT: usize = 100_000;

/// Thread counts to benchmark
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Maximum participants for flat array
const MAX_PARTICIPANTS: usize = 512;

/// Inactive epoch marker
const INACTIVE: u64 = u64::MAX;

/// Results from a benchmark run
#[derive(Debug)]
struct BenchResult {
    bench_type: &'static str,
    threads: usize,
    total_ops: u64,
    avg_latency_ns: f64,
    throughput_mops: f64,
}

impl BenchResult {
    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.3}",
            self.bench_type,
            self.threads,
            self.total_ops,
            self.avg_latency_ns,
            self.throughput_mops
        )
    }
}

/// Flat epoch collector that requires O(T) scan
struct FlatEpochCollector {
    epochs: Box<[AtomicU64; MAX_PARTICIPANTS]>,
    num_participants: usize,
}

impl FlatEpochCollector {
    fn new(num_participants: usize) -> Self {
        let epochs = Box::new(std::array::from_fn(|_| AtomicU64::new(INACTIVE)));
        Self {
            epochs,
            num_participants,
        }
    }

    /// Register a participant with an epoch value
    fn set_epoch(&self, id: usize, epoch: u64) {
        self.epochs[id].store(epoch, Ordering::Release);
    }

    /// Compute global minimum - O(T) operation
    ///
    /// This MUST scan ALL participants to find the minimum epoch.
    /// This is the critical performance bottleneck in flat schemes.
    #[inline(never)]
    fn global_minimum(&self) -> u64 {
        let mut min = INACTIVE;

        // O(T) scan - must check every participant
        for i in 0..self.num_participants {
            let epoch = self.epochs[i].load(Ordering::Acquire);
            if epoch != INACTIVE && epoch < min {
                min = epoch;
            }
        }

        min
    }
}

fn main() {
    eprintln!("O(T) vs O(log T) Global Minimum Query Benchmark");
    eprintln!("================================================\n");
    eprintln!("Measuring global_minimum() latency vs participant count\n");

    println!("type,threads,ops,latency_ns,throughput_mops");

    let mut all_results = Vec::new();

    for &num_participants in THREAD_COUNTS {
        eprintln!("Participants: {}", num_participants);

        // --- Baseline: Flat O(T) scan ---
        let baseline_result = run_flat_benchmark(num_participants);
        println!("{}", baseline_result.to_csv());
        eprintln!(
            "  Baseline (Flat O(T)):        {:.2} ns",
            baseline_result.avg_latency_ns
        );
        all_results.push(baseline_result);

        // --- Nexus: Hierarchical O(log T) ---
        let nexus_result = run_hierarchical_benchmark(num_participants);
        println!("{}", nexus_result.to_csv());
        eprintln!(
            "  Nexus (Hierarchical O(log T)): {:.2} ns",
            nexus_result.avg_latency_ns
        );

        // Calculate speedup
        let speedup = all_results.last().unwrap().avg_latency_ns / nexus_result.avg_latency_ns;
        eprintln!("  Speedup: {:.2}x\n", speedup);

        all_results.push(nexus_result);
    }

    // Summary
    eprintln!("\n--- Summary ---");
    eprintln!("Baseline (Flat): O(T) scan grows linearly with participant count");
    eprintln!("Nexus (Hierarchical): O(log T) tree read grows logarithmically");

    // Export to CSV file
    export_csv(&all_results);
}

/// Baseline: Flat epoch collector with O(T) global minimum query
fn run_flat_benchmark(num_participants: usize) -> BenchResult {
    let collector = FlatEpochCollector::new(num_participants);

    // Setup: Register all participants with varying epochs
    for i in 0..num_participants {
        // Assign epochs 1, 2, 3, ... so minimum is always 1
        collector.set_epoch(i, (i + 1) as u64);
    }

    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(collector.global_minimum());
    }

    // Measurement: Time global_minimum() queries
    let start = Instant::now();
    let mut sum = 0u64;

    for _ in 0..QUERIES_PER_MEASUREMENT {
        sum = sum.wrapping_add(collector.global_minimum());
    }

    let elapsed = start.elapsed();
    std::hint::black_box(sum); // Prevent optimization

    let total_ops = QUERIES_PER_MEASUREMENT as u64;
    let elapsed_ns = elapsed.as_nanos() as f64;

    BenchResult {
        bench_type: "baseline",
        threads: num_participants,
        total_ops,
        avg_latency_ns: elapsed_ns / total_ops as f64,
        throughput_mops: (total_ops as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
    }
}

/// Nexus: Hierarchical epoch with O(log T) global minimum query
fn run_hierarchical_benchmark(num_participants: usize) -> BenchResult {
    let capacity = num_participants.next_power_of_two().max(4);
    let hier_epoch = HierarchicalEpoch::new(capacity);

    // Setup: Register all participants with varying epochs
    for i in 0..num_participants {
        // Assign epochs 1, 2, 3, ... so minimum is always 1
        hier_epoch.update_local(i, (i + 1) as u64);
    }

    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(hier_epoch.global_minimum());
    }

    // Measurement: Time global_minimum() queries
    let start = Instant::now();
    let mut sum = 0u64;

    for _ in 0..QUERIES_PER_MEASUREMENT {
        sum = sum.wrapping_add(hier_epoch.global_minimum());
    }

    let elapsed = start.elapsed();
    std::hint::black_box(sum); // Prevent optimization

    let total_ops = QUERIES_PER_MEASUREMENT as u64;
    let elapsed_ns = elapsed.as_nanos() as f64;

    BenchResult {
        bench_type: "nexus",
        threads: num_participants,
        total_ops,
        avg_latency_ns: elapsed_ns / total_ops as f64,
        throughput_mops: (total_ops as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
    }
}

/// Export results to CSV file compatible with plot_scaling_theory.py
fn export_csv(results: &[BenchResult]) {
    use std::io::Write;

    let filename = "contention_scaling_results.csv";
    let mut file = match std::fs::File::create(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Warning: Could not create {}: {}", filename, e);
            return;
        }
    };

    // Header compatible with plot_scaling_theory.py
    writeln!(file, "type,threads,ops,latency_ns,throughput_mops").unwrap();

    for r in results {
        writeln!(file, "{}", r.to_csv()).unwrap();
    }

    eprintln!("\nResults exported to {}", filename);
}
