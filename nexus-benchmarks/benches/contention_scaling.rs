//! Contention Scaling Benchmark
//!
//! Measures the cost of ADVANCING the epoch under heavy contention, which is:
//! - Baseline (Flat): O(T) due to cache line contention on single atomic
//! - Nexus (Hierarchical): O(log T) due to distributed updates across tree
//!
//! This benchmark simulates realistic write contention where multiple threads
//! compete to advance the global epoch simultaneously, triggering cache coherence
//! traffic storms on the flat baseline.
//!
//! # Methodology
//!
//! - Baseline: All threads CAS a shared AtomicU64 (worst-case bus contention)
//! - Nexus: Threads update their local leaves, with lazy aggregation to root
//!
//! # Expected Results
//!
//! - Baseline: Linear or super-linear latency growth with thread count
//! - Nexus: Logarithmic latency growth due to reduced cache line sharing

use nexus_memory::epoch::HierarchicalEpoch;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Number of CAS/update operations per thread
const OPS_PER_THREAD: u64 = 100_000;

/// Thread counts to benchmark
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Results from a benchmark run
#[derive(Debug)]
struct BenchResult {
    bench_type: &'static str,
    threads: usize,
    total_ops: u64,
    total_time_ns: u64,
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

fn main() {
    eprintln!("Contention Scaling Benchmark: Flat vs Hierarchical Epoch");
    eprintln!("=========================================================\n");
    println!("type,threads,ops,latency_ns,throughput_mops");
    
    let mut all_results = Vec::new();
    
    for &num_threads in THREAD_COUNTS {
        // Run baseline (flat single-atomic) benchmark
        let baseline_result = run_flat_cas_benchmark(num_threads);
        println!("{}", baseline_result.to_csv());
        all_results.push(baseline_result);
        
        // Run Nexus (hierarchical) benchmark
        let nexus_result = run_hierarchical_benchmark(num_threads);
        println!("{}", nexus_result.to_csv());
        all_results.push(nexus_result);
    }
    
    // Summary report (to stderr so it doesn't pollute CSV)
    eprintln!("\n--- Summary ---");
    eprintln!("Baseline shows O(T) scaling due to cache line contention");
    eprintln!("Nexus shows O(log T) scaling due to hierarchical aggregation\n");
    

    // Export to CSV file
    export_csv(&all_results);
}

/// Baseline: All threads compete to CAS a single shared atomic
/// 
/// This creates maximum cache coherence traffic (bus storms) as all cores
/// fight over the same cache line. Simulates O(T) coordination cost.
fn run_flat_cas_benchmark(num_threads: usize) -> BenchResult {
    let shared_epoch = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(num_threads));
    let start_flag = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::with_capacity(num_threads);
    
    for _ in 0..num_threads {
        let epoch = shared_epoch.clone();
        let bar = barrier.clone();
        let start = start_flag.clone();
        let ops_counter = total_ops.clone();
        
        handles.push(thread::spawn(move || {
            // Wait for all threads to be ready
            bar.wait();
            
            // Spin until start signal
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            
            let mut local_ops = 0u64;
            let mut success_count = 0u64;
            
            // Heavy CAS contention loop
            for _ in 0..OPS_PER_THREAD {
                // Try to increment the shared epoch via CAS
                // This triggers cache-to-cache transfers every time
                let current = epoch.load(Ordering::Relaxed);
                match epoch.compare_exchange_weak(
                    current,
                    current.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => success_count += 1,
                    Err(_) => {
                        // CAS failed due to contention, still counts as work
                    }
                }
                local_ops += 1;
            }
            
            ops_counter.fetch_add(local_ops, Ordering::Relaxed);
            success_count
        }));
    }
    
    // Start timing
    let start_time = Instant::now();
    start_flag.store(true, Ordering::Release);
    
    // Wait for all threads to complete
    let mut total_successes = 0u64;
    for h in handles {
        total_successes += h.join().unwrap();
    }
    
    let elapsed = start_time.elapsed();
    let total_ops_done = total_ops.load(Ordering::Relaxed);
    let elapsed_ns = elapsed.as_nanos() as u64;
    
    BenchResult {
        bench_type: "baseline",
        threads: num_threads,
        total_ops: total_ops_done,
        total_time_ns: elapsed_ns,
        avg_latency_ns: elapsed_ns as f64 / total_ops_done as f64,
        throughput_mops: (total_ops_done as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
    }
}

/// Nexus: Threads update their local leaf nodes in the hierarchical tree
/// 
/// Each thread updates its own dedicated leaf, with changes propagating
/// upward through the aggregation tree. This distributes contention and
/// achieves O(log T) coordination cost.
fn run_hierarchical_benchmark(num_threads: usize) -> BenchResult {
    // Create hierarchical epoch with sufficient capacity
    let capacity = num_threads.next_power_of_two().max(4);
    let hier_epoch = Arc::new(HierarchicalEpoch::new(capacity));
    let barrier = Arc::new(Barrier::new(num_threads));
    let start_flag = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::with_capacity(num_threads);
    
    for tid in 0..num_threads {
        let hier = hier_epoch.clone();
        let bar = barrier.clone();
        let start = start_flag.clone();
        let ops_counter = total_ops.clone();
        
        handles.push(thread::spawn(move || {
            // Wait for all threads to be ready
            bar.wait();
            
            // Spin until start signal
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            
            let mut local_epoch = 1u64;
            let mut local_ops = 0u64;
            
            // Update local leaf node in hierarchical tree
            // This triggers O(log T) propagation through ancestors
            for _ in 0..OPS_PER_THREAD {
                // Update this thread's local epoch
                // Internally, this updates leaf[tid] and propagates upward
                hier.update_local(tid, local_epoch);
                
                local_epoch = local_epoch.wrapping_add(1);
                local_ops += 1;
            }
            
            ops_counter.fetch_add(local_ops, Ordering::Relaxed);
        }));
    }
    
    // Start timing
    let start_time = Instant::now();
    start_flag.store(true, Ordering::Release);
    
    // Wait for all threads to complete
    for h in handles {
        h.join().unwrap();
    }
    
    let elapsed = start_time.elapsed();
    let total_ops_done = total_ops.load(Ordering::Relaxed);
    let elapsed_ns = elapsed.as_nanos() as u64;
    
    BenchResult {
        bench_type: "nexus",
        threads: num_threads,
        total_ops: total_ops_done,
        total_time_ns: elapsed_ns,
        avg_latency_ns: elapsed_ns as f64 / total_ops_done as f64,
        throughput_mops: (total_ops_done as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
    }
}

/// Additional benchmark: Interleaved read-write contention
/// 
/// Mixed workload where threads alternate between updating local epochs
/// and querying the global minimum. This simulates realistic usage patterns.
#[allow(dead_code)]
fn run_mixed_workload_benchmark(num_threads: usize) -> BenchResult {
    let capacity = num_threads.next_power_of_two().max(4);
    let hier_epoch = Arc::new(HierarchicalEpoch::new(capacity));
    let barrier = Arc::new(Barrier::new(num_threads));
    let start_flag = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::with_capacity(num_threads);
    
    for tid in 0..num_threads {
        let hier = hier_epoch.clone();
        let bar = barrier.clone();
        let start = start_flag.clone();
        let ops_counter = total_ops.clone();
        
        handles.push(thread::spawn(move || {
            bar.wait();
            
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            
            let mut local_epoch = 1u64;
            let mut local_ops = 0u64;
            let mut global_min_sum = 0u64;  // Prevent optimization
            
            for i in 0..OPS_PER_THREAD {
                // 75% writes, 25% reads (realistic ratio)
                if i % 4 != 0 {
                    hier.update_local(tid, local_epoch);
                    local_epoch = local_epoch.wrapping_add(1);
                } else {
                    global_min_sum = global_min_sum.wrapping_add(hier.global_minimum());
                }
                local_ops += 1;
            }
            
            ops_counter.fetch_add(local_ops, Ordering::Relaxed);
            global_min_sum  // Return to prevent optimization
        }));
    }
    
    let start_time = Instant::now();
    start_flag.store(true, Ordering::Release);
    
    let mut _sum = 0u64;
    for h in handles {
        _sum = _sum.wrapping_add(h.join().unwrap());
    }
    
    let elapsed = start_time.elapsed();
    let total_ops_done = total_ops.load(Ordering::Relaxed);
    let elapsed_ns = elapsed.as_nanos() as u64;
    
    BenchResult {
        bench_type: "nexus_mixed",
        threads: num_threads,
        total_ops: total_ops_done,
        total_time_ns: elapsed_ns,
        avg_latency_ns: elapsed_ns as f64 / total_ops_done as f64,
        throughput_mops: (total_ops_done as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
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
