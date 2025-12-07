//! Contention Scaling Benchmark
//!
//! This benchmark isolates and measures the synchronization overhead of the
//! epoch-based reclamation primitives to empirically validate the O(log T)
//! scaling claim from the paper.
//!
//! # Methodology
//!
//! 1. Measure pure latency of `pin()` and `try_advance()` under 100% contention
//! 2. Vary thread counts from 1 to 128
//! 3. Export detailed latency data for regression analysis
//!
//! # Output
//!
//! Generates `results/internal_contention.csv` with columns:
//! - threads: Number of concurrent threads
//! - pin_latency_ns: Average pin() latency in nanoseconds
//! - advance_latency_ns: Average try_advance() latency in nanoseconds
//! - advance_success_rate: Fraction of successful epoch advances
//!
//! # Running
//!
//! ```bash
//! cargo bench --package nexus-benchmarks -- contention_scaling
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexus_memory::epoch::{Collector, InstrumentedCollector};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

/// Thread counts to benchmark
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

/// Number of iterations per thread
const ITERATIONS_PER_THREAD: u64 = 10_000;

/// Warm-up iterations
const WARMUP_ITERATIONS: u64 = 1_000;

/// Results from a contention scaling test.
#[derive(Debug, Clone)]
pub struct ContentionResult {
    pub threads: usize,
    pub total_operations: u64,
    pub total_time_ns: u64,
    pub avg_pin_latency_ns: f64,
    pub avg_advance_latency_ns: f64,
    pub advance_success_rate: f64,
    pub p50_latency_ns: u64,
    pub p99_latency_ns: u64,
}

/// Runs a contention test with the specified number of threads.
///
/// All threads do nothing but enter and exit epochs (100% contention),
/// measuring the pure synchronization overhead.
fn run_contention_test(thread_count: usize) -> ContentionResult {
    let collector = Arc::new(InstrumentedCollector::new());
    let running = Arc::new(AtomicBool::new(true));
    let completed = Arc::new(AtomicU64::new(0));
    
    // Collect individual latency samples for percentile calculation
    let latency_samples: Arc<std::sync::Mutex<Vec<u64>>> = 
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(
            (thread_count * ITERATIONS_PER_THREAD as usize).min(100_000)
        )));
    
    // Spawn worker threads
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let collector = Arc::clone(&collector);
            let running = Arc::clone(&running);
            let completed = Arc::clone(&completed);
            let samples = Arc::clone(&latency_samples);
            
            thread::spawn(move || {
                let mut local_samples = Vec::with_capacity(ITERATIONS_PER_THREAD as usize);
                
                // Warm-up phase
                for _ in 0..WARMUP_ITERATIONS {
                    let (guard, _) = collector.pin_timed();
                    black_box(&guard);
                    drop(guard);
                    let _ = collector.try_advance_timed();
                }
                
                // Measurement phase
                for _ in 0..ITERATIONS_PER_THREAD {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }
                    
                    let (guard, pin_latency) = collector.pin_timed();
                    black_box(&guard);
                    drop(guard);
                    
                    let (_, advance_latency) = collector.try_advance_timed();
                    
                    local_samples.push(pin_latency + advance_latency);
                    completed.fetch_add(1, Ordering::Relaxed);
                }
                
                // Contribute samples for percentile calculation
                if let Ok(mut global_samples) = samples.lock() {
                    // Only add a subset to avoid memory explosion
                    let step = local_samples.len().max(1) / 100;
                    for (i, &sample) in local_samples.iter().enumerate() {
                        if step == 0 || i % step == 0 {
                            global_samples.push(sample);
                        }
                    }
                }
            })
        })
        .collect();
    
    let start = Instant::now();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_time = start.elapsed();
    
    running.store(false, Ordering::SeqCst);
    
    // Get metrics
    let metrics = collector.get_metrics();
    let total_ops = completed.load(Ordering::Relaxed);
    
    // Calculate percentiles
    let (p50, p99) = {
        let mut samples = latency_samples.lock().unwrap();
        samples.sort_unstable();
        
        let p50_idx = samples.len() * 50 / 100;
        let p99_idx = samples.len() * 99 / 100;
        
        (
            samples.get(p50_idx).copied().unwrap_or(0),
            samples.get(p99_idx).copied().unwrap_or(0),
        )
    };
    
    ContentionResult {
        threads: thread_count,
        total_operations: total_ops,
        total_time_ns: total_time.as_nanos() as u64,
        avg_pin_latency_ns: metrics.avg_pin_latency_ns(),
        avg_advance_latency_ns: metrics.avg_advance_latency_ns(),
        advance_success_rate: metrics.advance_success_rate(),
        p50_latency_ns: p50,
        p99_latency_ns: p99,
    }
}

/// Exports results to CSV for analysis.
fn export_results_csv(results: &[ContentionResult], path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    // Write header
    writeln!(file, "threads,total_ops,total_time_ns,avg_pin_latency_ns,avg_advance_latency_ns,advance_success_rate,p50_latency_ns,p99_latency_ns")?;
    
    // Write data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{:.2},{:.2},{:.4},{},{}",
            r.threads,
            r.total_operations,
            r.total_time_ns,
            r.avg_pin_latency_ns,
            r.avg_advance_latency_ns,
            r.advance_success_rate,
            r.p50_latency_ns,
            r.p99_latency_ns,
        )?;
    }
    
    Ok(())
}

/// Benchmark: Measure pin() latency scaling with thread count.
fn bench_pin_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention_scaling/pin");
    group.measurement_time(Duration::from_secs(5));
    
    for &threads in THREAD_COUNTS {
        if threads > num_cpus() {
            continue; // Skip if more threads than CPUs
        }
        
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &thread_count| {
                let collector = Arc::new(Collector::new());
                
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let c = Arc::clone(&collector);
                            thread::spawn(move || {
                                for _ in 0..100 {
                                    let guard = c.pin();
                                    black_box(&guard);
                                    drop(guard);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                    
                    black_box(collector.epoch())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Measure try_advance() latency scaling with thread count.
fn bench_advance_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention_scaling/advance");
    group.measurement_time(Duration::from_secs(5));
    
    for &threads in THREAD_COUNTS {
        if threads > num_cpus() {
            continue;
        }
        
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &thread_count| {
                let collector = Arc::new(Collector::new());
                
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let c = Arc::clone(&collector);
                            thread::spawn(move || {
                                for _ in 0..100 {
                                    let _guard = c.pin();
                                    c.try_advance();
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                    
                    black_box(collector.epoch())
                });
            },
        );
    }
    
    group.finish();
}

/// Run the full contention scaling experiment and export results.
fn bench_contention_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention_scaling/full");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let mut all_results = Vec::new();
    
    for &threads in THREAD_COUNTS {
        if threads > num_cpus() {
            println!("Skipping {} threads (exceeds CPU count {})", threads, num_cpus());
            continue;
        }
        
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &thread_count| {
                b.iter(|| {
                    let result = run_contention_test(thread_count);
                    all_results.push(result.clone());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
    
    // Export results to CSV
    if !all_results.is_empty() {
        let results_dir = std::env::current_dir()
            .unwrap()
            .join("results");
        std::fs::create_dir_all(&results_dir).ok();
        
        let csv_path = results_dir.join("internal_contention.csv");
        if let Err(e) = export_results_csv(&all_results, csv_path.to_str().unwrap()) {
            eprintln!("Warning: Failed to export CSV: {}", e);
        } else {
            println!("Results exported to: {}", csv_path.display());
        }
    }
}

/// Returns the number of available CPUs.
fn num_cpus() -> usize {
    thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Prints a summary of the contention scaling results.
pub fn print_summary(results: &[ContentionResult]) {
    println!("\n===== Contention Scaling Summary =====\n");
    println!("{:>8} {:>15} {:>18} {:>12}", 
             "Threads", "Avg Pin (ns)", "Avg Advance (ns)", "Success Rate");
    println!("{}", "-".repeat(60));
    
    for r in results {
        println!("{:>8} {:>15.2} {:>18.2} {:>12.2}%",
                 r.threads,
                 r.avg_pin_latency_ns,
                 r.avg_advance_latency_ns,
                 r.advance_success_rate * 100.0);
    }
    
    println!("\n========================================\n");
}

criterion_group!(
    benches,
    bench_pin_contention,
    bench_advance_contention,
    bench_contention_scaling,
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_contention_single_thread() {
        let result = run_contention_test(1);
        assert!(result.total_operations > 0);
        assert!(result.avg_pin_latency_ns > 0.0);
    }
    
    #[test]
    fn test_contention_multi_thread() {
        let result = run_contention_test(4);
        assert!(result.total_operations > 0);
    }
}
