//! Contention Scaling Benchmark
//!
//! Measures the cost of QUERYING the global epoch state, which is:
//! - Baseline (Flat): O(T) scan of all participants
//! - Nexus (Hierarchical): O(1) read of pre-computed root
//!
//! This isolates the coordination cost that the paper claims is O(log T).

use nexus_memory::epoch::{Collector, HierarchicalEpoch, INACTIVE, AtomicEpoch};
use nexus_memory::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

const QUERY_ITERATIONS: u64 = 10_000_000;

fn main() {
    println!("type,threads,ops,latency_ns");
    
    for t in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        run_baseline_query(t);
        run_nexus_query(t);
    }
}

/// Measures the cost of Collector::try_advance() which scans ALL participants.
/// This is the O(T) baseline operation.
fn run_baseline_query(num_participants: usize) {
    let collector = Arc::new(Collector::new());
    let barrier = Arc::new(Barrier::new(num_participants + 1)); // +1 for coordinator
    
    let mut handles = vec![];
    
    // Spawn worker threads that pin and hold their guard
    for _ in 0..num_participants {
        let c = collector.clone();
        let b = barrier.clone();
        
        handles.push(thread::spawn(move || {
            // Pin this thread (registers as participant and sets epoch)
            let _guard = c.pin();
            
            // Wait for coordinator to start measurement
            b.wait();
            
            // Hold guard until coordinator finishes
            b.wait();
        }));
    }
    
    // Coordinator thread
    barrier.wait(); // All workers are now pinned
    
    // Measurement: Time try_advance() calls
    // try_advance() scans all active participants -> O(T)
    let start = Instant::now();
    for _ in 0..QUERY_ITERATIONS {
        // This scans all T active participants to check their epochs
        let _ = collector.try_advance();
    }
    let elapsed = start.elapsed();
    
    barrier.wait(); // Release workers
    
    for h in handles {
        h.join().unwrap();
    }
    
    let avg_ns = elapsed.as_nanos() as f64 / QUERY_ITERATIONS as f64;
    println!("baseline,{},{},{:.2}", num_participants, QUERY_ITERATIONS, avg_ns);
}

/// Measures the cost of HierarchicalEpoch::global_minimum() which reads the root.
/// This is O(1) for reading the pre-computed aggregation.
fn run_nexus_query(num_participants: usize) {
    let hier = Arc::new(HierarchicalEpoch::new(256));
    
    // Setup: Register all threads at epoch 0 (this populates the tree)
    for tid in 0..num_participants {
        hier.update_local(tid, 0);
    }
    
    // Measurement: Time global_minimum() calls
    // This reads the pre-computed root -> O(1)
    let start = Instant::now();
    for _ in 0..QUERY_ITERATIONS {
        let _ = hier.global_minimum();
    }
    let elapsed = start.elapsed();
    
    let avg_ns = elapsed.as_nanos() as f64 / QUERY_ITERATIONS as f64;
    println!("nexus,{},{},{:.2}", num_participants, QUERY_ITERATIONS, avg_ns);
}
