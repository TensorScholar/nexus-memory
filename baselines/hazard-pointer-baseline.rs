//! Hazard Pointer Baseline Implementation
//!
//! This module provides a hazard pointer-based memory reclamation baseline
//! for comparison against Nexus Memory's epoch-based approach.
//!
//! # Hazard Pointer Overview
//!
//! Hazard pointers provide bounded memory reclamation by:
//! 1. Each thread maintains a set of hazard pointers (HPs)
//! 2. Before accessing a node, a thread publishes its address to an HP
//! 3. Before freeing a node, check all HPs to ensure no thread holds it
//! 4. Complexity: O(H × T) per reclamation where H = hazards, T = threads
//!
//! # Expected Results
//!
//! Nexus epoch-based reclamation should show:
//! - Lower overhead for short-lived critical sections
//! - Better cache behavior (no per-access HP updates)
//! - Amortized reclamation costs
//!
//! Hazard pointers should show:
//! - Bounded memory usage (at most O(H × T²) unreclaimed)
//! - Better worst-case latency guarantees

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Maximum number of threads supported
const MAX_THREADS: usize = 128;
/// Hazard pointers per thread
const HAZARDS_PER_THREAD: usize = 2;
/// Total hazard pointer slots
const TOTAL_HAZARDS: usize = MAX_THREADS * HAZARDS_PER_THREAD;
/// Threshold for triggering reclamation
const RECLAIM_THRESHOLD: usize = 64;

/// A node that can be protected by hazard pointers
#[repr(C)]
pub struct HazardNode<T> {
    pub data: T,
    next: AtomicPtr<HazardNode<T>>,
}

impl<T> HazardNode<T> {
    pub fn new(data: T) -> *mut Self {
        Box::into_raw(Box::new(Self {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }))
    }
}

/// Per-thread retired list
struct RetiredList<T> {
    nodes: Vec<*mut HazardNode<T>>,
}

impl<T> RetiredList<T> {
    fn new() -> Self {
        Self { nodes: Vec::with_capacity(RECLAIM_THRESHOLD * 2) }
    }

    fn push(&mut self, node: *mut HazardNode<T>) {
        self.nodes.push(node);
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// Hazard pointer domain
pub struct HazardDomain<T> {
    /// Global hazard pointer array
    hazards: Box<[AtomicPtr<HazardNode<T>>; TOTAL_HAZARDS]>,
    /// Per-thread retired lists
    retired: Box<[UnsafeCell<RetiredList<T>>; MAX_THREADS]>,
    /// Number of registered threads
    thread_count: AtomicUsize,
    /// Total reclamation operations
    reclaim_count: AtomicU64,
    /// Total nodes reclaimed
    nodes_reclaimed: AtomicU64,
}

unsafe impl<T: Send> Send for HazardDomain<T> {}
unsafe impl<T: Send> Sync for HazardDomain<T> {}

impl<T> HazardDomain<T> {
    pub fn new() -> Self {
        Self {
            hazards: Box::new([(); TOTAL_HAZARDS].map(|_| AtomicPtr::new(ptr::null_mut()))),
            retired: Box::new([(); MAX_THREADS].map(|_| UnsafeCell::new(RetiredList::new()))),
            thread_count: AtomicUsize::new(0),
            reclaim_count: AtomicU64::new(0),
            nodes_reclaimed: AtomicU64::new(0),
        }
    }

    /// Register a new thread, returns thread ID
    pub fn register(&self) -> ThreadHandle {
        let id = self.thread_count.fetch_add(1, Ordering::Relaxed);
        assert!(id < MAX_THREADS, "Maximum thread count exceeded");
        ThreadHandle { id, domain: self }
    }

    /// Scan all hazard pointers - O(H × T) operation
    fn collect_hazards(&self) -> Vec<*mut HazardNode<T>> {
        let count = self.thread_count.load(Ordering::Relaxed);
        let mut hazards = Vec::with_capacity(count * HAZARDS_PER_THREAD);
        
        for i in 0..count * HAZARDS_PER_THREAD {
            let ptr = self.hazards[i].load(Ordering::Acquire);
            if !ptr.is_null() {
                hazards.push(ptr);
            }
        }
        
        hazards.sort_unstable();
        hazards.dedup();
        hazards
    }

    /// Attempt to reclaim retired nodes
    fn try_reclaim(&self, thread_id: usize) {
        self.reclaim_count.fetch_add(1, Ordering::Relaxed);
        
        let hazards = self.collect_hazards();
        
        let retired = unsafe { &mut *self.retired[thread_id].get() };
        
        let mut still_hazardous = Vec::new();
        let mut reclaimed = 0u64;
        
        for node in retired.nodes.drain(..) {
            if hazards.binary_search(&node).is_ok() {
                // Node is still protected, keep it
                still_hazardous.push(node);
            } else {
                // Safe to reclaim
                unsafe {
                    drop(Box::from_raw(node));
                }
                reclaimed += 1;
            }
        }
        
        retired.nodes = still_hazardous;
        self.nodes_reclaimed.fetch_add(reclaimed, Ordering::Relaxed);
    }

    pub fn stats(&self) -> HazardStats {
        HazardStats {
            reclaim_operations: self.reclaim_count.load(Ordering::Relaxed),
            nodes_reclaimed: self.nodes_reclaimed.load(Ordering::Relaxed),
            thread_count: self.thread_count.load(Ordering::Relaxed),
        }
    }
}

impl<T> Default for HazardDomain<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HazardStats {
    pub reclaim_operations: u64,
    pub nodes_reclaimed: u64,
    pub thread_count: usize,
}

/// Per-thread handle for hazard pointer operations
pub struct ThreadHandle<'a, T> {
    id: usize,
    domain: &'a HazardDomain<T>,
}

impl<'a, T> ThreadHandle<'a, T> {
    /// Protect a pointer with a hazard pointer - O(1) operation
    #[inline]
    pub fn protect(&self, ptr: *mut HazardNode<T>, slot: usize) {
        assert!(slot < HAZARDS_PER_THREAD);
        let hp_idx = self.id * HAZARDS_PER_THREAD + slot;
        self.domain.hazards[hp_idx].store(ptr, Ordering::Release);
    }

    /// Clear a hazard pointer slot - O(1) operation
    #[inline]
    pub fn clear(&self, slot: usize) {
        assert!(slot < HAZARDS_PER_THREAD);
        let hp_idx = self.id * HAZARDS_PER_THREAD + slot;
        self.domain.hazards[hp_idx].store(ptr::null_mut(), Ordering::Release);
    }

    /// Retire a node for later reclamation
    pub fn retire(&self, node: *mut HazardNode<T>) {
        let retired = unsafe { &mut *self.domain.retired[self.id].get() };
        retired.push(node);
        
        if retired.len() >= RECLAIM_THRESHOLD {
            self.domain.try_reclaim(self.id);
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub thread_count: usize,
    pub mean_latency_ns: f64,
    pub p50_latency_ns: f64,
    pub p99_latency_ns: f64,
    pub throughput_ops_per_sec: f64,
}

impl BenchmarkResult {
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.2},{:.2},{:.2},{:.2}",
            self.name,
            self.thread_count,
            self.mean_latency_ns,
            self.p50_latency_ns,
            self.p99_latency_ns,
            self.throughput_ops_per_sec
        )
    }
}

/// Run hazard pointer benchmarks
pub fn run_benchmarks() -> Vec<BenchmarkResult> {
    const ITERATIONS: usize = 10000;
    const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
    
    let mut results = vec![];
    
    println!("Hazard Pointer Baseline Benchmarks");
    println!("===================================\n");
    
    for &thread_count in THREAD_COUNTS {
        println!("Thread count: {}", thread_count);
        
        // Protect operation benchmark
        let domain: Arc<HazardDomain<u64>> = Arc::new(HazardDomain::new());
        let handles: Vec<_> = (0..thread_count)
            .map(|_| domain.register())
            .collect();
        
        // Create test nodes
        let nodes: Vec<_> = (0..thread_count)
            .map(|i| HazardNode::new(i as u64))
            .collect();
        
        let mut latencies = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            for (i, handle) in handles.iter().enumerate() {
                handle.protect(nodes[i % nodes.len()], 0);
            }
            latencies.push(start.elapsed().as_nanos() as f64);
        }
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        
        println!("  Protect latency: {:.2} ns", mean / thread_count as f64);
        results.push(BenchmarkResult {
            name: "hp_protect".to_string(),
            thread_count,
            mean_latency_ns: mean / thread_count as f64,
            p50_latency_ns: latencies[latencies.len() / 2] / thread_count as f64,
            p99_latency_ns: latencies[(latencies.len() as f64 * 0.99) as usize] / thread_count as f64,
            throughput_ops_per_sec: 1e9 / (mean / thread_count as f64),
        });
        
        // Scan operation benchmark (this is O(H × T))
        let mut latencies = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            let _ = domain.collect_hazards();
            latencies.push(start.elapsed().as_nanos() as f64);
        }
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        
        println!("  Scan latency: {:.2} ns (O(H×T) = O({}))", mean, thread_count * HAZARDS_PER_THREAD);
        results.push(BenchmarkResult {
            name: "hp_scan".to_string(),
            thread_count,
            mean_latency_ns: mean,
            p50_latency_ns: latencies[latencies.len() / 2],
            p99_latency_ns: latencies[(latencies.len() as f64 * 0.99) as usize],
            throughput_ops_per_sec: 1e9 / mean,
        });
        
        // Cleanup
        for node in nodes {
            unsafe { drop(Box::from_raw(node)); }
        }
        
        println!();
    }
    
    results
}

fn main() {
    let results = run_benchmarks();
    
    // Export CSV
    use std::io::Write;
    let mut file = std::fs::File::create("hazard_pointer_baseline.csv").unwrap();
    writeln!(file, "name,thread_count,mean_ns,p50_ns,p99_ns,throughput").unwrap();
    for r in &results {
        writeln!(file, "{}", r.to_csv_row()).unwrap();
    }
    println!("Results exported to hazard_pointer_baseline.csv");
}
