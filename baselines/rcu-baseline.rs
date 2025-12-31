//! Read-Copy-Update (RCU) Baseline Implementation
//!
//! This module provides an RCU-style memory reclamation baseline
//! for comparison against Nexus Memory's epoch-based approach.
//!
//! # RCU Overview
//!
//! Read-Copy-Update provides:
//! 1. Wait-free reads (readers never block)
//! 2. Quiescent state-based reclamation
//! 3. Writers must copy-on-write to modify data
//!
//! # RCU vs Epoch-Based Reclamation
//!
//! RCU advantages:
//! - Extremely low read-side overhead
//! - Well-understood grace period semantics
//!
//! Nexus epoch advantages:
//! - Lower writer overhead (no copy-on-write)
//! - Better write throughput
//! - More flexible for non-read-dominated workloads

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::thread;

/// Maximum threads supported
const MAX_THREADS: usize = 128;
/// Grace period batch threshold
const GRACE_PERIOD_BATCH: usize = 64;

/// Thread-local quiescent state tracking
#[repr(align(64))]
struct ThreadState {
    /// Current quiescent state counter
    qstate: AtomicU64,
    /// Whether thread is currently in critical section
    in_critical: AtomicU64,
}

impl ThreadState {
    const fn new() -> Self {
        Self {
            qstate: AtomicU64::new(0),
            in_critical: AtomicU64::new(0),
        }
    }
}

/// Callback for deferred reclamation
struct DeferredCallback {
    callback: Box<dyn FnOnce() + Send>,
    grace_period: u64,
}

/// RCU domain managing quiescent state detection
pub struct RcuDomain {
    /// Global grace period counter
    grace_period: AtomicU64,
    /// Per-thread quiescent states
    thread_states: Box<[ThreadState; MAX_THREADS]>,
    /// Number of registered threads
    thread_count: AtomicUsize,
    /// Deferred callbacks
    deferred: UnsafeCell<Vec<DeferredCallback>>,
    /// Statistics
    grace_periods_completed: AtomicU64,
    callbacks_executed: AtomicU64,
}

unsafe impl Send for RcuDomain {}
unsafe impl Sync for RcuDomain {}

impl RcuDomain {
    pub fn new() -> Self {
        Self {
            grace_period: AtomicU64::new(0),
            thread_states: Box::new([(); MAX_THREADS].map(|_| ThreadState::new())),
            thread_count: AtomicUsize::new(0),
            deferred: UnsafeCell::new(Vec::new()),
            grace_periods_completed: AtomicU64::new(0),
            callbacks_executed: AtomicU64::new(0),
        }
    }

    /// Register a new thread
    pub fn register(&self) -> RcuHandle<'_> {
        let id = self.thread_count.fetch_add(1, Ordering::Relaxed);
        assert!(id < MAX_THREADS, "Maximum thread count exceeded");
        RcuHandle { id, domain: self }
    }

    /// Get current grace period
    pub fn current_grace_period(&self) -> u64 {
        self.grace_period.load(Ordering::Acquire)
    }

    /// Check if all threads have passed through a quiescent state
    fn all_threads_quiescent(&self, target_gp: u64) -> bool {
        let count = self.thread_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            let state = &self.thread_states[i];
            
            // If thread is in critical section, check its observed GP
            let in_crit = state.in_critical.load(Ordering::Acquire);
            if in_crit != 0 {
                let thread_gp = state.qstate.load(Ordering::Acquire);
                if thread_gp < target_gp {
                    return false;
                }
            }
        }
        
        true
    }

    /// Synchronize - wait for a grace period to elapse
    /// This blocks until all pre-existing read-side critical sections complete
    pub fn synchronize(&self) {
        // Advance grace period
        let new_gp = self.grace_period.fetch_add(1, Ordering::AcqRel) + 1;
        
        // Wait for all threads to observe new grace period
        while !self.all_threads_quiescent(new_gp) {
            thread::yield_now();
        }
        
        self.grace_periods_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Defer a callback to execute after current grace period
    pub fn call_rcu<F: FnOnce() + Send + 'static>(&self, callback: F) {
        let gp = self.current_grace_period();
        let deferred = DeferredCallback {
            callback: Box::new(callback),
            grace_period: gp,
        };
        
        unsafe {
            (*self.deferred.get()).push(deferred);
            
            if (*self.deferred.get()).len() >= GRACE_PERIOD_BATCH {
                self.process_callbacks();
            }
        }
    }

    fn process_callbacks(&self) {
        self.synchronize();
        
        let current_gp = self.current_grace_period();
        let deferred = unsafe { &mut *self.deferred.get() };
        
        let (ready, pending): (Vec<_>, Vec<_>) = deferred
            .drain(..)
            .partition(|cb| cb.grace_period < current_gp);
        
        *deferred = pending;
        
        for cb in ready {
            (cb.callback)();
            self.callbacks_executed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn stats(&self) -> RcuStats {
        RcuStats {
            grace_periods: self.grace_periods_completed.load(Ordering::Relaxed),
            callbacks_executed: self.callbacks_executed.load(Ordering::Relaxed),
            thread_count: self.thread_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for RcuDomain {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RcuStats {
    pub grace_periods: u64,
    pub callbacks_executed: u64,
    pub thread_count: usize,
}

/// Per-thread RCU handle
pub struct RcuHandle<'a> {
    id: usize,
    domain: &'a RcuDomain,
}

impl<'a> RcuHandle<'a> {
    /// Enter read-side critical section - O(1), nearly free
    #[inline]
    pub fn read_lock(&self) {
        let state = &self.domain.thread_states[self.id];
        
        // Record current grace period
        let gp = self.domain.grace_period.load(Ordering::Acquire);
        state.qstate.store(gp, Ordering::Release);
        
        // Mark as in critical section
        state.in_critical.store(1, Ordering::Release);
        
        // Memory fence to ensure read-side operations are ordered
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Exit read-side critical section - O(1)
    #[inline]
    pub fn read_unlock(&self) {
        let state = &self.domain.thread_states[self.id];
        state.in_critical.store(0, Ordering::Release);
    }

    /// Report quiescent state (thread is not holding any RCU references)
    #[inline]
    pub fn quiescent_state(&self) {
        let state = &self.domain.thread_states[self.id];
        let gp = self.domain.grace_period.load(Ordering::Acquire);
        state.qstate.store(gp, Ordering::Release);
    }
}

/// RAII read-side critical section guard
pub struct RcuReadGuard<'a> {
    handle: &'a RcuHandle<'a>,
}

impl<'a> RcuReadGuard<'a> {
    pub fn new(handle: &'a RcuHandle<'a>) -> Self {
        handle.read_lock();
        Self { handle }
    }
}

impl Drop for RcuReadGuard<'_> {
    fn drop(&mut self) {
        self.handle.read_unlock();
    }
}

/// RCU-protected pointer wrapper
pub struct RcuPtr<T> {
    ptr: AtomicPtr<T>,
}

impl<T> RcuPtr<T> {
    pub fn new(value: T) -> Self {
        Self {
            ptr: AtomicPtr::new(Box::into_raw(Box::new(value))),
        }
    }

    pub fn null() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Read the current value (must be in read-side critical section)
    #[inline]
    pub fn read(&self) -> *const T {
        self.ptr.load(Ordering::Acquire)
    }

    /// Update the pointer, returning the old value for deferred reclamation
    pub fn update(&self, new_value: T) -> *mut T {
        let new_ptr = Box::into_raw(Box::new(new_value));
        self.ptr.swap(new_ptr, Ordering::AcqRel)
    }

    /// Swap with null
    pub fn take(&self) -> *mut T {
        self.ptr.swap(ptr::null_mut(), Ordering::AcqRel)
    }
}

impl<T> Default for RcuPtr<T> {
    fn default() -> Self {
        Self::null()
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

/// Run RCU benchmarks
pub fn run_benchmarks() -> Vec<BenchmarkResult> {
    const ITERATIONS: usize = 10000;
    const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
    
    let mut results = vec![];
    
    println!("RCU Baseline Benchmarks");
    println!("=======================\n");
    
    for &thread_count in THREAD_COUNTS {
        println!("Thread count: {}", thread_count);
        
        let domain = Arc::new(RcuDomain::new());
        let handles: Vec<_> = (0..thread_count)
            .map(|_| domain.register())
            .collect();
        
        // Read-side critical section benchmark
        let mut latencies = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            for handle in &handles {
                handle.read_lock();
                handle.read_unlock();
            }
            latencies.push(start.elapsed().as_nanos() as f64);
        }
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        
        println!("  Read lock/unlock: {:.2} ns", mean / thread_count as f64);
        results.push(BenchmarkResult {
            name: "rcu_read_section".to_string(),
            thread_count,
            mean_latency_ns: mean / thread_count as f64,
            p50_latency_ns: latencies[latencies.len() / 2] / thread_count as f64,
            p99_latency_ns: latencies[(latencies.len() as f64 * 0.99) as usize] / thread_count as f64,
            throughput_ops_per_sec: 1e9 / (mean / thread_count as f64),
        });
        
        // Grace period synchronization benchmark
        let mut latencies = Vec::with_capacity(ITERATIONS / 10);
        for _ in 0..ITERATIONS / 10 {
            let start = Instant::now();
            domain.synchronize();
            latencies.push(start.elapsed().as_nanos() as f64);
        }
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        
        println!("  Synchronize: {:.2} ns (O(T) = O({}))", mean, thread_count);
        results.push(BenchmarkResult {
            name: "rcu_synchronize".to_string(),
            thread_count,
            mean_latency_ns: mean,
            p50_latency_ns: latencies[latencies.len() / 2],
            p99_latency_ns: latencies[(latencies.len() as f64 * 0.99) as usize],
            throughput_ops_per_sec: 1e9 / mean,
        });
        
        println!();
    }
    
    results
}

fn main() {
    let results = run_benchmarks();
    
    // Export CSV
    use std::io::Write;
    let mut file = std::fs::File::create("rcu_baseline.csv").unwrap();
    writeln!(file, "name,thread_count,mean_ns,p50_ns,p99_ns,throughput").unwrap();
    for r in &results {
        writeln!(file, "{}", r.to_csv_row()).unwrap();
    }
    println!("Results exported to rcu_baseline.csv");
}
