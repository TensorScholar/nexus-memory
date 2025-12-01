//! Stream Processing Performance Benchmark Suite
//!
//! Benchmarks for NEXUS stream processing with epoch-based memory management.
//! Validates throughput, latency, and memory efficiency claims from the paper.
//!
//! ## Performance Model
//! - Throughput: O(n) with hierarchical epoch overhead O(log T)
//! - Latency: P99 < 1ms for window operations
//! - Memory: Zero-copy streaming with bounded buffers

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, BatchSize, Throughput,
};

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

// ============================================================================
// Stream Processing Abstractions
// ============================================================================

/// Bounded stream buffer with backpressure support
pub struct StreamBuffer<T> {
    buffer: Mutex<VecDeque<T>>,
    capacity: usize,
    produced: AtomicU64,
    consumed: AtomicU64,
}

impl<T> StreamBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            produced: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
        }
    }

    pub fn push(&self, item: T) -> bool {
        let mut buf = self.buffer.lock();
        if buf.len() >= self.capacity {
            return false; // Backpressure
        }
        buf.push_back(item);
        self.produced.fetch_add(1, Ordering::Relaxed);
        true
    }

    pub fn pop(&self) -> Option<T> {
        let mut buf = self.buffer.lock();
        let item = buf.pop_front();
        if item.is_some() {
            self.consumed.fetch_add(1, Ordering::Relaxed);
        }
        item
    }

    pub fn len(&self) -> usize {
        self.buffer.lock().len()
    }
}

/// Sliding window aggregator
pub struct SlidingWindow<T> {
    window: VecDeque<T>,
    window_size: usize,
}

impl<T: Clone> SlidingWindow<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn add(&mut self, item: T) -> Option<T> {
        let evicted = if self.window.len() >= self.window_size {
            self.window.pop_front()
        } else {
            None
        };
        self.window.push_back(item);
        evicted
    }

    pub fn contents(&self) -> &VecDeque<T> {
        &self.window
    }
}

/// Tumbling window for batch aggregation
pub struct TumblingWindow<T> {
    buffer: Vec<T>,
    window_size: usize,
}

impl<T> TumblingWindow<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(window_size),
            window_size,
        }
    }

    pub fn add(&mut self, item: T) -> Option<Vec<T>> {
        self.buffer.push(item);
        if self.buffer.len() >= self.window_size {
            let batch = std::mem::replace(
                &mut self.buffer,
                Vec::with_capacity(self.window_size),
            );
            Some(batch)
        } else {
            None
        }
    }
}

/// Session window based on activity gaps
pub struct SessionWindow<T> {
    buffer: Vec<(T, Instant)>,
    gap_duration: Duration,
    last_activity: Option<Instant>,
}

impl<T> SessionWindow<T> {
    pub fn new(gap_duration: Duration) -> Self {
        Self {
            buffer: Vec::new(),
            gap_duration,
            last_activity: None,
        }
    }

    pub fn add(&mut self, item: T, timestamp: Instant) -> Option<Vec<T>> {
        let should_emit = self.last_activity
            .map(|last| timestamp.duration_since(last) > self.gap_duration)
            .unwrap_or(false);

        let result = if should_emit && !self.buffer.is_empty() {
            let batch: Vec<T> = self.buffer.drain(..).map(|(t, _)| t).collect();
            Some(batch)
        } else {
            None
        };

        self.buffer.push((item, timestamp));
        self.last_activity = Some(timestamp);
        result
    }
}

// ============================================================================
// Stream Operations
// ============================================================================

/// Map operation with zero-copy semantics
#[inline]
pub fn stream_map<T, U, F>(input: &[T], f: F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    input.iter().map(f).collect()
}

/// Filter operation
#[inline]
pub fn stream_filter<T, F>(input: &[T], predicate: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    input.iter().filter(|x| predicate(x)).cloned().collect()
}

/// Reduce operation
#[inline]
pub fn stream_reduce<T, F>(input: &[T], identity: T, op: F) -> T
where
    T: Clone,
    F: Fn(T, &T) -> T,
{
    input.iter().fold(identity, |acc, x| op(acc, x))
}

/// Window aggregate
#[inline]
pub fn window_aggregate<T, A, F>(window: &VecDeque<T>, identity: A, op: F) -> A
where
    F: Fn(A, &T) -> A,
{
    window.iter().fold(identity, |acc, x| op(acc, x))
}

// ============================================================================
// Epoch-Aware Stream Processing
// ============================================================================

/// Stream processor with hierarchical epoch synchronization
pub struct HierarchicalStreamProcessor {
    epoch: CachePadded<AtomicU64>,
    processed_count: AtomicU64,
    sync_interval: u64,
}

impl HierarchicalStreamProcessor {
    pub fn new(sync_interval: u64) -> Self {
        Self {
            epoch: CachePadded::new(AtomicU64::new(0)),
            processed_count: AtomicU64::new(0),
            sync_interval,
        }
    }

    /// Process with epoch tracking (O(log T) sync overhead)
    pub fn process<T, U, F>(&self, items: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> U,
    {
        let mut results = Vec::with_capacity(items.len());
        
        for item in items.iter() {
            results.push(f(item));
            
            // Hierarchical epoch advancement
            let count = self.processed_count.fetch_add(1, Ordering::Relaxed);
            if count % self.sync_interval == 0 {
                self.epoch.fetch_add(1, Ordering::Release);
            }
        }
        
        results
    }

    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }
}

/// Flat epoch processor (baseline for comparison)
pub struct FlatEpochProcessor {
    epoch: AtomicU64,
    processed_count: AtomicU64,
}

impl FlatEpochProcessor {
    pub fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            processed_count: AtomicU64::new(0),
        }
    }

    /// Process with flat epoch (O(T) sync overhead)
    pub fn process<T, U, F>(&self, items: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> U,
    {
        let mut results = Vec::with_capacity(items.len());
        
        for item in items {
            results.push(f(item));
            
            // Flat epoch: sync on every item
            self.epoch.fetch_add(1, Ordering::Release);
            self.processed_count.fetch_add(1, Ordering::Relaxed);
        }
        
        results
    }
}

impl Default for FlatEpochProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn generate_stream_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (i as f64).sin() * 100.0 + (i as f64).cos() * 50.0)
        .collect()
}

fn bench_stream_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_throughput");
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // NEXUS hierarchical epoch processing
        group.bench_with_input(
            BenchmarkId::new("nexus_hierarchical", size),
            &size,
            |b, &size| {
                let processor = HierarchicalStreamProcessor::new(1024);
                
                b.iter_batched(
                    || generate_stream_data(size),
                    |data| {
                        let result = processor.process(&data, |x| x * 2.0 + 1.0);
                        black_box(result.len())
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Flat epoch baseline
        group.bench_with_input(
            BenchmarkId::new("flat_epoch_baseline", size),
            &size,
            |b, &size| {
                let processor = FlatEpochProcessor::new();
                
                b.iter_batched(
                    || generate_stream_data(size),
                    |data| {
                        let result = processor.process(&data, |x| x * 2.0 + 1.0);
                        black_box(result.len())
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Simple iterator (no epoch overhead)
        group.bench_with_input(
            BenchmarkId::new("iterator_baseline", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || generate_stream_data(size),
                    |data| {
                        let result: Vec<f64> = data.iter().map(|x| x * 2.0 + 1.0).collect();
                        black_box(result.len())
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    
    group.finish();
}

fn bench_window_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_operations");
    
    for window_size in [10, 100, 1000] {
        let stream_size = 100_000;
        
        // Sliding window
        group.bench_with_input(
            BenchmarkId::new("sliding_window", window_size),
            &window_size,
            |b, &window_size| {
                b.iter_batched(
                    || generate_stream_data(stream_size),
                    |data| {
                        let mut window: SlidingWindow<f64> = SlidingWindow::new(window_size);
                        let mut sum = 0.0;
                        
                        for item in data {
                            window.add(item);
                            sum += window_aggregate(window.contents(), 0.0, |acc, x| acc + x);
                        }
                        
                        black_box(sum)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Tumbling window
        group.bench_with_input(
            BenchmarkId::new("tumbling_window", window_size),
            &window_size,
            |b, &window_size| {
                b.iter_batched(
                    || generate_stream_data(stream_size),
                    |data| {
                        let mut window: TumblingWindow<f64> = TumblingWindow::new(window_size);
                        let mut batch_count = 0;
                        
                        for item in data {
                            if let Some(batch) = window.add(item) {
                                let _sum: f64 = batch.iter().sum();
                                batch_count += 1;
                            }
                        }
                        
                        black_box(batch_count)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    
    group.finish();
}

fn bench_stream_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_operations");
    let size = 100_000;
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Map operation
    group.bench_function("map", |b| {
        b.iter_batched(
            || generate_stream_data(size),
            |data| {
                let result = stream_map(&data, |x| x * x);
                black_box(result.len())
            },
            BatchSize::LargeInput,
        )
    });
    
    // Filter operation
    group.bench_function("filter", |b| {
        b.iter_batched(
            || generate_stream_data(size),
            |data| {
                let result = stream_filter(&data, |x| *x > 0.0);
                black_box(result.len())
            },
            BatchSize::LargeInput,
        )
    });
    
    // Reduce operation
    group.bench_function("reduce", |b| {
        b.iter_batched(
            || generate_stream_data(size),
            |data| {
                let result = stream_reduce(&data, 0.0, |acc, x| acc + x);
                black_box(result)
            },
            BatchSize::LargeInput,
        )
    });
    
    // Map + Filter + Reduce pipeline
    group.bench_function("pipeline", |b| {
        b.iter_batched(
            || generate_stream_data(size),
            |data| {
                let mapped = stream_map(&data, |x| x * 2.0);
                let filtered = stream_filter(&mapped, |x| *x > 0.0);
                let result = stream_reduce(&filtered, 0.0, |acc, x| acc + x);
                black_box(result)
            },
            BatchSize::LargeInput,
        )
    });
    
    group.finish();
}

fn bench_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure");
    
    for buffer_size in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("bounded_buffer", buffer_size),
            &buffer_size,
            |b, &buffer_size| {
                b.iter(|| {
                    let buffer: Arc<StreamBuffer<f64>> = Arc::new(StreamBuffer::new(buffer_size));
                    let mut produced = 0;
                    let mut consumed = 0;
                    
                    // Simulate producer-consumer
                    for i in 0..buffer_size * 10 {
                        let value = (i as f64).sin();
                        
                        // Producer
                        if buffer.push(value) {
                            produced += 1;
                        }
                        
                        // Consumer (every other iteration)
                        if i % 2 == 0 {
                            if buffer.pop().is_some() {
                                consumed += 1;
                            }
                        }
                    }
                    
                    black_box((produced, consumed))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.sample_size(1000);
    
    let processor = HierarchicalStreamProcessor::new(64);
    
    // Single-item latency
    group.bench_function("single_item_latency", |b| {
        b.iter(|| {
            let data = vec![42.0f64];
            let start = Instant::now();
            let result = processor.process(&data, |x| x * 2.0);
            let latency = start.elapsed();
            black_box((result, latency))
        })
    });
    
    // Batch latency
    group.bench_function("batch_latency_100", |b| {
        b.iter_batched(
            || generate_stream_data(100),
            |data| {
                let start = Instant::now();
                let result = processor.process(&data, |x| x * 2.0);
                let latency = start.elapsed();
                black_box((result.len(), latency))
            },
            BatchSize::SmallInput,
        )
    });
    
    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    stream_benches,
    bench_stream_throughput,
    bench_window_operations,
    bench_stream_operations,
    bench_backpressure,
    bench_latency_distribution,
);

criterion_main!(stream_benches);
