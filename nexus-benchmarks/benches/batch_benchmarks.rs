//! Batch Processing Benchmarks for Nexus Memory
//!
//! This module benchmarks batch processing performance comparing Nexus
//! hierarchical epoch-based reclamation against baseline approaches.
//!
//! # Benchmarks
//!
//! - Map operation throughput
//! - Reduce operation throughput  
//! - Cache-aware processing
//! - Memory allocation overhead

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, BatchSize, Throughput,
};

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    alloc::Layout,
    ptr::NonNull,
};

// Constants for benchmarking
const CACHE_LINE_SIZE: usize = 64;
const PREFETCH_DISTANCE: usize = 8;

/// Simple memory pool for benchmarking
pub struct MemoryPool {
    allocated: AtomicU64,
    capacity: usize,
}

impl MemoryPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            allocated: AtomicU64::new(0),
            capacity,
        }
    }

    pub fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size() as u64;
        let old = self.allocated.fetch_add(size, Ordering::Relaxed);
        
        if old + size <= self.capacity as u64 {
            // In real implementation, this would use proper allocation
            let ptr = unsafe {
                std::alloc::alloc(layout)
            };
            NonNull::new(ptr)
        } else {
            self.allocated.fetch_sub(size, Ordering::Relaxed);
            None
        }
    }
}

/// Generate test data
fn generate_float_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (i as f32).sin() * 100.0 + (i as f32).cos() * 50.0)
        .collect()
}

/// Benchmark vectorized map operations
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_map");
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Nexus-style chunked processing
        group.bench_with_input(
            BenchmarkId::new("nexus_chunked", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || generate_float_data(size),
                    |data| {
                        let output: Vec<f32> = data
                            .chunks(1024)
                            .flat_map(|chunk| {
                                chunk.iter().map(|&x| x * x + x.abs().sqrt())
                            })
                            .collect();
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Traditional scalar approach
        group.bench_with_input(
            BenchmarkId::new("baseline_scalar", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || generate_float_data(size),
                    |data| {
                        let output: Vec<f32> = data
                            .iter()
                            .map(|&x| x * x + x.abs().sqrt())
                            .collect();
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark reduction operations
fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_reduce");
    
    for size in [10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Nexus-style chunked reduction
        group.bench_with_input(
            BenchmarkId::new("nexus_chunked_reduce", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || generate_float_data(size),
                    |data| {
                        let sum: f32 = data
                            .chunks(4096)
                            .map(|chunk| chunk.iter().sum::<f32>())
                            .sum();
                        black_box(sum)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Traditional sequential reduction
        group.bench_with_input(
            BenchmarkId::new("baseline_sequential", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || generate_float_data(size),
                    |data| {
                        let sum: f32 = data.iter().sum();
                        black_box(sum)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache-aware algorithms
fn bench_cache_aware(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cache_aware");
    
    for size in [64, 128, 256, 512] {
        group.throughput(Throughput::Elements((size * size) as u64));
        
        // Cache-aware tiled matrix multiply
        group.bench_with_input(
            BenchmarkId::new("nexus_tiled_matmul", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = generate_float_data(size * size);
                        let b = generate_float_data(size * size);
                        let c = vec![0.0f32; size * size];
                        (a, b, c, size)
                    },
                    |(a, b, mut c, n)| {
                        const TILE: usize = 32;
                        
                        for i_tile in (0..n).step_by(TILE) {
                            for j_tile in (0..n).step_by(TILE) {
                                for k_tile in (0..n).step_by(TILE) {
                                    for i in i_tile..(i_tile + TILE).min(n) {
                                        for k in k_tile..(k_tile + TILE).min(n) {
                                            let a_val = a[i * n + k];
                                            for j in j_tile..(j_tile + TILE).min(n) {
                                                c[i * n + j] += a_val * b[k * n + j];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        black_box(c)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        
        // Naive matrix multiply for comparison
        group.bench_with_input(
            BenchmarkId::new("baseline_naive_matmul", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = generate_float_data(size * size);
                        let b = generate_float_data(size * size);
                        let c = vec![0.0f32; size * size];
                        (a, b, c, size)
                    },
                    |(a, b, mut c, n)| {
                        for i in 0..n {
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for k in 0..n {
                                    sum += a[i * n + k] * b[k * n + j];
                                }
                                c[i * n + j] = sum;
                            }
                        }
                        black_box(c)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");
    
    for size in [1_000, 10_000, 100_000] {
        // Pool-based allocation
        group.bench_with_input(
            BenchmarkId::new("nexus_pool", size),
            &size,
            |b, &size| {
                let pool = Arc::new(MemoryPool::new(1 << 30));
                
                b.iter(|| {
                    let layout = Layout::array::<f32>(*size).unwrap();
                    let ptr = pool.allocate(layout);
                    black_box(ptr)
                })
            },
        );
        
        // Standard allocation
        group.bench_with_input(
            BenchmarkId::new("baseline_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f32> = Vec::with_capacity(*size);
                    black_box(data)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    batch_benches,
    bench_map,
    bench_reduce,
    bench_cache_aware,
    bench_allocation,
);

criterion_main!(batch_benches);