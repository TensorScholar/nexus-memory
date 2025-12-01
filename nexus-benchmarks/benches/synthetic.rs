//! Synthetic Workload Generator for Nexus Memory Benchmarks
//!
//! This module provides synthetic workload generation for comprehensive
//! testing of memory reclamation across different access patterns.
//!
//! # Workload Types
//!
//! - CPU-bound: High compute, sequential access
//! - Memory-bound: Random access patterns
//! - Cache-hostile: Patterns that defeat caching

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, BatchSize, Throughput,
};

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

/// Complexity class for workload generation
#[derive(Debug, Clone, Copy)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
}

impl ComplexityClass {
    pub fn operation_count(&self, n: usize) -> usize {
        match self {
            ComplexityClass::Constant => 1,
            ComplexityClass::Logarithmic => (n as f64).log2() as usize + 1,
            ComplexityClass::Linear => n,
            ComplexityClass::Linearithmic => n * ((n as f64).log2() as usize + 1),
            ComplexityClass::Quadratic => n * n,
        }
    }
}

/// Memory access pattern
#[derive(Debug, Clone, Copy)]
pub enum MemoryPattern {
    Sequential { stride: usize },
    Random,
    Strided { strides: [usize; 4] },
    Tiled { tile_size: usize },
}

impl MemoryPattern {
    pub fn generate_indices(&self, size: usize, count: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        
        match self {
            MemoryPattern::Sequential { stride } => {
                for i in 0..count {
                    indices.push((i * stride) % size);
                }
            }
            MemoryPattern::Random => {
                // Simple LCG for reproducible "random" indices
                let mut state = 0x1337u64;
                for _ in 0..count {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    indices.push((state as usize) % size);
                }
            }
            MemoryPattern::Strided { strides } => {
                for i in 0..count {
                    let stride = strides[i % strides.len()];
                    indices.push((i * stride) % size);
                }
            }
            MemoryPattern::Tiled { tile_size } => {
                let tiles = size / tile_size.max(1);
                for i in 0..count {
                    let tile = (i / tile_size) % tiles.max(1);
                    let offset = i % tile_size;
                    indices.push((tile * tile_size + offset) % size);
                }
            }
        }
        
        indices
    }
}

/// Statistical distribution for workload values
#[derive(Debug, Clone, Copy)]
pub enum Distribution {
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std_dev: f64 },
}

impl Distribution {
    pub fn sample_batch(&self, count: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut values = Vec::with_capacity(count);
        
        match self {
            Distribution::Uniform { min, max } => {
                for _ in 0..count {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = (state as f64) / (u64::MAX as f64);
                    values.push(min + u * (max - min));
                }
            }
            Distribution::Normal { mean, std_dev } => {
                // Box-Muller transform
                for i in 0..(count / 2 + 1) {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u1 = (state as f64) / (u64::MAX as f64);
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u2 = (state as f64) / (u64::MAX as f64);
                    
                    let z0 = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let z1 = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                    
                    values.push(mean + std_dev * z0);
                    if values.len() < count {
                        values.push(mean + std_dev * z1);
                    }
                }
            }
        }
        
        values.truncate(count);
        values
    }
}

/// Workload specification
#[derive(Debug, Clone)]
pub struct WorkloadSpec {
    pub name: &'static str,
    pub complexity: ComplexityClass,
    pub memory_pattern: MemoryPattern,
    pub distribution: Distribution,
    pub parallel_fraction: f64,
}

/// Cross-paradigm workload
#[derive(Clone)]
pub struct Workload {
    pub batch_data: Vec<u64>,
    pub stream_data: Vec<f64>,
    pub graph_edges: Vec<(usize, usize)>,
    pub memory_indices: Vec<usize>,
}

impl Workload {
    pub fn generate(spec: &WorkloadSpec, size: usize) -> Self {
        let batch_data: Vec<u64> = (0..size).map(|i| i as u64).collect();
        let stream_data = spec.distribution.sample_batch(size, 0x1337);
        
        // Generate graph edges
        let num_nodes = (size as f64).sqrt() as usize;
        let mut edges = Vec::with_capacity(size);
        let mut state = 0x42u64;
        for _ in 0..size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let src = (state as usize) % num_nodes;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let dst = (state as usize) % num_nodes;
            edges.push((src, dst));
        }
        
        let memory_indices = spec.memory_pattern.generate_indices(size, size);
        
        Workload {
            batch_data,
            stream_data,
            graph_edges: edges,
            memory_indices,
        }
    }
    
    pub fn execute(&self) -> u64 {
        // Batch processing
        let batch_result: u64 = self.batch_data.iter()
            .enumerate()
            .map(|(i, &val)| {
                let idx = self.memory_indices[i % self.memory_indices.len()];
                val.wrapping_mul(idx as u64)
            })
            .sum();
        
        // Stream processing
        let stream_result: f64 = self.stream_data.iter()
            .zip(&self.memory_indices)
            .map(|(&val, &idx)| val * (idx as f64))
            .sum();
        
        // Graph processing
        let num_nodes = (self.batch_data.len() as f64).sqrt() as usize;
        let mut graph_state = vec![0u64; num_nodes.max(1)];
        for &(src, dst) in &self.graph_edges {
            if src < graph_state.len() && dst < graph_state.len() {
                graph_state[dst] = graph_state[dst].wrapping_add(graph_state[src] + 1);
            }
        }
        
        batch_result.wrapping_add(stream_result as u64)
            .wrapping_add(graph_state.iter().sum::<u64>())
    }
}

/// Benchmark synthetic workloads
fn bench_synthetic_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic_workloads");
    
    let specs = vec![
        WorkloadSpec {
            name: "cpu_bound",
            complexity: ComplexityClass::Quadratic,
            memory_pattern: MemoryPattern::Sequential { stride: 1 },
            distribution: Distribution::Normal { mean: 100.0, std_dev: 10.0 },
            parallel_fraction: 0.9,
        },
        WorkloadSpec {
            name: "memory_bound",
            complexity: ComplexityClass::Linear,
            memory_pattern: MemoryPattern::Random,
            distribution: Distribution::Uniform { min: 0.0, max: 1000.0 },
            parallel_fraction: 0.7,
        },
        WorkloadSpec {
            name: "cache_friendly",
            complexity: ComplexityClass::Linear,
            memory_pattern: MemoryPattern::Tiled { tile_size: 64 },
            distribution: Distribution::Normal { mean: 50.0, std_dev: 5.0 },
            parallel_fraction: 0.8,
        },
    ];
    
    for spec in &specs {
        for size in [1_000, 10_000, 100_000] {
            group.throughput(Throughput::Elements(size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(spec.name, size),
                &size,
                |b, &size| {
                    b.iter_batched(
                        || Workload::generate(spec, size),
                        |workload| black_box(workload.execute()),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    let patterns = vec![
        ("sequential", MemoryPattern::Sequential { stride: 1 }),
        ("strided_8", MemoryPattern::Sequential { stride: 8 }),
        ("strided_64", MemoryPattern::Sequential { stride: 64 }),
        ("random", MemoryPattern::Random),
        ("tiled_32", MemoryPattern::Tiled { tile_size: 32 }),
        ("tiled_256", MemoryPattern::Tiled { tile_size: 256 }),
    ];
    
    for (name, pattern) in patterns {
        for size in [10_000, 100_000, 1_000_000] {
            group.throughput(Throughput::Elements(size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &size,
                |b, &size| {
                    let data: Vec<u64> = (0..size).map(|i| i as u64).collect();
                    let indices = pattern.generate_indices(size, size);
                    
                    b.iter(|| {
                        let sum: u64 = indices.iter()
                            .map(|&idx| data[idx])
                            .sum();
                        black_box(sum)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark complexity scaling
fn bench_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_scaling");
    
    let complexities = vec![
        ComplexityClass::Constant,
        ComplexityClass::Logarithmic,
        ComplexityClass::Linear,
        ComplexityClass::Linearithmic,
    ];
    
    for complexity in complexities {
        for n in [100, 1_000, 10_000] {
            let ops = complexity.operation_count(n);
            group.throughput(Throughput::Elements(ops as u64));
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", complexity), n),
                &n,
                |b, &n| {
                    b.iter(|| {
                        let ops = complexity.operation_count(n);
                        let mut sum = 0u64;
                        for i in 0..ops {
                            sum = sum.wrapping_add(i as u64);
                        }
                        black_box(sum)
                    })
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    synthetic_benches,
    bench_synthetic_workloads,
    bench_memory_patterns,
    bench_complexity_scaling,
);

criterion_main!(synthetic_benches);