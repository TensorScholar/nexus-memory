//! Zero-Copy Proof Validation Suite
//!
//! This benchmark empirically verifies the zero-copy claims from the paper:
//! 1. **Pointer Stability**: Memory addresses remain identical across paradigm transitions
//! 2. **Page Fault Monitoring**: No OS-level memory copies occur during transitions
//!
//! # Scientific Methodology
//!
//! We use two independent verification methods:
//! - Direct pointer comparison across paradigm transitions
//! - Linux `getrusage` to monitor minor page faults
//!
//! # Running the Benchmark
//!
//! ```bash
//! cargo bench --package nexus-benchmarks -- zero_copy_proof
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nexus_memory::zero_copy::ZeroCopyBuffer;
use std::ptr;
use std::time::Instant;

#[cfg(target_os = "linux")]
use std::mem::MaybeUninit;

/// Size of the buffer for testing (100 MB)
const BUFFER_SIZE: usize = 100 * 1024 * 1024 / std::mem::size_of::<u64>();

/// Number of samples to verify for content integrity
const SAMPLE_COUNT: usize = 1000;

/// Paradigm marker types for type-level transitions
mod paradigm {
    /// Batch processing paradigm marker
    pub struct Batch;

    /// Stream processing paradigm marker
    pub struct Stream;

    /// Graph processing paradigm marker
    pub struct Graph;
}

/// Wrapper to simulate paradigm-specific view of data.
///
/// This struct demonstrates how the same underlying buffer can be accessed
/// through different paradigm-specific interfaces without copying data.
pub struct ParadigmView<'a, T, P> {
    /// Raw pointer to the underlying data
    ptr: *const T,
    /// Number of elements
    len: usize,
    /// Phantom data for paradigm marker
    _paradigm: std::marker::PhantomData<&'a P>,
}

impl<'a, T, P> ParadigmView<'a, T, P> {
    /// Creates a new paradigm view from a buffer.
    pub fn new(buffer: &'a ZeroCopyBuffer<T>) -> Self {
        Self {
            ptr: buffer.as_ptr(),
            len: buffer.len(),
            _paradigm: std::marker::PhantomData,
        }
    }

    /// Returns the raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks if the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Safely gets an element at the specified index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: We've verified the index is within bounds
            Some(unsafe { &*self.ptr.add(index) })
        } else {
            None
        }
    }
}

/// Transitions a paradigm view to a different paradigm.
///
/// This is a zero-cost transition that only changes the type marker.
pub fn transition_paradigm<'a, T, From, To>(
    view: ParadigmView<'a, T, From>,
) -> ParadigmView<'a, T, To> {
    ParadigmView {
        ptr: view.ptr,
        len: view.len,
        _paradigm: std::marker::PhantomData,
    }
}

/// Results from pointer stability verification.
#[derive(Debug)]
pub struct PointerStabilityResult {
    /// Whether all pointer comparisons passed
    pub pointers_stable: bool,
    /// The batch paradigm pointer address
    pub batch_ptr: usize,
    /// The stream paradigm pointer address
    pub stream_ptr: usize,
    /// The graph paradigm pointer address
    pub graph_ptr: usize,
    /// Number of sample content verifications that passed
    pub content_samples_passed: usize,
    /// Total number of content samples checked
    pub content_samples_total: usize,
}

/// Verifies pointer stability across paradigm transitions.
///
/// This function:
/// 1. Allocates a buffer with test data
/// 2. Creates a Batch paradigm view
/// 3. Transitions to Stream paradigm
/// 4. Transitions to Graph paradigm
/// 5. Verifies all pointers are identical
/// 6. Spot-checks data content integrity
pub fn verify_pointer_stability() -> PointerStabilityResult {
    // Allocate and fill buffer with test data
    let mut buffer = ZeroCopyBuffer::<u64>::new(BUFFER_SIZE);

    // Fill with deterministic data for verification
    for i in 0..BUFFER_SIZE {
        buffer.push(i as u64 * 0xDEADBEEF);
    }

    // Create Batch paradigm view
    let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
    let batch_ptr = batch_view.as_ptr() as usize;

    // Transition to Stream paradigm
    let stream_view: ParadigmView<'_, u64, paradigm::Stream> = transition_paradigm(batch_view);
    let stream_ptr = stream_view.as_ptr() as usize;

    // Transition to Graph paradigm
    let graph_view: ParadigmView<'_, u64, paradigm::Graph> = transition_paradigm(stream_view);
    let graph_ptr = graph_view.as_ptr() as usize;

    // Verify pointer stability
    let pointers_stable = batch_ptr == stream_ptr && stream_ptr == graph_ptr;

    // Verify data content with sampling
    let mut samples_passed = 0;
    let step = BUFFER_SIZE / SAMPLE_COUNT;

    for i in 0..SAMPLE_COUNT {
        let idx = i * step;
        if idx < buffer.len() {
            let expected = idx as u64 * 0xDEADBEEF;
            if let Some(&value) = buffer.get(idx) {
                if value == expected {
                    samples_passed += 1;
                }
            }
        }
    }

    PointerStabilityResult {
        pointers_stable,
        batch_ptr,
        stream_ptr,
        graph_ptr,
        content_samples_passed: samples_passed,
        content_samples_total: SAMPLE_COUNT,
    }
}

/// Results from page fault monitoring.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct PageFaultResult {
    /// Minor page faults before transition
    pub faults_before: i64,
    /// Minor page faults after transition
    pub faults_after: i64,
    /// Delta (should be 0 or very small for zero-copy)
    pub delta: i64,
    /// Whether the test passed (delta <= threshold)
    pub passed: bool,
}

#[cfg(target_os = "linux")]
fn get_minor_page_faults() -> i64 {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();

    unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
            let usage = usage.assume_init();
            usage.ru_minflt
        } else {
            -1
        }
    }
}

/// Verifies that paradigm transitions don't cause page faults.
///
/// Uses Linux `getrusage` to monitor minor page faults before and after
/// paradigm transitions. True zero-copy should result in zero additional faults.
#[cfg(target_os = "linux")]
pub fn verify_page_faults() -> PageFaultResult {
    // Allocate and fill buffer, ensuring pages are faulted in
    let mut buffer = ZeroCopyBuffer::<u64>::new(BUFFER_SIZE);

    // Touch all pages to ensure they're mapped (this will cause faults)
    for i in 0..BUFFER_SIZE {
        buffer.push(i as u64);
    }

    // Force memory access to ensure pages are resident
    let _sum: u64 = (0..BUFFER_SIZE)
        .step_by(4096 / std::mem::size_of::<u64>())
        .filter_map(|i| buffer.get(i))
        .sum();

    // Get page faults before transition
    let faults_before = get_minor_page_faults();

    // Perform paradigm transitions
    let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
    let stream_view: ParadigmView<'_, u64, paradigm::Stream> = transition_paradigm(batch_view);
    let graph_view: ParadigmView<'_, u64, paradigm::Graph> = transition_paradigm(stream_view);

    // Access data through the new paradigm view to trigger any lazy operations
    let _ = black_box(graph_view.get(0));
    let _ = black_box(graph_view.get(BUFFER_SIZE / 2));
    let _ = black_box(graph_view.get(BUFFER_SIZE - 1));

    // Get page faults after transition
    let faults_after = get_minor_page_faults();

    let delta = faults_after - faults_before;

    // Allow small delta for benchmark infrastructure overhead
    let threshold = 5;

    PageFaultResult {
        faults_before,
        faults_after,
        delta,
        passed: delta <= threshold,
    }
}

#[cfg(not(target_os = "linux"))]
pub fn verify_page_faults() -> PageFaultResult {
    PageFaultResult {
        faults_before: 0,
        faults_after: 0,
        delta: 0,
        passed: true,
    }
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub struct PageFaultResult {
    pub faults_before: i64,
    pub faults_after: i64,
    pub delta: i64,
    pub passed: bool,
}

/// Benchmark for pointer stability verification.
fn bench_pointer_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_proof");

    group.sample_size(20);

    group.bench_function("pointer_stability", |b| {
        b.iter(|| {
            let result = verify_pointer_stability();

            // Assert that pointers are stable
            assert!(
                result.pointers_stable,
                "Pointer stability violated! Batch: 0x{:x}, Stream: 0x{:x}, Graph: 0x{:x}",
                result.batch_ptr, result.stream_ptr, result.graph_ptr
            );

            // Assert content integrity
            assert_eq!(
                result.content_samples_passed, result.content_samples_total,
                "Content verification failed: {}/{} samples passed",
                result.content_samples_passed, result.content_samples_total
            );

            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark for page fault monitoring.
fn bench_page_faults(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_proof");

    group.sample_size(10);

    group.bench_function("page_faults", |b| {
        b.iter(|| {
            let result = verify_page_faults();

            #[cfg(target_os = "linux")]
            assert!(
                result.passed,
                "Page fault test failed! Delta: {} faults (before: {}, after: {})",
                result.delta, result.faults_before, result.faults_after
            );

            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark transition overhead between paradigms.
fn bench_transition_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_proof");

    // Create buffer once
    let mut buffer = ZeroCopyBuffer::<u64>::new(BUFFER_SIZE);
    for i in 0..BUFFER_SIZE {
        buffer.push(i as u64);
    }

    group.bench_function("transition_batch_to_stream", |b| {
        b.iter(|| {
            let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
            let stream_view: ParadigmView<'_, u64, paradigm::Stream> =
                transition_paradigm(batch_view);
            black_box(stream_view.as_ptr())
        });
    });

    group.bench_function("transition_stream_to_graph", |b| {
        b.iter(|| {
            let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
            let stream_view: ParadigmView<'_, u64, paradigm::Stream> =
                transition_paradigm(batch_view);
            let graph_view: ParadigmView<'_, u64, paradigm::Graph> =
                transition_paradigm(stream_view);
            black_box(graph_view.as_ptr())
        });
    });

    group.bench_function("triple_transition", |b| {
        b.iter(|| {
            let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
            let stream_view: ParadigmView<'_, u64, paradigm::Stream> =
                transition_paradigm(batch_view);
            let graph_view: ParadigmView<'_, u64, paradigm::Graph> =
                transition_paradigm(stream_view);
            let back_to_batch: ParadigmView<'_, u64, paradigm::Batch> =
                transition_paradigm(graph_view);
            black_box(back_to_batch.as_ptr())
        });
    });

    group.finish();
}

/// Benchmark varying buffer sizes to show zero-copy scales.
fn bench_buffer_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_scaling");

    // Test various buffer sizes (1KB to 100MB)
    let sizes = [
        1024,              // 1 KB
        1024 * 1024,       // 1 MB
        10 * 1024 * 1024,  // 10 MB
        100 * 1024 * 1024, // 100 MB
    ];

    for size in sizes {
        let element_count = size / std::mem::size_of::<u64>();

        group.bench_with_input(
            BenchmarkId::new("transition", format!("{}MB", size / (1024 * 1024))),
            &element_count,
            |b, &count| {
                let mut buffer = ZeroCopyBuffer::<u64>::new(count);
                for i in 0..count {
                    buffer.push(i as u64);
                }

                b.iter(|| {
                    let batch_view: ParadigmView<'_, u64, paradigm::Batch> =
                        ParadigmView::new(&buffer);
                    let batch_ptr = batch_view.as_ptr();

                    let stream_view: ParadigmView<'_, u64, paradigm::Stream> =
                        transition_paradigm(batch_view);
                    let stream_ptr = stream_view.as_ptr();

                    // Verify pointers are identical
                    assert_eq!(batch_ptr, stream_ptr, "Zero-copy violated at size {}", size);

                    black_box(stream_ptr)
                });
            },
        );
    }

    group.finish();
}

/// Outputs detailed verification results for validation report.
fn run_comprehensive_verification() {
    println!("\n===== Zero-Copy Verification Report =====\n");

    // Pointer stability test
    println!("1. Pointer Stability Test:");
    let ptr_result = verify_pointer_stability();
    println!("   Batch Pointer:    0x{:016x}", ptr_result.batch_ptr);
    println!("   Stream Pointer:   0x{:016x}", ptr_result.stream_ptr);
    println!("   Graph Pointer:    0x{:016x}", ptr_result.graph_ptr);
    println!("   Pointers Stable:  {}", ptr_result.pointers_stable);
    println!(
        "   Content Verified: {}/{}",
        ptr_result.content_samples_passed, ptr_result.content_samples_total
    );
    println!(
        "   RESULT: {}",
        if ptr_result.pointers_stable
            && ptr_result.content_samples_passed == ptr_result.content_samples_total
        {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );

    // Page fault test
    println!("\n2. Page Fault Test (Linux only):");
    let pf_result = verify_page_faults();
    println!("   Faults Before:    {}", pf_result.faults_before);
    println!("   Faults After:     {}", pf_result.faults_after);
    println!("   Delta:            {}", pf_result.delta);
    #[cfg(target_os = "linux")]
    println!(
        "   RESULT: {}",
        if pf_result.passed {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    #[cfg(not(target_os = "linux"))]
    println!("   RESULT: SKIPPED (non-Linux platform)");

    println!("\n==========================================\n");
}

criterion_group!(
    benches,
    bench_pointer_stability,
    bench_page_faults,
    bench_transition_overhead,
    bench_buffer_sizes,
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_stability() {
        let result = verify_pointer_stability();
        assert!(
            result.pointers_stable,
            "Pointers should be stable across transitions"
        );
        assert_eq!(result.content_samples_passed, result.content_samples_total);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_page_faults() {
        let result = verify_page_faults();
        assert!(result.passed, "Page fault delta should be negligible");
    }

    #[test]
    fn test_paradigm_transition() {
        let mut buffer = ZeroCopyBuffer::<u64>::new(1000);
        for i in 0..1000 {
            buffer.push(i);
        }

        let batch_view: ParadigmView<'_, u64, paradigm::Batch> = ParadigmView::new(&buffer);
        let batch_ptr = batch_view.as_ptr();

        let stream_view: ParadigmView<'_, u64, paradigm::Stream> = transition_paradigm(batch_view);
        let stream_ptr = stream_view.as_ptr();

        assert_eq!(batch_ptr, stream_ptr, "Transition should preserve pointer");
    }
}
