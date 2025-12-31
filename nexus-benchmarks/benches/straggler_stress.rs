//! Straggler Stress Test
//!
//! This benchmark exposes the "bounded memory" violation when a thread stalls.
//! In epoch-based reclamation, if one thread holds a guard indefinitely (e.g.,
//! due to GC pause, network stall, or just sleeping), the epoch cannot advance,
//! blocking garbage collection for ALL threads.
//!
//! # Scenario
//!
//! 1. Spawn 1 "straggler" thread that pins epoch and sleeps for 500ms
//! 2. Main thread attempts to advance epochs during the stall
//! 3. Measure how many epochs advance during the stall
//!
//! # Expected Result
//!
//! - Current implementation: **0 epoch advances** (System Freeze)
//! - Robust implementation (Section 7.3): Should continue advancing
//!
//! # Paper Reference
//!
//! This test exposes the need for "epoch freeze detection" as discussed in
//! Section 7.3 of the paper. Without it, a single stalled thread can halt
//! the entire system's garbage collection.

use nexus_memory::Collector;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Duration for the straggler to sleep (simulating GC pause or network stall)
const STRAGGLER_STALL_MS: u64 = 500;

fn main() {
    eprintln!("Straggler Stress Test: Bounded Memory Violation Detection");
    eprintln!("===========================================================\n");

    let result = run_straggler_test();

    // Output result in a machine-readable format
    println!("test,straggler_stall_ms,epochs_before,epochs_after,epochs_advanced,result");
    println!(
        "straggler_stress,{},{},{},{},{}",
        STRAGGLER_STALL_MS,
        result.epoch_before_stall,
        result.epoch_after_stall,
        result.epochs_advanced,
        if result.system_halted { "FAIL" } else { "PASS" }
    );

    eprintln!();
    if result.system_halted {
        eprintln!("FAIL: System halted by straggler");
        eprintln!("  - Epoch before stall: {}", result.epoch_before_stall);
        eprintln!(
            "  - Epoch after {}ms stall: {}",
            STRAGGLER_STALL_MS, result.epoch_after_stall
        );
        eprintln!("  - Epochs advanced: {}", result.epochs_advanced);
        eprintln!("  - Advance attempts: {}", result.advance_attempts);
        eprintln!();
        eprintln!("  This exposes the bounded memory violation from Section 7.3:");
        eprintln!("  A single stalled thread prevents ALL garbage collection.");
        std::process::exit(1);
    } else {
        eprintln!("PASS: System continued advancing epochs despite straggler");
        eprintln!(
            "  - Epochs advanced during {}ms stall: {}",
            STRAGGLER_STALL_MS, result.epochs_advanced
        );
        std::process::exit(0);
    }
}

struct TestResult {
    epoch_before_stall: u64,
    epoch_after_stall: u64,
    epochs_advanced: u64,
    advance_attempts: u64,
    system_halted: bool,
}

fn run_straggler_test() -> TestResult {
    let collector = Arc::new(Collector::new());
    let straggler_started = Arc::new(AtomicBool::new(false));
    let straggler_done = Arc::new(AtomicBool::new(false));
    let advance_attempts = Arc::new(AtomicU64::new(0));

    // Get initial epoch
    let epoch_before = collector.epoch();
    eprintln!("  [Setup] Initial epoch: {}", epoch_before);

    // Spawn straggler thread
    let straggler_handle = {
        let collector = collector.clone();
        let straggler_started = straggler_started.clone();
        let straggler_done = straggler_done.clone();

        thread::spawn(move || {
            // Pin the epoch - this is the critical moment that blocks advancement
            let guard = collector.pin();
            let pinned_epoch = guard.epoch();

            eprintln!(
                "  [Straggler] Pinned at epoch {}, sleeping for {}ms...",
                pinned_epoch, STRAGGLER_STALL_MS
            );

            // Signal that we've started (epoch is now pinned)
            straggler_started.store(true, Ordering::Release);

            // Simulate a GC pause or network stall
            thread::sleep(Duration::from_millis(STRAGGLER_STALL_MS));

            // Signal done before dropping guard
            straggler_done.store(true, Ordering::Release);

            let final_epoch = guard.epoch();
            eprintln!(
                "  [Straggler] Waking up. Guard epoch: {} (started at {})",
                final_epoch, pinned_epoch
            );

            // Guard is dropped here, allowing epoch to potentially advance
            drop(guard);
        })
    };

    // Wait for straggler to pin epoch
    while !straggler_started.load(Ordering::Acquire) {
        thread::yield_now();
    }

    // Small delay to ensure guard is fully established
    thread::sleep(Duration::from_millis(10));

    let epoch_at_stall_start = collector.epoch();
    eprintln!(
        "  [Monitor] Epoch when straggler started: {}",
        epoch_at_stall_start
    );

    // Try to advance epochs during the stall
    let start_time = Instant::now();
    let mut attempts = 0u64;

    while !straggler_done.load(Ordering::Acquire) {
        // Pin and immediately unpin to allow advancement
        {
            let guard = collector.pin();
            guard.flush(); // Try to trigger GC
            drop(guard);
        }

        attempts += 1;

        // Yield to allow other threads to run
        thread::yield_now();
    }

    advance_attempts.store(attempts, Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    // Wait for straggler to complete
    straggler_handle.join().unwrap();

    // Get final epoch
    let epoch_after = collector.epoch();
    let epochs_advanced = epoch_after.saturating_sub(epoch_at_stall_start);

    eprintln!(
        "  [Monitor] Final epoch: {} (after {:.1}ms, {} attempts)",
        epoch_after,
        elapsed.as_secs_f64() * 1000.0,
        attempts
    );

    TestResult {
        epoch_before_stall: epoch_at_stall_start,
        epoch_after_stall: epoch_after,
        epochs_advanced,
        advance_attempts: attempts,
        // If zero epochs advanced during 500ms with many attempts, system is halted
        system_halted: epochs_advanced == 0 && attempts > 100,
    }
}
