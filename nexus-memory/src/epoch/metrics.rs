//! Epoch Synchronization Metrics
//!
//! Provides thread-local metrics for measuring synchronization overhead.
//! Only active when `bench-metrics` feature is enabled.

use std::cell::RefCell;
use std::time::Duration;

#[derive(Debug, Default, Clone, Copy)]
#[allow(missing_docs)]
pub struct LatencyMetrics {
    /// Total time spent in pin()
    pub pin_nanos: u64,
    /// Total time spent in try_advance()
    pub advance_nanos: u64,
    /// Number of operations performed
    pub ops_count: u64,
}

thread_local! {
    /// Thread-local metrics storage
    pub static METRICS: RefCell<LatencyMetrics> = RefCell::new(LatencyMetrics::default());
}

/// Records latency for a pin operation
#[inline]
#[allow(dead_code)] // May be unused if instrumentation is conditional
pub fn record_pin(duration: Duration) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.pin_nanos += duration.as_nanos() as u64;
    });
}

/// Records latency for an advance operation
#[inline]
#[allow(dead_code)]
pub fn record_advance(duration: Duration) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.advance_nanos += duration.as_nanos() as u64;
        metrics.ops_count += 1;
    });
}

/// Returns the current metrics and resets them
pub fn report_metrics() -> LatencyMetrics {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        let current = *metrics;
        *metrics = LatencyMetrics::default();
        current
    })
}
