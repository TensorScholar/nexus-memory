//! Nexus Memory: Hierarchical Epoch-Based Memory Reclamation for Cross-Paradigm Processing
//!
//! This crate implements a novel memory reclamation scheme that enables safe, efficient
//! memory management across batch, stream, and graph processing paradigms without
//! cross-paradigm data copying.
//!
//! # Architecture
//!
//! The system is organized into three core modules:
//! - [`epoch`]: Hierarchical epoch-based garbage collection with wait-free progress
//! - [`zero_copy`]: Zero-copy buffer management with phantom lifetime tracking  
//! - [`numa`]: NUMA-aware memory allocation and topology optimization
//!
//! # Mathematical Foundation
//!
//! Memory safety is guaranteed through the epoch invariant:
//! ```text
//! ∀p ∈ Pointers, e ∈ Epochs: valid(p, e) ⟹ ¬freed(p) ∧ epoch(p) ≤ e
//! ```
//!
//! The hierarchical epoch structure provides O(log T) global synchronization where
//! T is the number of threads, improving upon the O(T) bound of flat epoch schemes.
//!
//! # Performance Characteristics
//!
//! - Cross-paradigm transitions: O(log n) overhead
//! - Memory reclamation latency: O(1) amortized
//! - Space overhead: O(T × G) where T = threads, G = garbage per epoch
//!
//! # Example
//!
//! ```rust
//! use nexus_memory::{Collector, Guard, Owned, Shared};
//!
//! // Create a collector for managing memory
//! let collector = Collector::new();
//!
//! // Pin the current thread to enter a protected region
//! let guard = collector.pin();
//!
//! // Allocate and share data safely
//! let owned = Owned::new(42);
//! let shared: Shared<'_, i32> = owned.into_shared(&guard);
//!
//! // Access data through the guard
//! unsafe {
//!     assert_eq!(*shared.deref(), 42);
//! }
//! ```
//!
//! # Feature Flags
//!
//! - `std` (default): Enable standard library support
//! - `numa`: Enable NUMA-aware allocation
//! - `zero-copy`: Enable zero-copy buffer management
//! - `statistics`: Enable runtime statistics collection
//!
//! # References
//!
//! - Keir Fraser. "Practical Lock-Freedom." PhD thesis, Cambridge, 2004.
//! - Hart et al. "Making Lockless Synchronization Fast." ACM TOCS, 2007.
//! - McKenney et al. "RCU: Usage and Correctness." Linux Weekly News, 2007.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod epoch;

#[cfg(feature = "zero-copy")]
pub mod zero_copy;

#[cfg(feature = "numa")]
pub mod numa;

// Re-export main types at crate root for convenience
pub use epoch::{Collector, Guard, Owned, Shared};

#[cfg(feature = "zero-copy")]
pub use zero_copy::{ZeroCopyBuffer, ZeroCopyRef};

#[cfg(feature = "numa")]
pub use numa::{NumaAllocator, NumaNode};

/// Error types for the nexus-memory crate
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Memory allocation failed
    AllocationFailed,
    /// Invalid memory alignment
    InvalidAlignment,
    /// Epoch not properly pinned
    EpochNotPinned,
    /// Buffer already borrowed
    BorrowConflict,
    /// NUMA topology unavailable
    #[cfg(feature = "numa")]
    NumaUnavailable,
    /// Invalid NUMA node
    #[cfg(feature = "numa")]
    InvalidNode(u32),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::AllocationFailed => write!(f, "memory allocation failed"),
            Error::InvalidAlignment => write!(f, "invalid memory alignment"),
            Error::EpochNotPinned => write!(f, "thread epoch not properly pinned"),
            Error::BorrowConflict => write!(f, "buffer already borrowed"),
            #[cfg(feature = "numa")]
            Error::NumaUnavailable => write!(f, "NUMA topology unavailable"),
            #[cfg(feature = "numa")]
            Error::InvalidNode(n) => write!(f, "invalid NUMA node: {}", n),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Result type for nexus-memory operations
pub type Result<T> = core::result::Result<T, Error>;

/// Compile-time configuration constants
pub mod config {
    /// Maximum number of threads supported
    pub const MAX_THREADS: usize = 256;
    
    /// Number of epochs before garbage collection is triggered
    pub const EPOCHS_PER_GC: usize = 128;
    
    /// Cache line size for alignment
    pub const CACHE_LINE_SIZE: usize = 64;
    
    /// Default NUMA node count limit
    #[cfg(feature = "numa")]
    pub const MAX_NUMA_NODES: usize = 64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let collector = Collector::new();
        let guard = collector.pin();
        
        let owned = Owned::new(42i32);
        let shared = owned.into_shared(&guard);
        
        unsafe {
            assert_eq!(*shared.deref(), 42);
        }
    }

    #[test]
    fn test_multiple_guards() {
        let collector = Collector::new();
        
        let guard1 = collector.pin();
        let guard2 = collector.pin();
        
        let owned = Owned::new("hello");
        let shared = owned.into_shared(&guard1);
        
        unsafe {
            assert_eq!(*shared.deref(), "hello");
        }
        
        drop(guard2);
        drop(guard1);
    }
}
