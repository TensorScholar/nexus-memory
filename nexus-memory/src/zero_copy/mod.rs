//! Zero-Copy Buffer Management
//!
//! This module implements zero-copy buffer semantics for cross-paradigm data sharing.
//! The key innovation is enabling batch, stream, and graph processing to access
//! the same underlying memory without copying, while maintaining type and memory safety.
//!
//! # Design Goals
//!
//! 1. **Zero Copies**: Data remains in place during paradigm transitions
//! 2. **Type Safety**: Compile-time guarantees prevent invalid access patterns  
//! 3. **Lifetime Tracking**: Phantom lifetimes ensure references don't outlive data
//! 4. **Cache Efficiency**: Buffer layouts optimized for sequential and random access
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   ZeroCopyBuffer                         │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
//! │  │ Region 0│  │ Region 1│  │ Region 2│  ...            │
//! │  └────┬────┘  └────┬────┘  └────┬────┘                 │
//! │       │            │            │                       │
//! │       ▼            ▼            ▼                       │
//! │  [PhantomLifetime tracking ensures safety]              │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use nexus_memory::zero_copy::{ZeroCopyBuffer, ZeroCopyRef};
//!
//! // Create a buffer
//! let mut buffer = ZeroCopyBuffer::<i32>::new(1024);
//!
//! // Write data
//! buffer.write(0, &[1, 2, 3, 4, 5]);
//!
//! // Get zero-copy reference for batch processing
//! let batch_ref = buffer.as_ref();
//! assert_eq!(batch_ref.len(), 1024);
//!
//! // Get same data for stream processing - no copy!
//! let stream_ref = buffer.as_ref();
//! ```

mod buffer;
mod phantom;

pub use buffer::{ZeroCopyBuffer, ZeroCopyBufferBuilder};
pub use phantom::{PhantomLifetime, ZeroCopyRef, ZeroCopyMut};

use core::marker::PhantomData;
use core::ptr::NonNull;

/// Error types for zero-copy operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroCopyError {
    /// Buffer allocation failed
    AllocationFailed,
    
    /// Invalid alignment for the requested type
    InvalidAlignment,
    
    /// Attempted to borrow while already mutably borrowed
    BorrowConflict,
    
    /// Index out of bounds
    OutOfBounds {
        index: usize,
        len: usize,
    },
    
    /// Buffer is not large enough
    InsufficientCapacity {
        required: usize,
        available: usize,
    },
    
    /// Invalid region specification
    InvalidRegion,
}

impl core::fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZeroCopyError::AllocationFailed => write!(f, "buffer allocation failed"),
            ZeroCopyError::InvalidAlignment => write!(f, "invalid memory alignment"),
            ZeroCopyError::BorrowConflict => write!(f, "buffer already mutably borrowed"),
            ZeroCopyError::OutOfBounds { index, len } => {
                write!(f, "index {} out of bounds for buffer of length {}", index, len)
            }
            ZeroCopyError::InsufficientCapacity { required, available } => {
                write!(f, "insufficient capacity: required {}, available {}", required, available)
            }
            ZeroCopyError::InvalidRegion => write!(f, "invalid region specification"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ZeroCopyError {}

/// Result type for zero-copy operations
pub type Result<T> = core::result::Result<T, ZeroCopyError>;

/// Paradigm tag for type-level paradigm tracking
pub trait Paradigm: Copy + 'static {
    /// Name of the paradigm for debugging
    const NAME: &'static str;
}

/// Batch processing paradigm marker
#[derive(Debug, Clone, Copy)]
pub struct BatchParadigm;

impl Paradigm for BatchParadigm {
    const NAME: &'static str = "Batch";
}

/// Stream processing paradigm marker
#[derive(Debug, Clone, Copy)]
pub struct StreamParadigm;

impl Paradigm for StreamParadigm {
    const NAME: &'static str = "Stream";
}

/// Graph processing paradigm marker
#[derive(Debug, Clone, Copy)]
pub struct GraphParadigm;

impl Paradigm for GraphParadigm {
    const NAME: &'static str = "Graph";
}

/// A region within a zero-copy buffer
///
/// Regions allow dividing a buffer into logical sections that can be
/// independently accessed and potentially migrated between paradigms.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    /// Start offset in elements
    pub start: usize,
    /// Length in elements  
    pub len: usize,
}

impl Region {
    /// Creates a new region.
    pub const fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }

    /// Returns the end offset (exclusive).
    pub const fn end(&self) -> usize {
        self.start + self.len
    }

    /// Checks if this region overlaps with another.
    pub fn overlaps(&self, other: &Region) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    /// Checks if this region is contained within another.
    pub fn contained_in(&self, other: &Region) -> bool {
        self.start >= other.start && self.end() <= other.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_overlap() {
        let r1 = Region::new(0, 10);
        let r2 = Region::new(5, 10);
        let r3 = Region::new(10, 5);
        
        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_region_contained() {
        let outer = Region::new(0, 100);
        let inner = Region::new(10, 20);
        let outside = Region::new(90, 20);
        
        assert!(inner.contained_in(&outer));
        assert!(!outside.contained_in(&outer));
    }
}
