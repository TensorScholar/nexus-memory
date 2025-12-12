//! Conditional Synchronization Primitives
//!
//! This module provides a unified interface for atomic types that works with
//! both standard library atomics and Loom's model-checking atomics.
//!
//! When the `loom` feature is enabled, this module re-exports Loom's atomics
//! which enable exhaustive concurrency testing via model checking.
//!
//! # Loom Integration
//!
//! Loom is a testing tool for concurrent Rust code that explores all possible
//! interleavings of concurrent operations. To use Loom testing:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --features loom
//! ```
//!
//! # Standard Mode
//!
//! In standard mode (without `loom` feature), this module simply re-exports
//! from `core::sync::atomic`, providing zero overhead.

#[cfg(all(feature = "loom", loom))]
pub mod atomic {
    //! Atomic types for Loom model checking.
    //!
    //! These re-exports provide Loom's atomics which track all memory orderings
    //! and enable exhaustive interleaving exploration.

    pub use loom::sync::atomic::{
        fence, AtomicBool, AtomicIsize, AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicU8,
        AtomicUsize, Ordering,
    };

    // Loom doesn't model compiler_fence, so we use a regular fence for model checking
    // which is more conservative but correct for verification purposes.
    pub use loom::sync::atomic::fence as compiler_fence;
}

#[cfg(not(all(feature = "loom", loom)))]
pub mod atomic {
    //! Standard library atomic types.
    //!
    //! In non-Loom mode, these are zero-cost re-exports from `core::sync::atomic`.

    pub use core::sync::atomic::{
        compiler_fence, fence, AtomicBool, AtomicIsize, AtomicPtr, AtomicU16, AtomicU32, AtomicU64,
        AtomicU8, AtomicUsize, Ordering,
    };
}

#[cfg(all(feature = "loom", loom))]
pub mod cell {
    //! Loom-aware cell types.
    //!
    //! Loom's UnsafeCell has a different API than std's - it returns
    //! ConstPtr/MutPtr wrappers instead of raw pointers. We provide
    //! helper functions to abstract this difference.

    pub use loom::cell::UnsafeCell;

    /// Get a mutable raw pointer from an UnsafeCell.
    ///
    /// In Loom mode, this uses `with_mut` to get a tracked raw pointer.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to the cell's contents.
    #[inline]
    pub unsafe fn get_mut_ptr<T>(cell: &UnsafeCell<T>) -> *mut T {
        cell.get_mut().deref()
    }

    /// Access the contents of an UnsafeCell mutably through a closure.
    ///
    /// This is the preferred way to access UnsafeCell contents in Loom.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to the cell's contents.
    #[inline]
    pub unsafe fn with_mut<T, R>(cell: &UnsafeCell<T>, f: impl FnOnce(&mut T) -> R) -> R {
        cell.with_mut(|ptr| f(&mut *ptr))
    }
}

#[cfg(not(all(feature = "loom", loom)))]
pub mod cell {
    //! Standard library cell types.

    pub use core::cell::UnsafeCell;

    /// Get a mutable raw pointer from an UnsafeCell.
    ///
    /// In standard mode, this is just `cell.get()`.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to the cell's contents.
    #[inline]
    pub unsafe fn get_mut_ptr<T>(cell: &UnsafeCell<T>) -> *mut T {
        cell.get()
    }

    /// Access the contents of an UnsafeCell mutably through a closure.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to the cell's contents.
    #[inline]
    pub unsafe fn with_mut<T, R>(cell: &UnsafeCell<T>, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut *cell.get())
    }
}

#[cfg(all(feature = "loom", loom))]
pub mod thread {
    //! Loom thread primitives for model checking.

    pub use loom::thread::{spawn, yield_now, Builder, JoinHandle};

    /// Runs a Loom model checking session.
    ///
    /// This function explores all possible thread interleavings to verify
    /// correctness of concurrent code.
    pub fn model<F>(f: F)
    where
        F: Fn() + Sync + Send + 'static,
    {
        loom::model(f);
    }
}

#[cfg(not(all(feature = "loom", loom)))]
pub mod thread {
    //! Standard library thread primitives.

    #[cfg(feature = "std")]
    pub use std::thread::{spawn, yield_now, Builder, JoinHandle};

    /// No-op in non-Loom mode.
    ///
    /// When not using Loom, this simply runs the function once.
    #[cfg(feature = "std")]
    pub fn model<F>(f: F)
    where
        F: Fn() + Sync + Send + 'static,
    {
        f();
    }
}

#[cfg(all(feature = "loom", loom))]
pub mod sync {
    //! Loom-aware synchronization primitives.

    pub use loom::sync::{Arc, Condvar, Mutex, RwLock};
}

#[cfg(not(all(feature = "loom", loom)))]
pub mod sync {
    //! Standard library synchronization primitives.

    #[cfg(feature = "std")]
    pub use std::sync::{Arc, Condvar, Mutex, RwLock};

    #[cfg(not(feature = "std"))]
    extern crate alloc;

    #[cfg(not(feature = "std"))]
    pub use alloc::sync::Arc;
}

/// Helper macro for creating thread-local storage that works with Loom.
///
/// When using Loom, thread-local storage needs special handling since
/// Loom simulates multiple threads within a single OS thread.
#[macro_export]
macro_rules! loom_thread_local {
    ($(#[$attr:meta])* $vis:vis static $name:ident: $ty:ty = $init:expr;) => {
        #[cfg(all(feature = "loom", loom))]
        loom::thread_local! {
            $(#[$attr])*
            $vis static $name: $ty = $init;
        }

        #[cfg(not(all(feature = "loom", loom)))]
        std::thread_local! {
            $(#[$attr])*
            $vis static $name: $ty = $init;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_atomic_basic() {
        let counter = AtomicUsize::new(0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        counter.store(42, Ordering::SeqCst);
        assert_eq!(counter.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_atomic_cas() {
        let counter = AtomicUsize::new(10);
        let result = counter.compare_exchange(10, 20, Ordering::SeqCst, Ordering::SeqCst);
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 20);
    }
}
