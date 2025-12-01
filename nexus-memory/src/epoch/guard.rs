//! Epoch Guard Implementation
//!
//! Guards provide RAII-based epoch pinning. While a guard exists, the thread
//! is considered active in the epoch, preventing garbage collection of data
//! that might be accessed through the guard.
//!
//! # Lifetime Safety
//!
//! The guard's lifetime is used to bound `Shared<'g, T>` pointers, ensuring
//! that shared references cannot outlive the protection window.
//!
//! # Memory Ordering
//!
//! Guards use SeqCst ordering for epoch updates to ensure proper synchronization
//! with the collector's epoch advancement logic.

use core::sync::atomic::Ordering;
use core::marker::PhantomData;

use super::{Collector, Participant, INACTIVE};

/// A guard that pins the current thread in an epoch.
///
/// `Guard` is an RAII structure that represents a thread's participation in
/// the current epoch. While a guard exists, any data protected by epoch-based
/// reclamation is guaranteed to remain valid.
///
/// # Lifetime Bounds
///
/// The guard's lifetime `'_` bounds all `Shared<'_, T>` pointers created
/// through it, ensuring memory safety without runtime checks.
///
/// # Drop Behavior
///
/// When dropped, the guard decrements the pin count and potentially marks
/// the participant as inactive, allowing epoch advancement.
///
/// # Example
///
/// ```rust
/// use nexus_memory::Collector;
///
/// let collector = Collector::new();
///
/// {
///     let guard = collector.pin();
///     // Protected region - data is safe to access
///     
///     // Guard dropped here, exiting the protected region
/// }
///
/// // Outside the protected region
/// ```
pub struct Guard<'a> {
    /// Reference to the collector
    collector: &'a Collector,
    
    /// Reference to this thread's participant
    participant: &'a Participant,
    
    /// Marker to prevent Send/Sync
    _marker: PhantomData<*mut ()>,
}

impl<'a> Guard<'a> {
    /// Creates a new guard for the given collector and participant.
    #[inline]
    pub(crate) fn new(collector: &'a Collector, participant: &'a Participant) -> Self {
        Self {
            collector,
            participant,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the collector.
    #[inline]
    pub fn collector(&self) -> &Collector {
        self.collector
    }

    /// Returns the current epoch as observed by this guard.
    #[inline]
    pub fn epoch(&self) -> super::Epoch {
        self.participant.epoch.load(Ordering::SeqCst)
    }

    /// Refreshes the guard to observe the current epoch.
    ///
    /// This is useful for long-running operations that want to allow
    /// garbage collection to proceed while still maintaining protection.
    #[inline]
    pub fn refresh(&self) {
        let epoch = self.collector.global_epoch.load(Ordering::SeqCst);
        self.participant.epoch.store(epoch, Ordering::SeqCst);
    }

    /// Defers destruction of an object until the guard is dropped.
    ///
    /// The object will be destroyed in a future epoch when it's safe to do so.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and the caller must have ownership of the object.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use nexus_memory::Collector;
    ///
    /// let collector = Collector::new();
    /// let guard = collector.pin();
    ///
    /// // Allocate a boxed value
    /// let boxed = Box::new(42i32);
    /// let ptr = Box::into_raw(boxed);
    ///
    /// // Defer its destruction - the guard ensures it won't be freed
    /// // while other threads might still access it
    /// unsafe {
    ///     guard.defer_destroy(ptr);
    /// }
    /// ```
    #[inline]
    pub unsafe fn defer_destroy<T>(&self, ptr: *mut T) {
        // SAFETY: Caller guarantees ownership and validity
        unsafe { self.collector.defer(ptr) };
    }

    /// Flushes any thread-local garbage to the global collector.
    ///
    /// This can be useful before a long blocking operation to ensure
    /// garbage can be collected by other threads.
    pub fn flush(&self) {
        // SAFETY: We have access to the participant through the guard
        let bag = unsafe { &mut *self.participant.local_garbage.get() };
        
        if !bag.is_empty() {
            // Move garbage to global bags
            // For simplicity, we just collect immediately in this implementation
            unsafe { bag.collect() };
        }
    }

    /// Creates a new owned pointer within this epoch.
    ///
    /// This is a convenience method for creating new protected data.
    #[inline]
    pub fn alloc<T>(&self, data: T) -> super::Owned<T> {
        super::Owned::new(data)
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        // Decrement pin count
        let prev = self.participant.pin_count.fetch_sub(1, Ordering::Relaxed);
        
        // If this was the last pin, mark as inactive
        if prev == 1 {
            self.participant.epoch.store(INACTIVE, Ordering::SeqCst);
        }
    }
}

// Guard is not Send or Sync because it represents a thread-local pinning
// This is enforced by the PhantomData<*mut ()> marker

/// An unprotected reference that bypasses epoch checking.
///
/// `Unprotected` provides a way to access epoch-protected data without
/// actually pinning the current thread. This is useful for initialization
/// and cleanup code that runs when no other threads could be accessing
/// the data structure.
///
/// # Safety
///
/// Using `Unprotected` is inherently unsafe because it bypasses the
/// memory reclamation safety guarantees. It should only be used when
/// you can guarantee that:
///
/// 1. No other thread could be reclaiming memory, OR
/// 2. You don't care about the validity of the data (e.g., during shutdown)
///
/// # Example
///
/// ```rust
/// use nexus_memory::epoch::Unprotected;
///
/// // During initialization when we know we're single-threaded
/// let unprotected = unsafe { Unprotected::new() };
///
/// // Can now access data without pinning
/// ```
pub struct Unprotected {
    _marker: PhantomData<*mut ()>,
}

impl Unprotected {
    /// Creates a new unprotected reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that either:
    /// - No concurrent memory reclamation is possible, OR
    /// - Any accessed data is known to be valid
    #[inline]
    pub const unsafe fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Converts to a guard-like lifetime for API compatibility.
    ///
    /// # Safety
    ///
    /// Same safety requirements as `new()`.
    #[inline]
    pub unsafe fn as_guard<'a>(&self) -> UnprotectedGuard<'a> {
        UnprotectedGuard {
            _marker: PhantomData,
        }
    }
}

/// A fake guard for unprotected access.
///
/// This type mimics the API of `Guard` but provides no actual protection.
/// It's used for code that needs to work both in protected and unprotected
/// contexts.
pub struct UnprotectedGuard<'a> {
    _marker: PhantomData<&'a ()>,
}

impl UnprotectedGuard<'_> {
    /// Fake epoch for compatibility.
    #[inline]
    pub fn epoch(&self) -> super::Epoch {
        0
    }
}

/// A scoped guard that provides additional lifetime control.
///
/// `ScopedGuard` is similar to `Guard` but takes a closure, ensuring
/// that all operations are completed before the guard is released.
/// This can help prevent accidental lifetime extension of protected references.
pub struct ScopedGuard<'a, F>
where
    F: FnOnce(&Guard<'a>),
{
    guard: Guard<'a>,
    callback: Option<F>,
}

impl<'a, F> ScopedGuard<'a, F>
where
    F: FnOnce(&Guard<'a>),
{
    /// Creates a new scoped guard with a cleanup callback.
    pub fn new(collector: &'a Collector, callback: F) -> Self {
        Self {
            guard: collector.pin(),
            callback: Some(callback),
        }
    }

    /// Gets a reference to the underlying guard.
    pub fn guard(&self) -> &Guard<'a> {
        &self.guard
    }
}

impl<'a, F> Drop for ScopedGuard<'a, F>
where
    F: FnOnce(&Guard<'a>),
{
    fn drop(&mut self) {
        if let Some(callback) = self.callback.take() {
            callback(&self.guard);
        }
    }
}

/// Helper trait for guard-like types.
///
/// This trait allows generic code to work with both `Guard` and
/// `UnprotectedGuard` types.
pub trait Guarded {
    /// Returns whether this is a real (protected) guard.
    fn is_protected(&self) -> bool;
}

impl Guarded for Guard<'_> {
    #[inline]
    fn is_protected(&self) -> bool {
        true
    }
}

impl Guarded for UnprotectedGuard<'_> {
    #[inline]
    fn is_protected(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_creation() {
        let collector = Collector::new();
        let guard = collector.pin();
        
        assert!(guard.epoch() < INACTIVE);
    }

    #[test]
    fn test_guard_refresh() {
        let collector = Collector::new();
        let guard = collector.pin();
        
        let epoch1 = guard.epoch();
        guard.refresh();
        let epoch2 = guard.epoch();
        
        // Epoch should be same or higher after refresh
        assert!(epoch2 >= epoch1);
    }

    #[test]
    fn test_unprotected() {
        // SAFETY: Test environment, no concurrency
        let unprotected = unsafe { Unprotected::new() };
        
        // SAFETY: Same as above
        let guard = unsafe { unprotected.as_guard() };
        assert_eq!(guard.epoch(), 0);
    }

    #[test]
    fn test_guarded_trait() {
        let collector = Collector::new();
        let guard = collector.pin();
        
        assert!(guard.is_protected());
        
        // SAFETY: Test environment
        let unprotected = unsafe { Unprotected::new() };
        let unprotected_guard = unsafe { unprotected.as_guard() };
        
        assert!(!unprotected_guard.is_protected());
    }

    #[test]
    fn test_guard_flush() {
        let collector = Collector::new();
        let guard = collector.pin();
        
        // Flush should not panic even with no garbage
        guard.flush();
    }
}
